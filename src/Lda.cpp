#ifndef ROSNEURO_DECODER_LDA_CPP
#define ROSNEURO_DECODER_LDA_CPP

#include "rosneuro_decoder_lda/Lda.hpp"

namespace rosneuro{
    namespace decoder{
        Lda::Lda(void) : p_nh_("~"){
            this->setName("lda");
            this->is_configured_ = false;
        }

        Lda::~Lda(void){}

        template<typename T>
        bool Lda::getParamAndCheck(const std::string& param_name, T& param_value) {
            if (!GenericDecoder::getParam(param_name, param_value)) {
                ROS_ERROR("[%s] Cannot find param '%s'", this->getName().c_str(), param_name.c_str());
                return false;
            }
            return true;
        }

        bool Lda::configure(void){
            if (!getParamAndCheck("filename", this->config_.filename)) return false;
            if (!getParamAndCheck("subject", this->config_.subject)) return false;
            if (!getParamAndCheck("n_classes", this->config_.n_classes)) return false;
            if (!getParamAndCheck("class_lbs", this->config_.class_lbs)) return false;
            if (!getParamAndCheck("n_features", this->config_.n_features)) return false;
            if (!getParamAndCheck("lambda", this->config_.lambda)) return false;
            if (!getParamAndCheck("idchans", this->config_.idchans)) return false;
            if (!getParamAndCheck("priors", this->config_.priors)) return false;

            std::string means_str, covs_str, freqs_str;

            if (!getParamAndCheck("freqs", freqs_str)) return false;
            if(!this->loadVectorOfVector(freqs_str, this->config_.freqs)){
                ROS_ERROR("[%s] Cannot convert param 'freqs' to vector of vector", this->getName().c_str());
                return false;
            }

            if (!getParamAndCheck("means", means_str)) return false;
            this->means_ = Eigen::MatrixXf::Zero(this->config_.n_features, this->config_.n_classes);
            if(!this->loadEigen(means_str, this->means_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for means", this->getName().c_str());
                return false;
            }

            if (!getParamAndCheck("covs", covs_str)) return false;
            this->covs_ = Eigen::MatrixXf::Zero(this->config_.n_features, this->config_.n_features);
            if(!this->loadEigen(covs_str, this->covs_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for covs", this->getName().c_str());
                return false;
            }

            if(!this->checkDimension()){
                ROS_ERROR("[%s] Error in the dimension", this->getName().c_str());
                return false;
            }

            this->is_configured_ = true;
            return this->is_configured_;
        }

        bool Lda::isSet(void){
            if(!this->is_configured_){
                ROS_ERROR("[%s] Decoder not configured", this->getName().c_str());
                return false;
            }
            return this->is_configured_;
        }

        Eigen::VectorXf Lda::apply(const Eigen::VectorXf& input) {
            double normalization_coefficient = calculateNormalizationCoefficient(input.size());

            std::vector<double> likelihoods;
            double posterior_denominator = calculatePosteriorDenominator(input, likelihoods);

            Eigen::VectorXf posterior_probabilities = computePosteriorProbabilities(likelihoods, posterior_denominator);

            return posterior_probabilities;
        }

        double Lda::calculateNormalizationCoefficient(int input_size) const {
            double determinant = this->covs_.determinant();
            double coefficient = 1 / std::sqrt(determinant * pow(2 * M_PI, input_size));
            return coefficient;
        }

        double Lda::calculateLikelihood(const Eigen::VectorXf& input, int class_index) const {
            Eigen::VectorXf mean_difference = input - this->means_.col(class_index);
            Eigen::MatrixXf cov_inverse = this->covs_.inverse();
            double exponent = -0.5 * (mean_difference.transpose() * cov_inverse * mean_difference)(0, 0);
            double likelihood = std::exp(exponent);
            return likelihood;
        }

        double Lda::calculatePosteriorDenominator(const Eigen::VectorXf& input, std::vector<double>& likelihoods) const {
            double posterior_denominator = 0.0;
            for (int i = 0; i < this->config_.n_classes; i++) {
                double likelihood = calculateLikelihood(input, i);
                likelihoods.push_back(likelihood);
                posterior_denominator += likelihood * this->config_.priors.at(i);
            }
            return posterior_denominator;
        }

        Eigen::VectorXf Lda::computePosteriorProbabilities(const std::vector<double>& likelihoods, double posterior_denominator) const {
            Eigen::VectorXf posterior_probabilities(likelihoods.size(), 1);
            for (int i = 0; i < likelihoods.size(); i++) {
                double posterior = (likelihoods.at(i) * this->config_.priors.at(i)) / posterior_denominator;
                posterior_probabilities(i, 0) = posterior;
            }
            return posterior_probabilities;
        }

        std::string Lda::getPath(void){
            this->isSet();
            return this->config_.filename;
        }

        std::vector<int> Lda::getClasses(void){
            this->isSet();
            std::vector<int> classes_lbs;
            for(int i = 0; i < this->config_.class_lbs.size(); i++){
                classes_lbs.push_back((int) this->config_.class_lbs.at(i));
            }
            return classes_lbs;
        }

        Eigen::VectorXf Lda::getFeatures(const Eigen::MatrixXf& input) {
            this->isSet();
            Eigen::VectorXf features(this->config_.n_features);

            int feature_index = 0;
            for (int channel_index = 0; channel_index < this->config_.idchans.size(); channel_index++) {
                int channel_id = this->config_.idchans.at(channel_index) - 1;
                for (const auto& frequency : this->config_.freqs.at(channel_index)) {
                    int frequency_id = static_cast<int>(frequency / 2.0);
                    features(feature_index) = input(channel_id, frequency_id);
                    feature_index++;
                }
            }
            return features.transpose();
        }

        bool Lda::checkDimension(void){
            if(this->means_.rows() != this->config_.n_features ||
               this->means_.cols() != this->config_.n_classes){
                ROS_ERROR("[%s] Wrong dimensions in the 'means' parameter", this->getName().c_str());
                return false;
            }

            if(this->covs_.rows() != this->config_.n_features ||
               this->covs_.cols() != this->config_.n_features){
                ROS_ERROR("[%s] Wrong dimensions in the 'covs' parameter", this->getName().c_str());
                return false;
            }

            if(this->config_.priors.size() != this->config_.n_classes |\
               this->config_.n_classes != this->config_.class_lbs.size()){
                ROS_ERROR("[%s] Wrong dimensions in the given classes parameters", this->getName().c_str());
                return false;
            }

            int sum = 0;
            for(int i = 0; i < this->config_.freqs.size(); i++){
                std::vector<uint32_t> temp = this->config_.freqs.at(i);
                sum = sum + temp.size();
            } 
            if(sum != this->config_.n_features){
                ROS_ERROR("[%s] Incorrect dimension for 'freqs' different from 'n_features'", this->getName().c_str());
                return false;
            }

            return true;
        }

        PLUGINLIB_EXPORT_CLASS(rosneuro::decoder::Lda, rosneuro::decoder::GenericDecoder);
    }
}

#endif