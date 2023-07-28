#ifndef ROSNEURO_DECODER_LDA_CPP
#define ROSNEURO_DECODER_LDA_CPP

#include "rosneuro_decoder_lda/Lda.hpp"

namespace rosneuro{
    namespace decoder{

        Lda::Lda(void) : p_nh_("~"){
            this->setname("lda");
            this->is_configured_ = false;
        }

        Lda::~Lda(void){};

        bool Lda::configure(void){
            // get the parameters
            if(!GenericDecoder::getParam(std::string("filename"), this->config_.filename)){
                ROS_ERROR("[%s] Cannot find param 'filename'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("subject"), this->config_.subject)){
                ROS_ERROR("[%s] Cannot find param 'subject'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("nclasses"), this->config_.nclasses)){
                ROS_ERROR("[%s] Cannot find param 'nclasses'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("classlbs"), this->config_.classlbs)){
                ROS_ERROR("[%s] Cannot find param 'classlbs'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("nfeatures"), this->config_.nfeatures)){
                ROS_ERROR("[%s] Cannot find param 'nfeatures'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("lambda"), this->config_.lambda)){
                ROS_ERROR("[%s] Cannot find param 'lambda'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("idchans"), this->config_.idchans)){
                ROS_ERROR("[%s] Cannot find param 'idchans'", this->name().c_str());
                return false;
            }
            std::string freqs_str;
            if(!GenericDecoder::getParam(std::string("freqs"), freqs_str)){
                ROS_ERROR("[%s] Cannot find param 'freqs'", this->name().c_str());
                return false;
            }
            if(!this->load_vectorOfVector(freqs_str, this->config_.freqs)){
                ROS_ERROR("[%s] Cannot convert param 'freqs' to vctor of vector", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("priors"), this->config_.priors)){
                ROS_ERROR("[%s] Cannot find param 'priors'", this->name().c_str());
                return false;
            }
            std::string means_str, covs_str;
            if(!GenericDecoder::getParam(std::string("means"), means_str)){
                ROS_ERROR("[%s] Cannot find param 'means'", this->name().c_str());
                return false;
            }
            this->means_ = Eigen::MatrixXf::Zero(this->config_.nfeatures, this->config_.nclasses);
            if(!this->load_eigen(means_str, this->means_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for means", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("covs"), covs_str)){
                ROS_ERROR("[%s] Cannot find param 'covs'", this->name().c_str());
                return false;
            }
            this->covs_ = Eigen::MatrixXf::Zero(this->config_.nfeatures, this->config_.nfeatures);
            if(!this->load_eigen(covs_str, this->covs_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for covs", this->name().c_str());
                return false;
            }

            // fast check for the correct dimension for the features
            if(!this->check_dimension()){
                ROS_ERROR("[%s] Error in the dimension", this->name().c_str());
                return false;
            }

            // Check if more than 2 classes
            if(this->config_.nclasses != 2 || this->config_.classlbs.size() != 2){
                ROS_ERROR("[%s] The classifier works only for two classes", this->name().c_str());
                return false;
            }

            this->is_configured_ = true;

            return this->is_configured_;
        }

        bool Lda::isSet(void){
            if(this->is_configured_ == false){
                ROS_ERROR("[%s] Decoder not configured", this->name().c_str());
                return false;
            }

            return this->is_configured_;
        }

        Eigen::VectorXf Lda::apply(const Eigen::VectorXf& in){
            double coeff = 1/ std::sqrt(this->covs_.determinant() * pow( 2 * M_PI, in.size()));
            double exp1 = - 0.5 * ((in - this->means_.col(0)).transpose() * this->covs_.inverse() * (in - this->means_.col(0)))(0,0);
            double exp2 = - 0.5 * ((in - this->means_.col(1)).transpose() * this->covs_.inverse() * (in - this->means_.col(1)))(0,0);
            double lh1 = coeff * std::exp(exp1);
            double lh2 = coeff * std::exp(exp2);

            double post1 = lh1 * this->config_.priors.at(0) / (lh1 * this->config_.priors.at(0) + lh2 * this->config_.priors.at(1));
            double post2 = lh2 * this->config_.priors.at(1) / (lh1 * this->config_.priors.at(0) + lh2 * this->config_.priors.at(1));

            Eigen::VectorXf output(2,1);
            if(std::max(post1, post2) == post1){
                output << std::max(post1, post2), 1.0-std::max(post1, post2);
            }else{
                output << 1.0 - std::max(post1, post2), std::max(post1, post2);
            }

            return output;
        }

        std::string Lda::path(void){
            this->isSet();
            return this->config_.filename;
        }

        std::vector<int> Lda::classes(void){
            this->isSet();
            std::vector<int> classeslbs;
            for(int i = 0; i < this->config_.classlbs.size(); i++){
                classeslbs.push_back((int) this->config_.classlbs.at(i));
            }
            return classeslbs;
        }

        Eigen::VectorXf Lda::getFeatures(const Eigen::MatrixXf& in){
            // check if set and prepare for the correct dimension
            Eigen::VectorXf out(this->config_.nfeatures);
            this->isSet();

            // iterate over channels
            int c_feature = 0;
            for(int it_chan = 0; it_chan < this->config_.idchans.size(); it_chan++){
                int idchan = this->config_.idchans.at(it_chan) - 1; // -1 bc: channels starts from 1 and not 0
                // iterate over freqs for that channel
                for(const auto& freq : this->config_.freqs.at(it_chan)){
                    // we have the freq value and not the id of that freq
                    int idfreq = (int) freq/2.0;
                    out(c_feature) = in(idchan, idfreq);
                    c_feature ++;
                }
            }

            return out.transpose();
        }

        bool Lda::check_dimension(void){
            // check the means
            if(this->means_.rows() != this->config_.nfeatures || 
               this->means_.cols() != this->config_.nclasses){
                ROS_ERROR("[%s] Wrong dimensions in the 'means' parameter", this->name().c_str());
                return false;
            }

            // check the cov
            if(this->covs_.rows() != this->config_.nfeatures || 
               this->covs_.cols() != this->config_.nfeatures){
                ROS_ERROR("[%s] Wrong dimensions in the 'covs' parameter", this->name().c_str());
                return false;
            }

            if(this->config_.priors.size() != this->config_.nclasses){
                ROS_ERROR("[%s] Wrong dimensions in the 'priors' parameter", this->name().c_str());
                return false;
            }

            // check the features
            int sum = 0;
            for(int i = 0; i < this->config_.freqs.size(); i++){
                std::vector<uint32_t> temp = this->config_.freqs.at(i);
                sum = sum + temp.size();
            } 
            if(sum != this->config_.nfeatures){
                ROS_ERROR("[%s] Incorrect dimension for 'freqs' different from 'nfeatures'", this->name().c_str());
                return false;
            }

            return true;
        }


PLUGINLIB_EXPORT_CLASS(rosneuro::decoder::Lda, rosneuro::decoder::GenericDecoder);
    }
}

#endif