#ifndef ROSNEURO_DECODER_LDA_HPP
#define ROSNEURO_DECODER_LDA_HPP

#include <pluginlib/class_list_macros.h>
#include <dynamic_reconfigure/server.h>
#include <ros/ros.h>
#include <gtest/gtest_prod.h>
#include "rosneuro_decoder/GenericDecoder.h"

namespace rosneuro{
    namespace decoder{
        typedef struct {
        	std::string 		   filename;
        	std::string		       subject;
        	std::uint32_t		   n_classes;
            double                 lambda;

            std::vector<double>     priors;
        	std::vector<uint32_t>  class_lbs;
        	std::uint32_t      	   n_features;

        	std::vector<uint32_t>               idchans;
        	std::vector<std::vector<uint32_t>>  freqs;

        } lda_configuration;

        class Lda : public GenericDecoder{
            public:
                Lda(void);
                ~Lda(void);

                bool configure(void);
                Eigen::VectorXf apply(const Eigen::VectorXf& in);
                Eigen::VectorXf getFeatures(const Eigen::MatrixXf& in);

                std::string getPath(void);
                std::vector<int> getClasses(void);

            private:
                double calculateNormalizationCoefficient(int input_size) const;
                double calculateLikelihood(const Eigen::VectorXf& input, int class_index) const;
                double calculatePosteriorDenominator(const Eigen::VectorXf& input, std::vector<double>& likelihoods) const;
                Eigen::VectorXf computePosteriorProbabilities(const std::vector<double>& likelihoods, double posterior_denominator) const;
                template<typename T>
                bool getParamAndCheck(const std::string& param_name, T& param_value);
                bool checkDimension(void);

            private:
                ros::NodeHandle p_nh_;
                Eigen::MatrixXf means_;
                Eigen::MatrixXf covs_;
                lda_configuration config_;

                FRIEND_TEST(LdaTestSuite, Constructor);
                FRIEND_TEST(LdaTestSuite, Configure);
                FRIEND_TEST(LdaTestSuite, CheckDimensionMeans);
                FRIEND_TEST(LdaTestSuite, CheckDimensionCovs);
                FRIEND_TEST(LdaTestSuite, CheckDimensionSize);
                FRIEND_TEST(LdaTestSuite, Apply);
                FRIEND_TEST(LdaTestSuite, Integration);
        };
    }
}

#endif