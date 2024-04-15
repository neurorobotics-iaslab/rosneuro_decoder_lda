#include "rosneuro_decoder_lda/Lda.hpp"
#include <fstream>
#include <ros/package.h>
#include <gtest/gtest.h>

namespace rosneuro {
    namespace decoder {
        class LdaTestSuite : public ::testing::Test {
            public:
                LdaTestSuite() {}
                ~LdaTestSuite() {}
                void SetUp() { lda = new Lda(); }
                void TearDown() { delete lda; }
                Lda* lda;
        };

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> readCSV(const std::string& filename) {
            std::vector<double> values;
            std::ifstream file(filename);
            std::string row;
            std::string entry;
            int n_rows = 0;

            while (getline(file, row)) {
                std::stringstream rowstream(row);
                while (getline(rowstream, entry, ',')) {
                    values.push_back(std::stod(entry));
                }
                n_rows++;
            }
            return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), n_rows, values.size() / n_rows);
        }

        void config_lda(std::map<std::string, XmlRpc::XmlRpcValue>& params) {
            params["name"] = "lda";
            params["filename"] = "file1";
            params["subject"] = "s1";
            params["n_classes"] = 2;
            params["n_features"] = 5;
            params["lambda"] = 0.5;

            XmlRpc::XmlRpcValue priors, class_lbs, idchans;
            priors[0] = 0.5;
            priors[1] = 0.5;
            params["priors"] = priors;

            class_lbs[0] = 771;
            class_lbs[1] = 773;
            params["class_lbs"] = class_lbs;

            idchans[0] = 1;
            idchans[1] = 2;
            params["idchans"] = idchans;

            params["freqs"] = "10 12; 20 22 24;";
            params["means"] = "0.4964 1.4994;"
                              "0.5297 1.5036;"
                              "0.4903 1.5054;"
                              "0.4491 1.4950;"
                              "0.4956 1.4733;";

            params["covs"] = "1.1340 0.1117 0.1027 0.1137 0.1006;"
                             "0.1117 1.1363 0.1144 0.1189 0.1126;"
                             "0.1027 0.1144 1.1470 0.1132 0.1099;"
                             "0.1137 0.1189 0.1132 1.1372 0.1028;"
                             "0.1006 0.1126 0.1099 0.1028 1.1131;";
        }

        TEST_F(LdaTestSuite, Constructor) {
            EXPECT_EQ(lda->getName(), "lda");
            EXPECT_EQ(lda->is_configured_, false);
        }

        TEST_F(LdaTestSuite, Configure) {
            config_lda(lda->params_);
            ASSERT_TRUE(lda->configure());
        }

        TEST_F(LdaTestSuite, CheckDimensionMeans) {
            config_lda(lda->params_);
            lda->params_["means"] = "1 2;";
            ASSERT_FALSE(lda->configure());
        }

        TEST_F(LdaTestSuite, CheckDimensionCovs) {
            config_lda(lda->params_);
            lda->params_["covs"] = "1 2;";
            ASSERT_FALSE(lda->configure());
        }

        TEST_F(LdaTestSuite, CheckDimensionSize) {
            config_lda(lda->params_);
            XmlRpc::XmlRpcValue priors;
            priors[0] = 0.5;
            lda->params_["priors"] = priors;
            ASSERT_FALSE(lda->configure());
        }

        TEST_F(LdaTestSuite, Apply) {
            config_lda(lda->params_);
            lda->configure();

            Eigen::VectorXf in(5);
            in << 0.1, 0.2, 0.3, 0.4, 0.5;

            Eigen::VectorXf expected(2);
            expected << 0.9006, 0.0993;

            ASSERT_TRUE(lda->apply(in).isApprox(expected, 0.001));
        }

        TEST_F(LdaTestSuite, Integration) {
            config_lda(lda->params_);

            std::string package_path = ros::package::getPath("rosneuro_decoder_lda");
            const std::string file_input  = package_path + "/test/data/features.csv";
            const std::string file_output = package_path + "/test/data/output.csv";

            Eigen::MatrixXd input = readCSV(file_input);
            Eigen::MatrixXd output = readCSV(file_output);

            ASSERT_TRUE(lda->configure());

            for(int i = 0; i < input.rows(); i++){
                Eigen::VectorXf vector_casted = input.row(i).cast<float>();
                Eigen::VectorXf result = lda->apply(vector_casted);
                Eigen::VectorXf expected = output.row(i).cast<float>();
                ASSERT_TRUE(result.isApprox(expected, 1e-5));
            }
        }
    }
}

int main(int argc, char **argv) {
    // ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Fatal);
    ros::init(argc, argv, "test_lda");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}