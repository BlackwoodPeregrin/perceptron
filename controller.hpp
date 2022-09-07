#pragma once

#include "network.hpp"

namespace s21_network {
class Controller {
 public:
  Controller();
  ~Controller();

  std::string LoadWeightsNetwork(const std::string &filename,
                                   const int &index_network) {
    try {
      network_->LoadWeightsFromFile(filename, index_network);
      return "Weights load SUCSESS";
    } catch (const std::exception &e) {
      return e.what();
    }
  }
  void SaveWeightsNetwork(const std::string &filename) {
    network_->SaveWeightsToFile(filename);
  }
  std::pair<size_t, size_t> StartTestNetwork(const std::string &testfile) {
    return network_->StartTestNetwork(testfile);
  }
  std::pair<size_t, size_t> StartTestNetwork(const std::string &testfile,
                                               const double &sample_percentage) {
    try {
      return network_->StartTestNetwork(testfile, sample_percentage);
    } catch (const std::exception &e) {
      return {0, 0};
    }
  }
  S21Matrix StartConfusionTest(const std::string &testfile,
                                 const double &sample_percentage) {
    try {
      return network_->StartConfusionTest(testfile, sample_percentage);
    } catch (const std::exception &e) {
      S21Matrix A{0, 0};
      return A;
    }
  }
  double CalcAccuracy(const S21Matrix &conf_mx) {
    return network_->CalcAccuracy(conf_mx);
  }
  double CalcPrecision(const S21Matrix &conf_mx) {
    return network_->CalcPrecision(conf_mx);
  }
  double CalcRecall(const S21Matrix &conf_mx) {
    return network_->CalcRecall(conf_mx);
  }
  double CalcFMeasure(double prec, double recall) {
    return network_->CalcFMeasure(prec, recall);
  }
  std::vector<double> StartLearnNetwork(const std::string &train_file,
                                          const int &sum_epoch,
                                          const bool &continue_learn,
                                          const std::string &test_file) {
    return network_->StartLearnNetwork(train_file, sum_epoch, continue_learn,
                                       test_file);
  }
  std::vector<double> StartCVLearn(const std::string &train_file,
                                     const unsigned coef,
                                     const bool &continue_learn) {
    return network_->StartCVLearn(train_file, coef, continue_learn);
  }
  size_t get_result_network(const std::vector<unsigned> &input_layer) {
    return network_->PredictionNetwork(input_layer);
  }
  void SwitchNetwork(const int &index_network, const bool &type_network) {
    network_->ChangeCurrentNetwork(index_network, type_network);
  }

 private:
  Network *network_;  //  сеть
};
}  // namespace s21_network
