#pragma once

#include <string>
#include <vector>

namespace s21_network {

constexpr unsigned kInputLayer = 784;
constexpr unsigned kSumNeironsHiddenLayer = 140;
constexpr unsigned kSumNeironsOutputLayer = 26;

class InterfaceNetwork {
 public:
  void virtual InstallRandomWeights() = 0;
  void virtual LoadWeights(const std::string &filename) = 0;
  void virtual SaveWeights(const std::string &filename) = 0;
  size_t virtual Prediction(const std::vector<unsigned> &input_layer) = 0;
  void virtual LearnNetwork(const std::vector<unsigned> &input_layer,
                            const size_t &expected_value) = 0;
};
}  // namespace s21_network
