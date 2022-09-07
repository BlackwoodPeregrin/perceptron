#pragma once

#include <vector>

namespace s21_network {

enum ActFunction { kLinear, kSigmoid };

class Neuron {
  std::vector<Neuron*> input_{};
  std::vector<float> weight_{};
  float value_{};
  int act_mode_{};
  float deriv_{};

 public:
  Neuron() {}

  float get_value() { return value_; }
  float& weight(int index) { return weight_[index]; }
  void set_value(float value) { value_ = value; }

  void AddInput(Neuron* inp, float wgt);
  void AddInput(Neuron* inp) { AddInput(inp, 1); }
  void ClearInput();
  void set_mode(int src);

  void Activate();

  void set_deriv(const float& val);
  float get_deriv();
  void CorrectWeights(float learning_rate);

 private:
  void CorrectWeidht(int inp_index, float learning_rate);
  float SumInput();
};

}  // namespace s21_network
