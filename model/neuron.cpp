#include "neuron.h"

#include <cmath>

namespace s21_network {

float SigmaFunction(float x) { return 1.0 / (1.0 + pow(M_E, -x)); }

void Neuron::AddInput(Neuron* inp, float wgt) {
  input_.push_back(inp);
  weight_.push_back(wgt);
}

void Neuron::ClearInput() { input_.clear(); }

void Neuron::set_mode(int src) {
  if (src == ActFunction::kLinear || src == ActFunction::kSigmoid)
    act_mode_ = src;
}

void Neuron::Activate() {
  if (act_mode_ == ActFunction::kLinear) {
    value_ = SumInput();
  } else {
    value_ = SigmaFunction(SumInput());
  }
}

void Neuron::set_deriv(const float& val) {
  deriv_ = val;
}

float Neuron::get_deriv() {
  return deriv_;
}

float Neuron::SumInput() {
  float res{};
  for (size_t i{}; i < input_.size(); i++)
    res += weight_[i] * input_[i]->get_value();
  return res;
}

void Neuron::CorrectWeights(float learning_rate) {
  for (size_t i{}; i < weight_.size(); i++) {
    CorrectWeidht(i, learning_rate);
  }
}

void Neuron::CorrectWeidht(int inp_index, float learning_rate) {
  weight_[inp_index] -= learning_rate * input_[inp_index]->get_value() * deriv_;
}

}  // namespace s21_network
