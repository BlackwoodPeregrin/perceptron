#include "graphNetwork.hpp"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace s21_network {

enum NetSize { kInpWdt, kHidDpt, kHidWdt, kOutWdt };

int Roll(int base);

GraphNetwork::~GraphNetwork() { Clear(); }

void GraphNetwork::DelInputLayer() {
  for (auto i : input_layer_) delete i;
  input_layer_.clear();
}

void GraphNetwork::DelHiddenLayer() {
  for (auto i : hidden_layer_)
    for (auto j : i) delete j;
  hidden_layer_.clear();
}

void GraphNetwork::DelOutputLayer() {
  for (auto i : output_layer_) delete i;
  output_layer_.clear();
}

void GraphNetwork::CreateInputLayer(int num) {
  if (input_layer_.size()) DelInputLayer();
  if (hidden_layer_.size()) SeparateInputHidden();
  for (int i{}; i < num; i++) {
    Neuron* one = new Neuron();
    input_layer_.push_back(one);
  }
}

void GraphNetwork::CreateHiddenLayer(int depth, int width) {
  if (hidden_layer_.size()) DelHiddenLayer();
  if (output_layer_.size()) SeparateHiddenOutput();
  std::vector<Neuron*> A{};
  for (int i{}; i < depth; i++) hidden_layer_.push_back(A);
  for (int i{}; i < depth; i++) {
    for (int j{}; j < width; j++) {
      Neuron* one = new Neuron();
      one->set_mode(ActFunction::kSigmoid);
      hidden_layer_[i].push_back(one);
    }
  }
  StitchHidden();
}

void GraphNetwork::CreateOutputLayer(int num) {
  if (output_layer_.size()) DelOutputLayer();
  for (int i{}; i < num; i++) {
    Neuron* one = new Neuron();
    one->set_mode(ActFunction::kSigmoid);
    output_layer_.push_back(one);
  }
}

void GraphNetwork::StitchHidden() {
  int depth = hidden_layer_.size();
  int width = hidden_layer_[1].size();
  for (int i{1}; i < depth; i++) {
    for (int j{}; j < width; j++) {
      for (int k{}; k < width; k++) {
        hidden_layer_[i][j]->AddInput(hidden_layer_[i - 1][k]);
      }
    }
  }
}

void GraphNetwork::ConnectInputHidden() {
  for (auto i : hidden_layer_[0]) {
    for (auto j : input_layer_) i->AddInput(j);
  }
}

void GraphNetwork::ConnectHiddenOutput() {
  for (auto i : output_layer_) {
    for (auto j : hidden_layer_[hidden_layer_.size() - 1]) i->AddInput(j);
  }
}

void GraphNetwork::SeparateInputHidden() {
  if (hidden_layer_.size()) {
    for (auto i : hidden_layer_[0]) i->ClearInput();
  }
}

void GraphNetwork::SeparateHiddenOutput() {
  if (output_layer_.size()) {
    for (auto i : output_layer_) i->ClearInput();
  }
}

void GraphNetwork::SetupNetwork(int wdt_in, int num_hid, int wdt_hid,
                                int wdt_out) {
  if (input_layer_.size()) DelInputLayer();
  if (hidden_layer_.size()) DelHiddenLayer();
  if (output_layer_.size()) DelOutputLayer();
  CreateInputLayer(wdt_in);
  CreateHiddenLayer(num_hid, wdt_hid);
  CreateOutputLayer(wdt_out);
  ConnectInputHidden();
  ConnectHiddenOutput();
}

void GraphNetwork::Clear() {
  DelInputLayer();
  DelHiddenLayer();
  DelOutputLayer();
}

void GraphNetwork::ResizeHidden(size_t depth) {
  if (depth > 0 && depth != hidden_layer_.size() && hidden_layer_.size()) {
    int width = hidden_layer_[0].size();
    SeparateHiddenOutput();
    DelHiddenLayer();
    CreateHiddenLayer(depth, width);
    ConnectInputHidden();
    ConnectHiddenOutput();
  }
}

int GraphNetwork::Run(const std::vector<float>& src) {
  if (!is_set_up()) throw std::out_of_range("network not set up");

  Feed(src);
  Execute();
  return get_result();
}

void GraphNetwork::set_expected_values(const std::vector<float>& val) {
  int dif = output_layer_.size() - val.size();
  if (dif >= 0) {
    expected_values_ = val;
    for (int i{}; i < dif; i++) expected_values_.push_back(0.0);
  } else {
    for (size_t i{}; i < output_layer_.size(); i++)
      expected_values_.push_back(val[i]);
  }
}

void GraphNetwork::set_learning_rate(float src) {
  if (src > 0)
    learning_rate_ = src;
}

void GraphNetwork::EducateOneStep(const std::vector<float>& src, int expectation) {
  if (!is_set_up()) throw std::out_of_range("network not set up");

  Feed(src);
  Execute();
  set_expected_values(FormExpectationVector(expectation));
  CalcDerivOutput();
  CalcDerivHidden();
  CorrectWeights();
}

bool GraphNetwork::is_set_up() { return (input_layer_.size()) ? true : false; }

void GraphNetwork::Feed(const std::vector<float>& src) {
  for (size_t i{}; i < input_layer_.size(); i++) {
    if (i < src.size()) {
      input_layer_[i]->set_value(src[i]);
    } else {
      input_layer_[i]->set_value(0);
    }
  }
}

void GraphNetwork::Execute() {
  for (auto &i : hidden_layer_) {
    for (auto j : i) j->Activate();
  }
  for (auto i : output_layer_) i->Activate();
}

int GraphNetwork::get_result() {
  int res{};
  float max = output_layer_[0]->get_value();
  for (size_t i{1}; i < output_layer_.size(); i++) {
    float val = output_layer_[i]->get_value();
    if (max < val) {
      max = val;
      res = i;
    }
  }
  return res;
}

float& GraphNetwork::hidden_weight(int layer, int num, int inp) {
  return hidden_layer_[layer][num]->weight(inp);
}

float& GraphNetwork::output_weight(int num, int inp) {
  return output_layer_[num]->weight(inp);
}

int GraphNetwork::get_inp_width() { return input_layer_.size(); }

int GraphNetwork::get_hid_width() {
  if (hidden_layer_.size())
    return hidden_layer_[0].size();
  else
    return 0;
}

int GraphNetwork::get_hid_depth() { return hidden_layer_.size(); }

int GraphNetwork::get_out_width() { return output_layer_.size(); }

void GraphNetwork::SaveWeights(const std::string& filename) {
  if (!is_set_up()) return;

  std::ofstream stream(filename);
  if (stream.is_open()) {
    setlocale(LC_ALL, "en_US.UTF-8");
    stream << "Weights Network" << std::endl;
    stream << std::to_string(get_hid_depth()) + " Hiddens Layers" << std::endl;
    for (size_t i{}; i < hidden_layer_.size() + 1; i++) {
      for (int k{}; k < get_num_inputs(i); k++) {
        for (int j{}; j < get_num_neuron(i); j++) {
          if (j != get_num_neuron(i) - 1)
            stream << get_neuron(i, j)->weight(k) << " ";
          else
            stream << get_neuron(i, j)->weight(k) << std::endl;
        }
      }
      stream << "Layer weights are over" << std::endl;
    }
    stream.close();
  }
}

void GraphNetwork::LoadWeights(const std::string& filename) {
  std::ifstream stream(filename);

  if (stream.is_open()) {
    setlocale(LC_ALL, "en_US.UTF-8");
    std::string line{};
    std::string value{};
    size_t index_row = 0;
    size_t index_col = 0;
    std::string type_network{};
    std::getline(stream, type_network);
    if (type_network == "Weights Network") {
      std::getline(stream, type_network);
      if (type_network == (std::to_string(get_hid_depth()) + " Hiddens Layers")) {
        int index_layer{};
        while (!stream.eof()) {
          std::getline(stream, line);
          if (line == "Layer weights are over") {
            ++index_layer;
            std::getline(stream, line);
            index_row = 0;
            index_col = 0;
          }
          size_t line_size = line.size();
          for (size_t i{}; i < line_size; ++i) {
            if (line[i] == ' ') {
              get_neuron(index_layer, index_col)->weight(index_row) = std::stod(value);
              ++index_col;
              value.clear();
            } else if (i == line_size - 1) {
              value.push_back(line[i]);
              get_neuron(index_layer, index_col)->weight(index_row) = std::stod(value);
              index_col = 0;
              value.clear();
            } else {
              value.push_back(line[i]);
            }
          }
          ++index_row;
        }
      } else {
        std::string error_text =
            "Error, Network have " + std::to_string(get_hid_width()) +
            " hidden Layers. But you try load Network with" + type_network;
        throw std::invalid_argument(error_text);
      }
    } else {
      std::string error_text = "The file isn't a weights for the Network";
      throw std::invalid_argument(error_text);
    }
    stream.close();
  }
}

void GraphNetwork::InstallRandomWeights() {
  srand(time(0));
  for (int i{}; i < get_hid_depth() + 1; i++) {
    for (int j{}; j < get_num_neuron(i); j++) {
      for (int k{}; k < get_num_inputs(i); k++) {
        weight(i, j, k) = Roll(201) / 100.0 - 1.0;
      }
    }
  }
}

size_t GraphNetwork::Prediction(const std::vector<unsigned> &input_values) {
  return (size_t)Run(FormFeedVector(input_values)) + 1;
}

void GraphNetwork::LearnNetwork(const std::vector<unsigned> &input_values,
      const size_t &expected_value) {
  EducateOneStep(FormFeedVector(input_values), (int)(expected_value - 1));
}

// random number generation
int Roll(int base) {
  int res{};
  if (base > 0) res = rand() % base;
  return res;
}

// слои считаются с hidden[0] до output
float& GraphNetwork::weight(int layer, int num, int inp) {
  if (layer > -1) {
    if (layer < get_hid_depth())
      return hidden_weight(layer, num, inp);
    else if (layer == get_hid_depth())
      return output_weight(num, inp);
  }
  throw std::out_of_range("no weight with such indices");
}

// слои считаются с hidden[0] до output
int GraphNetwork::get_num_neuron(int layer) {
  int res{};
  if (layer > -1) {
    if (layer < get_hid_depth())
      res = get_hid_width();
    else if (layer == get_hid_depth())
      res = get_out_width();
  }
  return res;
}

// возвращает количество источников для нейрона на layer уровне,
// исключая input, считая output после последнего слоя hidden
int GraphNetwork::get_num_inputs(int layer) {
  int res{};
  if (layer > -1) {
    if (layer == 0)
      res = get_inp_width();
    else if (layer <= get_hid_depth())
      res = get_hid_width();
  }
  return res;
}

void GraphNetwork::CalcDerivOutput() {
  for (int i{}; i < get_out_width(); i++) {
    float value = output_layer_[i]->get_value();
    float deriv = (value - expected_values_[i]) * value * (1 - value);
    output_layer_[i]->set_deriv(deriv);
  }
}

void GraphNetwork::CalcDerivHidden() {
  for (int i = get_hid_depth() - 1; i >= 0; i--) {
    for (int j{}; j < get_num_neuron(i); j++) {
      float sum{};
      for (int k{}; k < get_num_neuron(i + 1); k++) {
        sum += get_neuron(i + 1, k)->get_deriv() * weight(i + 1, k, j);
      }
      float value = get_neuron(i, j)->get_value();
      float deriv = sum * value * (1 - value);
      get_neuron(i, j)->set_deriv(deriv);
    }
  }
}

// void GraphNetwork::CalcDerivInput() {
//   for (int i{}; i < get_inp_width(); i++) {
//     float sum{};
//     for (int j{}; j < get_hid_width(); j++) {
//       sum += get_neuron(0, j)->get_deriv() * weight(0, j, i);
//     }
//     float value = input_layer_[i]->get_value();
//     float deriv = sum * value * (1 - value);
//     input_layer_[i]->set_deriv(deriv);
//   }
// }

Neuron* GraphNetwork::get_neuron(int i, int j) {
  if (i < get_hid_depth())
    return hidden_layer_[i][j];
  else if (i == get_hid_depth())
    return output_layer_[j];
  throw std::out_of_range("no such neuron");
}

std::vector<float> GraphNetwork::FormExpectationVector(int exp) {
  std::vector<float> res{};
  int i = 0;
  while (i < get_out_width()) {
    if (i != exp)
      res.push_back(0.0);
    else
      res.push_back(1.0);
    i++;
  }
  return res;
}

std::vector<float> GraphNetwork::FormFeedVector(const std::vector<unsigned> &src) {
  std::vector<float> data{};
  for (auto i : src)
    data.push_back(i / 255.0);
  return data;
}

void GraphNetwork::CorrectWeights() {
  for (int i{}; i < get_hid_depth() + 1; i++) {
    for (int j{}; j < get_num_neuron(i); j++) {
      get_neuron(i, j)->CorrectWeights(learning_rate_);
    }
  }
}

}  // namespace s21_network
