#pragma once

#include "neuron.h"
#include "interfaceNetwork.hpp"

namespace s21_network {

class GraphNetwork : public InterfaceNetwork {
  std::vector<Neuron*> input_layer_{};
  std::vector<std::vector<Neuron*>> hidden_layer_{};
  std::vector<Neuron*> output_layer_{};
  std::vector<float> expected_values_{};
  float learning_rate_ = 0.2;

 public:
  GraphNetwork() {}
  GraphNetwork(int hidden_layers, float learning_rate) : learning_rate_(learning_rate) {
    SetupNetwork(kInputLayer, hidden_layers, kSumNeironsHiddenLayer, kSumNeironsOutputLayer);
  }
  virtual ~GraphNetwork();

  void SetupNetwork(int wdt_in, int num_hid, int wdt_hid, int wdt_out);
  void Clear();
  void ResizeHidden(size_t depth);
  float& hidden_weight(int layer, int num, int inp);
  float& output_weight(int num, int inp);
  float& weight(int layer, int num, int inp);

  int get_inp_width();
  int get_hid_width();
  int get_hid_depth();
  int get_out_width();
  int get_num_neuron(int layer);
  int get_num_inputs(int layer);

  void SaveWeights(const std::string& filename) override;
  void LoadWeights(const std::string& filename) override;
  void InstallRandomWeights() override;
  size_t Prediction(const std::vector<unsigned> &input_values) override;
  void LearnNetwork(const std::vector<unsigned> &input_values,
        const size_t &expected_value) override;

  void set_expected_values(const std::vector<float>& val);
  void set_learning_rate(float src);

 private:
  void CreateInputLayer(int num);
  void CreateHiddenLayer(int num, int size);
  void CreateOutputLayer(int num);

  void DelInputLayer();
  void DelHiddenLayer();
  void DelOutputLayer();

  void StitchHidden();
  void ConnectInputHidden();
  void ConnectHiddenOutput();

  void SeparateInputHidden();
  void SeparateHiddenOutput();

  bool is_set_up();
  int Run(const std::vector<float>& src);
  void EducateOneStep(const std::vector<float>& src, int expectation);
  void Feed(const std::vector<float>& src);
  void Execute();
  int get_result();

  void CalcDerivOutput();
  void CalcDerivHidden();
  void CalcDerivInput();
  Neuron* get_neuron(int i, int j);
  std::vector<float> FormExpectationVector(int exp);
  std::vector<float> FormFeedVector(const std::vector<unsigned> &src);
  void CorrectWeights();
};

}  // namespace s21_network
