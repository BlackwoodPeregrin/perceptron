#pragma once

#include <stdlib.h>

#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>

#include "interfaceNetwork.hpp"
#include "s21_matrix_oop.h"

namespace s21_network {

class MatrixNetwork : public InterfaceNetwork {
 public:
  MatrixNetwork(const int &sum_hidden_layers, const double &learning_rate);
  virtual ~MatrixNetwork();

  void InstallRandomWeights() override;
  void LoadWeights(const std::string &filename) override;
  void SaveWeights(const std::string &filename) override;
  size_t Prediction(const std::vector<unsigned> &input_layer) override;
  void LearnNetwork(const std::vector<unsigned> &input_layer,
                    const size_t &expectedValue) override;
  void FeedForward(const std::vector<unsigned> &input_layer);

 protected:
  void set_input_layer(const std::vector<unsigned> &input_layer);
  void CorrectWeights();

 private:
  class HiddenLayer {
   public:
    HiddenLayer(const unsigned &rows_weights_matrix,
                const unsigned &columns_weights_matrix);
    ~HiddenLayer();

    void LoadWeights(std::ifstream *stream);
    void SaveWeights(std::ofstream *stream);
    void CorrectWeights(const S21Matrix &output_matrix_prev_layer,
                        const double &learning_rate);
    void InstallRandomWeights();

    void CalcOutputMatrix(const S21Matrix &output_matrix_prev_layer);
    void CalcWeightsDeltaMatrix(const S21Matrix &delta_matrix_prev_layer);

    /*----getters HiddenLayer-------*/
    const S21Matrix &get_output_matrix();
    const S21Matrix &get_weights_delta_matrix();
    const size_t &get_sum_neirons();

    /*-----print functions-----*/
    void print_weghts();
    void print_output_values();
    void print_weights_delta();
    /*------------------------*/

   protected:
    S21Matrix *m_output_;   // матрица значений нейронов
    S21Matrix *m_weights_;  // матрица весов
    S21Matrix *m_weights_delta_;
    size_t sum_neirons_;  // количество нейронов в скрытых слоях
  };

  class OutputLayer : public HiddenLayer {
   public:
    OutputLayer(const unsigned &rows_weight_layer,
                const unsigned &cols_weight_layer);

    void CalcWeightsDeltaMatrix();
    void CalcWeightsDeltaMatrix(const S21Matrix &delta_matrix_prevLayer) =
        delete;

    void set_expected_value(const size_t &value);
    const size_t &get_expected_value();

    size_t ResultNeiron();

   private:
    size_t expected_value_;  // ожидаемое значение
  };

 private:
  S21Matrix *input_layer_;                    // входной слой
  std::vector<HiddenLayer *> hidden_layers_;  // скрытые слои
  OutputLayer *output_layer_;                 // выходной слой
  double learning_rate_;  // коэффициент скорости обучения
};

}  // namespace s21_network
