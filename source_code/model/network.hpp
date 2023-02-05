#pragma once

#include "matrixNetwork.hpp"
#include "graphNetwork.hpp"

namespace s21_network {

enum SumHiddenLayers { TwoHids = 2, ThreeHids, FourHids, FiveHids, N };
enum typeNetwork { Matrix, Graph };
constexpr double kLearningRate = 0.12;
constexpr size_t kSumNetworks = 4;

class Network {
 public:
  explicit Network(const double &learning_rate);
  ~Network();

  void LoadWeightsFromFile(const std::string &filename, const int &index_network);
  void SaveWeightsToFile(const std::string &filename);

  std::vector<double> StartLearnNetwork(const std::string &train_file, const int &sum_epoch,
                         const bool &continue_learn, const std::string &test_file);
  std::vector<double> StartCVLearn(const std::string &train_file, const unsigned coef,
                                   const bool &continue_learn);

  /*---возвращает общее количество тестов и корреткные предсказания сети---*/
  std::pair<size_t, size_t> StartTestNetwork(const std::string &test_file_name);
  std::pair<size_t, size_t> StartTestNetwork(const std::string &test_file_name,
                                             const double &sample_percentage);
  S21Matrix StartConfusionTest(const std::string &test_file_name,
                                               const double &sample_percentage);
  // calculation of stats
  double CalcAccuracy(const S21Matrix& conf_mx);
  double CalcPrecision(const S21Matrix& conf_mx);
  double CalcRecall(const S21Matrix& conf_mx);
  double CalcFMeasure(double prec, double recall);

  void ChangeCurrentNetwork(const int &index_network, const bool &type_network);
  size_t PredictionNetwork(const std::vector<unsigned> &input_layer);

 protected:
  void ReadLineFromFileWithPixels(const std::string &line, size_t *expected_value,
                                  std::vector<unsigned> *input_values);

 private:
  std::vector<MatrixNetwork *> matrix_network_;  // вектор матрирчных сетей
  std::vector<GraphNetwork *> graph_network_;  // вектор графовых сетей
  InterfaceNetwork *current_network_;  // указатель на интерфес сети
};
}  // namespace s21_network
