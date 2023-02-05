#include "network.hpp"

namespace s21_network {

Network::Network(const double &learning_rate) {
  for (int hidden_layers = SumHiddenLayers::TwoHids;
       hidden_layers < SumHiddenLayers::N; ++hidden_layers) {
    /*---добавляем матричную сеть---*/
    matrix_network_.push_back(new MatrixNetwork(hidden_layers, learning_rate));
    /*---устанавливаем случайные значения весов для матричной сети---*/
    matrix_network_.back()->InstallRandomWeights();

    /*---добавляем графовую сеть---*/
    graph_network_.push_back(new GraphNetwork(hidden_layers, kLearningRate));
    /*---устанавливаем случайные значения весов графовой для сети---*/
    graph_network_.back()->InstallRandomWeights();
  }

  /*---по умолчанию текущая сеть является матричной двухслойной---*/
  current_network_ = matrix_network_.front();
}

Network::~Network() {
  size_t size_matrixN = matrix_network_.size();
  for (size_t i = 0; i < size_matrixN; ++i) {
    delete matrix_network_[i];
  }
  size_t size_graphN = graph_network_.size();
  for (size_t i = 0; i < size_graphN; ++i) {
    delete graph_network_[i];
  }
}

void Network::LoadWeightsFromFile(const std::string &filename,
                                  const int &index_network) {
  if (index_network < 0 || (size_t)index_network >= matrix_network_.size() ||
      (size_t)index_network >= graph_network_.size()) {
    throw std::out_of_range("Error, index Network out of range");
  }
  matrix_network_[index_network]->LoadWeights(filename);
  graph_network_[index_network]->LoadWeights(filename);
}

void Network::SaveWeightsToFile(const std::string &filename) {
  current_network_->SaveWeights(filename);
}

std::vector<double> Network::StartLearnNetwork(const std::string &train_file,
                                               const int &sum_epoch,
                                               const bool &continue_learn,
                                               const std::string &test_file) {
  if (sum_epoch < 1) {
    throw std::invalid_argument("Error in startLearnNetwork(), sumEpoch < 1");
  }
  std::vector<double> res{};
  /*---устанавливаем случайные значения весов для сети, если обучение начинается
   * с нуля---*/
  if (continue_learn == false) {
    current_network_->InstallRandomWeights();
  }

  /*---запускаем оубчение на отведенное количество эпох---*/
  for (int i = 0; i < sum_epoch; ++i) {
    std::ifstream stream(train_file);
    if (stream.is_open()) {
      setlocale(LC_ALL, "en_US.UTF-8");
      while (!stream.eof()) {
        std::string line{};
        std::getline(stream, line);
        if (!line.empty()) {
          std::vector<unsigned> input_values;
          size_t expected_value{};
          ReadLineFromFileWithPixels(line, &expected_value, &input_values);
          /*---запуск обучения текущей сети, выбранной из интерфейса---*/
          //          if (pixels.size() == 784)
          current_network_->LearnNetwork(input_values, expected_value);
        }
      }
      stream.close();
    }
    if (test_file.size() && sum_epoch > 1) {
      auto reply = StartTestNetwork(test_file);
      res.push_back((double)reply.second / (double)reply.first);
    }
  }
  return res;
}

std::vector<double> Network::StartCVLearn(const std::string &train_file,
                                          const unsigned coef,
                                          const bool &continue_learn) {
  std::vector<double> res{};
  if (continue_learn == false) current_network_->InstallRandomWeights();

  for (unsigned i{}; i < coef; i++) {
    size_t correct_pr{}, all_pr{};
    std::ifstream stream(train_file);
    if (stream.is_open()) {
      setlocale(LC_ALL, "en_US.UTF-8");
      unsigned line_index{};
      std::string line{};
      while (!stream.eof()) {
        std::getline(stream, line);
        if (!line.empty()) {
          if (line_index != i) {
            std::vector<unsigned> input_values{};
            size_t expected_value{};
            ReadLineFromFileWithPixels(line, &expected_value, &input_values);
            current_network_->LearnNetwork(input_values, expected_value);
          }
          line_index++;
          if (line_index == coef) line_index = 0;
        }
      }
      stream.clear();
      stream.seekg(0, std::ios_base::beg);
      line_index = 0;
      while (!stream.eof()) {
        std::getline(stream, line);
        if (!line.empty()) {
          if (line_index == i) {
            std::vector<unsigned> input_values{};
            size_t expected_value{};
            ReadLineFromFileWithPixels(line, &expected_value, &input_values);
            if (current_network_->Prediction(input_values) == expected_value) {
              ++correct_pr;
            }
            ++all_pr;
          }
          line_index++;
          if (line_index == coef) line_index = 0;
        }
      }
      stream.close();
    }
    res.push_back((double)correct_pr / (double)all_pr);
  }
  return res;
}

std::pair<size_t, size_t> Network::StartTestNetwork(
    const std::string &test_file_name) {
  size_t all_prediction = 0;
  size_t correct_prediction = 0;

  std::ifstream stream(test_file_name);
  if (stream.is_open()) {
    setlocale(LC_ALL, "en_US.UTF-8");
    while (!stream.eof()) {
      std::string line{};
      std::getline(stream, line);
      if (!line.empty()) {
        std::vector<unsigned> input_values;
        size_t expected_value{};
        ReadLineFromFileWithPixels(line, &expected_value, &input_values);
        /*---запускаем проход по сети и сравниваем с ожидаемым занчением---*/
        if (current_network_->Prediction(input_values) == expected_value) {
          ++correct_prediction;
        }
        ++all_prediction;
      }
    }
    stream.close();
  }
  return {all_prediction, correct_prediction};
}

std::pair<size_t, size_t> Network::StartTestNetwork(
    const std::string &test_file_name, const double &sample_percentage) {
  if (sample_percentage > 1.00 || sample_percentage <= 0.0) {
    throw std::invalid_argument("Error sample percentage");
  }

  size_t all_prediction = 0;
  size_t correct_prediction = 0;

  std::ifstream stream(test_file_name);
  if (stream.is_open()) {
    int sum_test_in_file = 0;
    std::string line{};
    while (!stream.eof()) {
      std::getline(stream, line);
      if (!line.empty()) {
        ++sum_test_in_file;
      }
    }
    sum_test_in_file *= sample_percentage;
    stream.clear();
    stream.seekg(0, std::ios_base::beg);
    for (int i = 0; i < sum_test_in_file; ++i) {
      std::getline(stream, line);
      if (!line.empty()) {
        std::vector<unsigned> input_values;
        size_t expected_value{};
        ReadLineFromFileWithPixels(line, &expected_value, &input_values);
        if (current_network_->Prediction(input_values) == expected_value) {
          ++correct_prediction;
        }
        ++all_prediction;
      }
    }
    stream.close();
  }
  return {all_prediction, correct_prediction};
}

S21Matrix Network::StartConfusionTest(const std::string &test_file_name,
                                      const double &sample_percentage) {
  if (sample_percentage > 1.00 || sample_percentage <= 0.0) {
    throw std::invalid_argument("Error sample percentage");
  }

  S21Matrix res(kSumNeironsOutputLayer,
                kSumNeironsOutputLayer);  // (expected / prediction)

  std::ifstream stream(test_file_name);
  if (stream.is_open()) {
    int sum_test_in_file = 0;
    std::string line{};
    while (!stream.eof()) {
      std::getline(stream, line);
      if (!line.empty()) {
        ++sum_test_in_file;
      }
    }
    sum_test_in_file *= sample_percentage;
    stream.clear();
    stream.seekg(0, std::ios_base::beg);
    for (int i = 0; i < sum_test_in_file; ++i) {
      std::getline(stream, line);
      if (!line.empty()) {
        std::vector<unsigned> input_values;
        size_t expected_value{};
        ReadLineFromFileWithPixels(line, &expected_value, &input_values);
        res(expected_value - 1,
            current_network_->Prediction(input_values) - 1) += 1;
      }
    }
    stream.close();
  }
  return res;
}

double Network::CalcAccuracy(const S21Matrix &conf_mx) {
  double correct{}, total{};
  for (unsigned i{}; i < kSumNeironsOutputLayer; i++) {
    for (unsigned j{}; j < kSumNeironsOutputLayer; j++) {
      if (i == j) correct += conf_mx(i, j);
      total += conf_mx(i, j);
    }
  }
  return correct / total;
}

double Network::CalcPrecision(const S21Matrix &conf_mx) {
  double correct[kSumNeironsOutputLayer]{}, positives[kSumNeironsOutputLayer]{};
  for (unsigned i{}; i < kSumNeironsOutputLayer; i++) {
    for (unsigned j{}; j < kSumNeironsOutputLayer; j++) {
      if (i == j) correct[j] += conf_mx(i, j);
      positives[j] += conf_mx(i, j);
    }
  }
  double res{};
  unsigned existing_cases{};
  for (unsigned i{}; i < kSumNeironsOutputLayer; i++) {
    if (positives[i] > 0) {
      res += correct[i] / positives[i];
      existing_cases++;
    }
  }
  return res / existing_cases;
}

double Network::CalcRecall(const S21Matrix &conf_mx) {
  double correct[kSumNeironsOutputLayer]{}, positives[kSumNeironsOutputLayer]{};
  for (unsigned i{}; i < kSumNeironsOutputLayer; i++) {
    for (unsigned j{}; j < kSumNeironsOutputLayer; j++) {
      if (i == j) correct[i] += conf_mx(i, j);
      positives[i] += conf_mx(i, j);
    }
  }
  double res{};
  unsigned existing_cases{};
  for (unsigned i{}; i < kSumNeironsOutputLayer; i++) {
    if (positives[i] > 0) {
      res += correct[i] / positives[i];
      existing_cases++;
    }
  }
  return res / existing_cases;
}

double Network::CalcFMeasure(const double prec, const double recall) {
  return 2 * prec * recall / (prec + recall);
}

void Network::ChangeCurrentNetwork(const int &index_network,
                                   const bool &type_network) {
  if (index_network < 0 || (size_t)index_network >= kSumNetworks) {
    throw std::invalid_argument(
        "Error in changeCurrentNetwork(), index Network out of range");
  }
  if (type_network == typeNetwork::Matrix) {
    if ((size_t)index_network >= matrix_network_.size()) {
      throw std::out_of_range("Index network out of range");
    }
    current_network_ = matrix_network_[index_network];
  } else if (type_network == typeNetwork::Graph) {
    if ((size_t)index_network >= graph_network_.size()) {
      throw std::out_of_range("Index network out of range");
    }
    current_network_ = graph_network_[index_network];
  }
}

size_t Network::PredictionNetwork(const std::vector<unsigned> &input_layer) {
  return current_network_->Prediction(input_layer);
}

void Network::ReadLineFromFileWithPixels(const std::string &line,
                                         size_t *expected_value,
                                         std::vector<unsigned> *input_values) {
  size_t line_size = line.size();
  std::string value{};
  bool flag(false);
  for (size_t i = 0; i < line_size; ++i) {
    if (line[i] == ',') {
      if (flag) {
        input_values->push_back(std::stoul(value));
      } else {
        *expected_value = std::stoul(value);
        flag = true;
      }
      value.clear();
    } else if (i == line_size - 1) {
      value.push_back(line[i]);
      input_values->push_back(std::stoul(value));
    } else {
      value.push_back(line[i]);
    }
  }
}

}  // namespace s21_network
