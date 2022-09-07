#include "matrixNetwork.hpp"

namespace s21_network {

/*––––––––––– class MatrixNetwork –––––––––––––––––*/

MatrixNetwork::MatrixNetwork(const int &sum_hidden_layers,
                             const double &learn_rate)
    : input_layer_(nullptr),
      output_layer_(nullptr),
      learning_rate_(learn_rate) {
  if (sum_hidden_layers < 1) {
    throw std::invalid_argument(
        "Error in constructor MatrixNetwork, argument < 1");
  }

  /*---добавляем первый слой, его матрица весов зависит от входного слоя---*/
  hidden_layers_.push_back(
      new HiddenLayer(kInputLayer, kSumNeironsHiddenLayer));

  /*---добавляем остальные скрытые слой, если они есть---*/
  for (int i = 1; i < sum_hidden_layers; ++i) {
    hidden_layers_.push_back(
        new HiddenLayer(kSumNeironsHiddenLayer, kSumNeironsHiddenLayer));
  }

  /*---добавляем выходной слой---*/
  output_layer_ =
      new OutputLayer(kSumNeironsHiddenLayer, kSumNeironsOutputLayer);
}

MatrixNetwork::~MatrixNetwork() {
  if (input_layer_ != nullptr) {
    delete input_layer_;
  }
  if (output_layer_ != nullptr) {
    delete output_layer_;
  }
  for (size_t i = 0; i < hidden_layers_.size(); ++i) {
    delete hidden_layers_[i];
  }
}

void MatrixNetwork::InstallRandomWeights() {
  srand(time(NULL));
  /*---устанавливаем рандомные веса для скрытых слоев---*/
  size_t sum_hidden_layers = hidden_layers_.size();
  for (size_t index_layer = 0; index_layer < sum_hidden_layers; ++index_layer) {
    hidden_layers_[index_layer]->InstallRandomWeights();
  }
  /*---устанавливаем рандомные веса для выходного слоя---*/
  output_layer_->InstallRandomWeights();
}

void MatrixNetwork::LoadWeights(const std::string &filename) {
  std::ifstream stream(filename);
  if (stream.is_open()) {
    setlocale(LC_ALL, "en_US.UTF-8");
    std::string type_network{};
    /*---считываем первую строку и проверяем является ли файл корректным---*/
    std::getline(stream, type_network);
    if (type_network == "Weights Network") {
      /*---считываем вторую строку и проверяем является ли файл подходящем для
       * данной сети по количеству скрытых слоев---*/
      std::getline(stream, type_network);
      if (type_network ==
          (std::to_string(hidden_layers_.size()) + " Hiddens Layers")) {
        /*---загружаем веса скрытых слоев---*/
        for (size_t i = 0; i < hidden_layers_.size(); ++i) {
          hidden_layers_[i]->LoadWeights(&stream);
        }
        /*---загружаем веса выходного слоя---*/
        output_layer_->LoadWeights(&stream);
      } else {
        std::string error_text =
            "Error, current Network have " +
            std::to_string(hidden_layers_.size()) +
            " hidden Layers. But you try load Network with " + type_network +
            ", switch current Network, if u wanna load this file with weights.";
        throw std::invalid_argument(error_text);
      }
    } else {
      std::string error_text = "The file isn't a weights for the Network";
      throw std::invalid_argument(error_text);
    }
    stream.close();
  }
}

void MatrixNetwork::SaveWeights(const std::string &filename) {
  std::ofstream stream(filename);
  if (stream.is_open()) {
    setlocale(LC_ALL, "en_US.UTF-8");
    /*---первая строка файла нужна для определения, что это файл с весами
     * сети---*/
    stream << "Weights Network" << std::endl;
    /*---вторая строка файла, это количетсво скрытых слоев данной сети---*/
    stream << std::to_string(hidden_layers_.size()) + " Hiddens Layers"
           << std::endl;
    /*---далее сохраняем веса скрытых слоев---*/
    for (size_t i = 0; i < hidden_layers_.size(); ++i) {
      hidden_layers_[i]->SaveWeights(&stream);
    }
    /*---далее сохраняем веса выходного слоя---*/
    output_layer_->SaveWeights(&stream);
    stream.close();
  }
}

size_t MatrixNetwork::Prediction(const std::vector<unsigned> &input_layer) {
  FeedForward(input_layer);
  return output_layer_->ResultNeiron();
}

void MatrixNetwork::LearnNetwork(const std::vector<unsigned> &input_layer,
                                 const size_t &expected_value) {
  /*---задаем ожидаемое значени---*/
  output_layer_->set_expected_value(expected_value);

  /*---прямой проход по сети---*/
  FeedForward(input_layer);

  /*---back Propogation---*/
  CorrectWeights();
}

void MatrixNetwork::FeedForward(const std::vector<unsigned> &input_layer) {
  /*---задаем входной слой---*/
  set_input_layer(input_layer);

  /*---счиатем значения первого слоя, они зависят от входного слоя---*/
  hidden_layers_.front()->CalcOutputMatrix(*input_layer_);

  /*---счиатем значения оставшихся слоев, если они есть---*/
  size_t sumHiddensLayer = hidden_layers_.size();
  for (size_t i = 1; i < sumHiddensLayer; ++i) {
    hidden_layers_[i]->CalcOutputMatrix(
        hidden_layers_[i - 1]->get_output_matrix());
  }

  /*---счиатем значения выходного слоя---*/
  output_layer_->CalcOutputMatrix(hidden_layers_.back()->get_output_matrix());
}

void MatrixNetwork::set_input_layer(const std::vector<unsigned> &input_layer) {
  if (input_layer_ != nullptr) {
    delete input_layer_;
  }

  size_t len_input_layer = input_layer.size();
  input_layer_ = new S21Matrix(1, len_input_layer);

  for (size_t i = 0; i < len_input_layer; ++i) {
    (*input_layer_)(0, i) = (double)input_layer[i] / 255;
  }
}

void MatrixNetwork::CorrectWeights() {
  /*---вычисляем m_weightsDelta_ выходного слоя---*/
  output_layer_->CalcWeightsDeltaMatrix();

  /*---корректируем веса выходного слоя---*/
  output_layer_->CorrectWeights(hidden_layers_.back()->get_output_matrix(),
                                learning_rate_);

  /*---далее корректируем веса в обратном порядке от выходного слоя---*/

  unsigned int sum_hiddens = hidden_layers_.size() - 1;
  for (int index_layer = sum_hiddens; index_layer >= 0; --index_layer) {
    if (index_layer == (int)sum_hiddens) {
      hidden_layers_[index_layer]->CalcWeightsDeltaMatrix(
          output_layer_->get_weights_delta_matrix());
    } else {
      hidden_layers_[index_layer]->CalcWeightsDeltaMatrix(
          hidden_layers_[index_layer + 1]->get_weights_delta_matrix());
    }

    if (index_layer != 0) {
      hidden_layers_[index_layer]->CorrectWeights(
          hidden_layers_[index_layer - 1]->get_output_matrix(), learning_rate_);
    } else {
      hidden_layers_[index_layer]->CorrectWeights(*input_layer_,
                                                  learning_rate_);
    }
  }
}

/*–––––––––––––––––––––––––––––––––––––––––––––––--*/

/*––––––––––– class HiddenLayer –––––––––––––––––*/

MatrixNetwork::HiddenLayer::HiddenLayer(const unsigned &rows_weight_layer,
                                        const unsigned &cols_weight_layer)
    : m_output_(nullptr),
      m_weights_(nullptr),
      m_weights_delta_(nullptr),
      sum_neirons_(cols_weight_layer) {
  m_weights_ = new S21Matrix(rows_weight_layer, cols_weight_layer);
}

MatrixNetwork::HiddenLayer::~HiddenLayer() {
  if (m_output_ != nullptr) {
    delete m_output_;
  }
  if (m_weights_ != nullptr) {
    delete m_weights_;
  }
  if (m_weights_delta_ != nullptr) {
    delete m_weights_delta_;
  }
}

void MatrixNetwork::HiddenLayer::LoadWeights(std::ifstream *stream) {
  std::string line{};
  std::string value{};

  size_t index_row = 0;
  size_t index_col = 0;

  while (!stream->eof()) {
    std::getline(*stream, line);

    if (line == "Layer weights are over") {
      break;  // для перехода к следующему слою
    }

    size_t line_size = line.size();
    for (size_t i = 0; i < line_size; ++i) {
      if (line[i] == ' ') {
        (*m_weights_)(index_row, index_col) = std::stod(value);
        ++index_col;
        value.clear();
      } else if (i == line_size - 1) {
        value.push_back(line[i]);
        (*m_weights_)(index_row, index_col) = std::stod(value);
        index_col = 0;
        value.clear();
      } else {
        value.push_back(line[i]);
      }
    }
    ++index_row;
  }
}

void MatrixNetwork::HiddenLayer::SaveWeights(std::ofstream *stream) {
  size_t row = m_weights_->get_rows();
  size_t col = m_weights_->get_columns();

  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      if (j != col - 1) {
        *stream << (*m_weights_)(i, j) << " ";
      } else {
        *stream << (*m_weights_)(i, j) << std::endl;
      }
    }
  }

  *stream << "Layer weights are over" << std::endl;
}

void MatrixNetwork::HiddenLayer::CorrectWeights(
    const S21Matrix &output_matrix_prev_layer, const double &learning_rate) {
  if (m_weights_delta_ == nullptr) {
    throw std::out_of_range("can't correct weight, delta matrix is nullptr");
  }

  size_t r_m_weights = m_weights_->get_rows();
  size_t c_m_weights = m_weights_->get_columns();

  for (size_t col = 0; col < c_m_weights; ++col) {
    for (size_t row = 0; row < r_m_weights; ++row) {
      (*m_weights_)(row, col) = (*m_weights_)(row, col) +
                                (output_matrix_prev_layer(0, row) *
                                 (*m_weights_delta_)(0, col) * learning_rate);
    }
  }
}

void MatrixNetwork::HiddenLayer::InstallRandomWeights() {
  int rows = m_weights_->get_rows();
  int columns = m_weights_->get_columns();

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      (*m_weights_)(i, j) = (rand() % 201 - 100) * 0.01;
    }
  }
}

void MatrixNetwork::HiddenLayer::CalcOutputMatrix(
    const S21Matrix &output_matrix_prev_layer) {
  if (m_output_ != nullptr) {
    delete m_output_;
  }
  m_output_ = new S21Matrix(output_matrix_prev_layer);
  m_output_->MulMatrixWithSigmoid(*m_weights_);
}

void MatrixNetwork::HiddenLayer::CalcWeightsDeltaMatrix(
    const S21Matrix &delta_matrix_prev_layer) {
  if (m_output_ == nullptr) {
    throw std::out_of_range("output matrix is nullptr");
  }

  if (m_weights_delta_ != nullptr) {
    delete m_weights_delta_;
  }
  m_weights_delta_ = new S21Matrix(1, sum_neirons_);

  for (size_t j = 0; j < sum_neirons_; ++j) {
    double sigmoid = (*m_output_)(0, j);
    double sigmoid_dx = sigmoid * (1 - sigmoid);
    double sum_error = 0.0;
    for (int i = 0; i < delta_matrix_prev_layer.get_columns(); ++i) {
      sum_error += (*m_weights_)(j, i) * delta_matrix_prev_layer(0, i);
    }
    (*m_weights_delta_)(0, j) = sigmoid_dx * sum_error;
  }
}

/*----getters HiddenLayer-------*/

const S21Matrix &MatrixNetwork::HiddenLayer::get_output_matrix() {
  if (m_output_ == nullptr) {
    throw std::invalid_argument("can't get output matrix, is nullptr");
  }
  return *m_output_;
}

const S21Matrix &MatrixNetwork::HiddenLayer::get_weights_delta_matrix() {
  if (m_weights_delta_ == nullptr) {
    throw std::invalid_argument("can't get delta matrix, is nullptr");
  }
  return *m_weights_delta_;
}

const size_t &MatrixNetwork::HiddenLayer::get_sum_neirons() {
  return sum_neirons_;
}

/*-----print functions-----*/

void MatrixNetwork::HiddenLayer::print_weghts() {
  std::cout << "----weights----\n";
  for (int i = 0; i < m_weights_->get_rows(); ++i) {
    for (int j = 0; j < m_weights_->get_columns(); ++j) {
      std::cout << (*m_weights_)(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "----weights----\n\n";
}

void MatrixNetwork::HiddenLayer::print_output_values() {
  std::cout << "----Signals neirons----" << std::endl;
  for (int i = 0; i < m_output_->get_rows(); ++i) {
    for (int j = 0; j < m_output_->get_columns(); ++j) {
      std::cout << (*m_output_)(i, j) << std::endl;
    }
  }
  std::cout << "----Signals neirons----\n\n" << std::endl;
}

void MatrixNetwork::HiddenLayer::print_weights_delta() {
  std::cout << "----delta----" << std::endl;
  for (int i = 0; i < m_weights_delta_->get_rows(); ++i) {
    for (int j = 0; j < m_weights_delta_->get_columns(); ++j) {
      std::cout << (*m_weights_delta_)(i, j) << std::endl;
    }
  }
  std::cout << "----delta----\n\n" << std::endl;
}

/*–––––––––––––––––––––––––––––––––––––––––––––––*/

/*––––––––––– class OutputLayer –––––––––––––––––*/

MatrixNetwork::OutputLayer::OutputLayer(const unsigned &rows_weight_layer,
                                        const unsigned &cols_weight_layer)
    : MatrixNetwork::HiddenLayer::HiddenLayer(rows_weight_layer,
                                              cols_weight_layer) {}

void MatrixNetwork::OutputLayer::CalcWeightsDeltaMatrix() {
  if (this->m_output_ == nullptr) {
    throw std::out_of_range("output matrix is nullptr");
  }

  if (this->m_weights_delta_ != nullptr) {
    delete this->m_weights_delta_;
  }
  this->m_weights_delta_ = new S21Matrix(1, this->sum_neirons_);

  for (size_t j = 0; j < this->sum_neirons_; ++j) {
    double sigmoid = (*this->m_output_)(0, j);
    double sigmoid_dx = sigmoid * (1 - sigmoid);
    if (j + 1 == expected_value_) {
      (*m_weights_delta_)(0, j) = (1.0 - sigmoid) * sigmoid_dx;
    } else {
      (*m_weights_delta_)(0, j) = (0.0 - sigmoid) * sigmoid_dx;
    }
  }
}

void MatrixNetwork::OutputLayer::set_expected_value(const size_t &value) {
  expected_value_ = value;
}

const size_t &MatrixNetwork::OutputLayer::get_expected_value() {
  return expected_value_;
}

size_t MatrixNetwork::OutputLayer::ResultNeiron() {
  if (this->m_output_ == nullptr) {
    throw std::out_of_range(
        "Error int resultNeiron(), outputMatrix is nullptr");
  }
  std::pair<double, size_t> result{(*m_output_)(0, 0), 0};

  /*---матрица значений состоит только из одной строки---*/
  int columns = m_output_->get_columns();
  for (int j = 0; j < columns; ++j) {
    if (result.first < (*m_output_)(0, j)) {
      result.first = (*m_output_)(0, j);
      result.second = j;
    }
  }

  /*---прибавляем единицу к ответу, чтобы исключить ноль---*/
  return (result.second + 1);
}

/*–––––––––––––––––––––––––––––––––––––––––––––––--*/

}  // namespace s21_network
