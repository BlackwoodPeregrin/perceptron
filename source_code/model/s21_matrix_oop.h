#pragma once

#include <cmath>
#include <iostream>

class S21Matrix {
 private:
  int rows_, columns_;
  double** matrix_;

 private:
  double SigmoidFunc(const double& value) { return (1 / (1 + exp(-value))); }

  void DistributionMemory(const int& rows, const int& columns) {
    rows_ = rows;
    columns_ = columns;
    matrix_ = new double*[rows_];
    for (int i = 0; i < rows_; i++) {
      matrix_[i] = new double[columns_]{};
    }
  }

  void CheckDimensionMatrix(const S21Matrix& other) {
    if (columns_ != other.rows_) {
      throw std::invalid_argument(
          "ERROR in mult matrix columns unequal rows mult matrix");
    }
  }

 public:
  // constructors
  S21Matrix(const int& rows, const int& columns) : matrix_(nullptr) {
    if (rows < 1 || columns < 1) {
      throw std::invalid_argument("ERROR, invalid input");
    }
    DistributionMemory(rows, columns);
  }

  S21Matrix(const S21Matrix& other) : matrix_(nullptr) { *this = other; }

  S21Matrix(S21Matrix&& other) : matrix_(nullptr) {
    Clear();
    Swap(other);
  }
  // destructor
  ~S21Matrix() { Clear(); }

  /* Clear memory in matrix */
  void Clear() {
    if (matrix_ != nullptr) {
      for (int i = 0; i < rows_; i++) {
        delete[] matrix_[i];
        matrix_[i] = nullptr;
      }
      delete[] matrix_;
      matrix_ = nullptr;
      rows_ = 0;
      columns_ = 0;
    }
  }

  void Swap(S21Matrix& other) {
    if (this == &other) {
      throw std::out_of_range("Error, in swap(), you try swap yourself");
    }
    std::swap(rows_, other.rows_);
    std::swap(columns_, other.columns_);
    std::swap(matrix_, other.matrix_);
  }

  /* accessors */
  int get_rows() const { return rows_; }
  int get_columns() const { return columns_; }

  void MulMatrix(const S21Matrix& other) {
    CheckDimensionMatrix(other);
    S21Matrix result(rows_, other.columns_);
    for (int i = 0; i < result.rows_; i++) {
      for (int j = 0; j < result.columns_; j++) {
        for (int k = 0; k < columns_; k++) {
          result.matrix_[i][j] += matrix_[i][k] * other.matrix_[k][j];
        }
      }
    }
    Swap(result);
  }

  void MulMatrixWithSigmoid(const S21Matrix& other) {
    CheckDimensionMatrix(other);
    S21Matrix result(rows_, other.columns_);
    for (int i = 0; i < result.rows_; i++) {
      for (int j = 0; j < result.columns_; j++) {
        for (int k = 0; k < columns_; k++) {
          result.matrix_[i][j] += matrix_[i][k] * other.matrix_[k][j];
        }
        result.matrix_[i][j] = SigmoidFunc(result.matrix_[i][j]);
      }
    }
    Swap(result);
  }

  /* operators overloads */
  S21Matrix& operator=(const S21Matrix& other) {
    if (this == &other) {
      throw std::invalid_argument(
          "ERROR in costruct, can't copy or move yourself");
    }
    Clear();
    DistributionMemory(other.rows_, other.columns_);
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < columns_; j++) {
        matrix_[i][j] = other.matrix_[i][j];
      }
    }
    return *this;
  }

  S21Matrix operator*(const S21Matrix& other) {
    S21Matrix newMatrix(*this);
    newMatrix.MulMatrix(other);
    return newMatrix;
  }

  S21Matrix& operator*=(const S21Matrix& other) {
    MulMatrix(other);
    return *this;
  }

  double& operator()(const int& row, const int& column) {
    if (rows_ <= row || columns_ <= column) {
      throw std::out_of_range("ERROR index out of range");
    }
    return matrix_[row][column];
  }

  double operator()(const int& row, const int& column) const {
    if (rows_ <= row || columns_ <= column) {
      throw std::out_of_range("ERROR index out of range");
    }
    return matrix_[row][column];
  }

  std::pair<int, int> SearchMaxElement() {
    if (matrix_ == nullptr) {
      throw std::out_of_range("matrix is nullptr, can't search max element");
    }

    std::pair<int, int> result{0, 0};
    double max_element = matrix_[0][0];

    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < columns_; ++j) {
        if (max_element < matrix_[i][j]) {
          max_element = matrix_[i][j];
          result = {i, j};
        }
      }
    }
    return result;
  }
};
