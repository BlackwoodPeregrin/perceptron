#include <gtest/gtest.h>

#include "s21_matrix_oop.h"
#include "neuron.h"

constexpr double kEPS = 1e-7;

TEST(mx_oop, simple) {
  S21Matrix A(3, 3);
  S21Matrix B(A);
  S21Matrix C(std::move(B));
  A.Clear();
  ASSERT_EQ(A.get_columns(), 0);
  ASSERT_EQ(A.get_rows(), 0);
}

TEST(mx_oop, mul) {
  S21Matrix A(1, 1);
  A(0, 0) = 2;
  S21Matrix B(1, 1);
  B(0, 0) = 3;
  S21Matrix C = A * B;
  ASSERT_NEAR(6.0, C(0, 0), kEPS);
  S21Matrix D(1, 1);
  A.MulMatrixWithSigmoid(D);
  ASSERT_NEAR(0.5, A(0, 0), kEPS);
  B *= D;
  ASSERT_NEAR(0.0, B(0, 0), kEPS);
}

TEST(mx_oop, search) {
  S21Matrix A(1, 3);
  A(0, 0) = 1;
  A(0, 1) = 3;
  A(0, 2) = 2;
  ASSERT_EQ(1, A.SearchMaxElement().second);
}

TEST(neuron, set_get) {
  s21_network::Neuron A{};
  A.set_value(1.1);
  ASSERT_NEAR(1.1, A.get_value(), kEPS);
  A.set_deriv(0.9);
  ASSERT_NEAR(0.9, A.get_deriv(), kEPS);
}

TEST(neuron, simple_net) {
  s21_network::Neuron A{}, B{};
  B.AddInput(&A, 1);
  A.set_value(1);
  B.Activate();
  ASSERT_NEAR(1.0, B.get_value(), kEPS);
  B.weight(0) = 0;
  B.set_mode(s21_network::kSigmoid);
  B.Activate();
  ASSERT_NEAR(0.5, B.get_value(), kEPS);
  B.ClearInput();
}

TEST(neuron, correction) {
  s21_network::Neuron A{}, B{};
  B.AddInput(&A, 1);
  A.set_value(1);
  B.set_mode(s21_network::kSigmoid);
  B.Activate();
  B.set_deriv(0);
  B.CorrectWeights(0.1);
  ASSERT_NEAR(B.weight(0), 1, kEPS);
  B.set_deriv(0.1);
  B.CorrectWeights(0.1);
  ASSERT_LT(B.weight(0), 1);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
