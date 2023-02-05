#include "controller.hpp"

namespace s21_network {
Controller::Controller() : network_(new Network(kLearningRate)) {}

Controller::~Controller() { delete network_; }

}
