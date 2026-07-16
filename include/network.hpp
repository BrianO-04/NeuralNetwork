#ifndef NEURAL_H_
#define NEURAL_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

#include "layer.hpp"

class NeuralNetwork{
private:
    std::vector<Layer> layers;
public:
    NeuralNetwork(std::vector<Layer> l);
    Eigen::VectorXd getOutput();
    void backPass(Eigen::VectorXd& input, Eigen::VectorXd& target, double learning_rate);
    void train(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets, double learning_rate, int epochs);
};

#endif