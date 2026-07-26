#ifndef NEURAL_H_
#define NEURAL_H_

#include <Eigen/Dense>
#include <vector>
#include "layer.hpp"

class NeuralNetwork{
private:
    std::vector<Layer> layers;
public:
    NeuralNetwork(std::vector<Layer> _layers);
    Eigen::VectorXd getOutput();
    void forwardPass(Eigen::VectorXd& input);
    void backPass(Eigen::VectorXd& input, Eigen::VectorXd& probabilities, Eigen::VectorXd& target, double learningRate);
    void train(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets, double learningRate, int epochs);
    void test(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets);
    int exportWeights();
    int loadWeights(std::string filename);
};

#endif