#include "layer.hpp"
#include <Eigen/Core>

Layer::Layer(int input_size_n, int neurons, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> ac_func, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> ac_func_deriv){
    weights = Eigen::MatrixXd::Random(neurons, input_size_n) * std::sqrt(6.0 / static_cast<double>(input_size_n));
    biases = Eigen::VectorXd::Zero(neurons);
    activation_func = ac_func;
    activation_func_derivative = ac_func_deriv;
}

void Layer::forward(const Eigen::VectorXd &in){
    Eigen::VectorXd Y = weights * in + biases;
    neuron_vals = Y;
    neuron_activations = activation_func(Y);
}

Eigen::VectorXd Layer::getActiveNeurons(){
    return neuron_activations;
}

void Layer::setDelta(const Eigen::VectorXd &d){
    delta = d;
}

Eigen::VectorXd Layer::getDelta(){
    return delta;
}

Eigen::MatrixXd Layer::getWeights(){
    return weights;
}

void Layer::updateWeights(const Eigen::VectorXd &input, double learning_rate){
    weights -= learning_rate * delta * input.transpose();
    biases -= learning_rate * delta;
}

Eigen::VectorXd Layer::activationDerivative(){
    return activation_func_derivative(neuron_vals);
}