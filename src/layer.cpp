#include "layer.hpp"

Layer::Layer(int input_size_n, int neurons, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> ac_func, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> ac_func_deriv){
    weights = Eigen::MatrixXd::Random(neurons, input_size_n) * std::sqrt(6.0 / static_cast<double>(input_size_n));
    biases = Eigen::VectorXd::Zero(neurons);
    Activation = ac_func;
    Activation_derivative = ac_func_deriv;
}

void Layer::forward(const Eigen::VectorXd &in){
    Eigen::VectorXd Y = weights * in + biases;
    neuronVals = Y;
    neuronActivations = Activation(Y);
}

Eigen::VectorXd Layer::getActiveNeurons(){
    return neuronActivations;
}

void Layer::setDelta(const Eigen::VectorXd &d){
    delta = d;
}

Eigen::VectorXd Layer::getDelta(){
    return delta;
}

Eigen::VectorXd Layer::getBiases(){
    return biases;
}

Eigen::MatrixXd Layer::getWeights(){
    return weights;
}

void Layer::updateWeights(const Eigen::VectorXd &input, double learning_rate){
    weights -= learning_rate * delta * input.transpose();
    biases -= learning_rate * delta;
}

Eigen::VectorXd Layer::getActiveDerivative(){
    return Activation_derivative(neuronVals);
}

void Layer::setWeights(Eigen::MatrixXd w, Eigen::VectorXd b, Eigen::Index rows, Eigen::Index cols, Eigen::Index bSize){
    weights.resize(rows, cols);
    weights = w;

    biases.resize(bSize);
    biases = b;
}