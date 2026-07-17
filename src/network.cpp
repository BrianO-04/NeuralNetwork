#include "network.hpp"
#include <Eigen/Core>
#include "funcs.hpp"

NeuralNetwork::NeuralNetwork(std::vector<Layer> l){
    layers = l;
}

Eigen::VectorXd NeuralNetwork::getOutput(){
    return layers.back().getActiveNeurons();
}

void NeuralNetwork::forwardPass(Eigen::VectorXd& input){
    layers[0].forward(input);
    for(int i = 1; i < layers.size(); i++){
        layers[i].forward(layers[i-1].getActiveNeurons());
    }
}

void NeuralNetwork::backPass(Eigen::VectorXd &input, Eigen::VectorXd &target, double learning_rate){
    Eigen::VectorXd out_error = error_func_derivative(getOutput(), target);
    Eigen::VectorXd out_derivatives = layers.back().activationDerivative();
    Eigen::VectorXd out_delta = out_error.cwiseProduct(out_derivatives);
    layers.back().setDelta(out_delta);

}

void NeuralNetwork::train(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targets, double learning_rate, int epochs){

}