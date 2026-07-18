#include "network.hpp"
#include <Eigen/Core>
#include "funcs.hpp"
#include <iostream>

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

    for(int i = layers.size()-2; i >= 0; i--){
        Eigen::MatrixXd next_weights = layers[i+1].getWeights();
        Eigen::VectorXd next_delta = layers[i+1].getDelta();

        Eigen::VectorXd hidden_error = next_weights.transpose() * next_delta;
        Eigen::VectorXd hidden_derivative = layers[i].activationDerivative();
        Eigen::VectorXd hidden_delta = hidden_error.cwiseProduct(hidden_derivative);

        layers[i].setDelta(hidden_delta);
    }

    for(int i = 0; i < layers.size(); i++){
        Eigen::VectorXd in;
        if(i == 0){
            in = input;
        }else{
            in = layers[i-1].getActiveNeurons();
        }
        layers[i].updateWeights(in, learning_rate);
    }
}

void NeuralNetwork::train(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targets, double learning_rate, int epochs){
    for(int epoch = 0; epoch < epochs; epoch++){
        double epochLoss = 0.0;

        for(ssize_t i = 0; i < inputs.size(); i++){
            Eigen::VectorXd input = inputs[i];
            Eigen::VectorXd target = targets[i];

            forwardPass(input);

            Eigen::VectorXd output = getOutput();
            double loss = error_func(output, target);
            epochLoss += loss;

            backPass(input, target, learning_rate);
        }
        epochLoss /= inputs.size();

        std::cout << "Epoch " << epoch+1 << "/" << epochs << " - Loss: " << epochLoss << std::endl;
    }
}

void NeuralNetwork::test(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targets){
    int correct_predictions = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            Eigen::VectorXd input = inputs[i];
            Eigen::VectorXd target_output = targets[i];

            forwardPass(input);

            Eigen::VectorXd output = getOutput();

            int predicted_index;
            int target_index;
            output.maxCoeff(&predicted_index);
            target_output.maxCoeff(&target_index);

            if (predicted_index == target_index) {
                correct_predictions++;
            }
        }

        double accuracy = static_cast<double>(correct_predictions) / inputs.size();
        std::cout << "Test Accuracy: " << accuracy << std::endl;
}