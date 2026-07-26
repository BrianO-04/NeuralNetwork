#include "network.hpp"
#include "funcs.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

NeuralNetwork::NeuralNetwork(std::vector<Layer> _layers){
    layers = _layers;
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

void NeuralNetwork::backPass(Eigen::VectorXd& input, Eigen::VectorXd& probabilities, Eigen::VectorXd& target, double learningRate){
    Eigen::VectorXd out_delta = probabilities - target;
    layers.back().setDelta(out_delta);

    for(int i = layers.size()-2; i >= 0; i--){
        Eigen::MatrixXd next_weights = layers[i+1].getWeights();
        Eigen::VectorXd next_delta = layers[i+1].getDelta();

        Eigen::VectorXd hidden_error = next_weights.transpose() * next_delta;
        Eigen::VectorXd hidden_derivative = layers[i].getActiveDerivative();
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
        layers[i].updateWeights(in, learningRate);
    }
}

void NeuralNetwork::train(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targets, double learningRate, int epochs){
    for(int epoch = 0; epoch < epochs; epoch++){
        double epochLoss = 0.0;

        for(ssize_t i = 0; i < inputs.size(); i++){
            Eigen::VectorXd input = inputs[i];
            Eigen::VectorXd target = targets[i];

            forwardPass(input);

            Eigen::VectorXd probs = SoftMax(getOutput());
            double loss = CrossEntropy(probs, target);
            epochLoss += loss;

            backPass(input, probs, target, learningRate);
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

int NeuralNetwork::exportWeights(){
    std::string fileName = "weights/Weights.bin";

    std::ofstream outFile(fileName, std::ios::out | std::ios::binary);

    if (!outFile) {
        std::cerr << "Error creating data file!" << std::endl;
        return 1;
    }

    // Layer Count Header
    int layerCount = layers.size();
    outFile.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));

    // Layer saving
    for(int i = 0; i < layers.size(); i++){
        const std::int64_t rows = layers[i].getWeights().rows();
        const std::int64_t cols = layers[i].getWeights().cols();
        Eigen::MatrixXd weights = layers[i].getWeights();
        outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

        outFile.write(
            reinterpret_cast<const char*>(layers[i].getWeights().data()),
            static_cast<std::streamsize>(
                layers[i].getWeights().size() * sizeof(Eigen::MatrixXd::Scalar)
            )
        );

        const auto& biases = layers[i].getBiases();
        std::int64_t biasSize = biases.size();

        outFile.write(
            reinterpret_cast<const char*>(&biasSize),
            sizeof(biasSize)
        );

        outFile.write(
            reinterpret_cast<const char*>(biases.data()),
            biasSize * sizeof(Eigen::VectorXd::Scalar)
        );

        if(!outFile){
            std::cerr << "Error while writing matrix data.\n";
            return 1;
        }
    }

    return 0;
}

int NeuralNetwork::loadWeights(std::string filename){
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);

    if (!inFile) {
        std::cerr << "Error loading data file!" << std::endl;
        return 1;
    }

    // Layer Count Header
    int layerCount = 0;
    inFile.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));

    // Layer loading
    for(int i = 0; i < layers.size(); i++){
        std::int64_t rows = 0;
        std::int64_t cols = 0;
        
        inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        if(!inFile){
            std::cerr << "Could not read matrix dimensions.\n";
            return false;
        }

        Eigen::MatrixXd matrix;
        matrix.resize(rows, cols);

        inFile.read(
            reinterpret_cast<char*>(matrix.data()),
            static_cast<std::streamsize>(
                matrix.size() * sizeof(Eigen::MatrixXd::Scalar)
            )
        );

        Eigen::VectorXd biases(1);
        std::int64_t biasSize = 0;

        inFile.read(
            reinterpret_cast<char*>(&biasSize),
            sizeof(biasSize)
        );

        biases.resize(biasSize);

        inFile.read(
            reinterpret_cast<char*>(biases.data()),
            biasSize * sizeof(Eigen::VectorXd::Scalar)
        );

        layers[i].setWeights(matrix, biases, rows, cols, biasSize);
    }

    return 0;
}