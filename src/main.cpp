#include <Eigen/Core>
#include <iostream>
#include <ostream>
#include <vector>

#include "layer.hpp"
#include "network.hpp"
#include "funcs.hpp"

int main(int, char**){
    std::srand(static_cast<unsigned int>(time(nullptr)));

    Eigen::VectorXd testData(784);
    testData.setRandom();

    Layer hidden1(784, 64, ReLU, ReLU_Derivative);
    Layer hidden2(64, 64, ReLU, ReLU_Derivative);
    Layer output(64, 10, Identity, Identity_Derivative);

    NeuralNetwork net({hidden1, hidden2, output});

    net.forwardPass(testData);

    Eigen::Index predicted_index;
    double max = net.getOutput().maxCoeff(&predicted_index);

    std::cout << predicted_index << std::endl;
    std::cout << max << std::endl;
}
