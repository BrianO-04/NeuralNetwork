#ifndef LAYER_H_
#define LAYER_H_

#include <Eigen/Core>
#include <Eigen/Dense>

class Layer{
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases, neuronVals, neuronActivations, delta;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> Activation, Activation_derivative;
public:
    Layer(int input_size_n, int neurons, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> ac_func, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> ac_func_deriv);
    void forward(const Eigen::VectorXd &in);
    void setDelta(const Eigen::VectorXd& d);
    void updateWeights(const Eigen::VectorXd& input, double learning_rate);
    Eigen::VectorXd getActiveNeurons();
    Eigen::VectorXd getActiveDerivative();
    Eigen::VectorXd getDelta();
    Eigen::VectorXd getBiases();
    Eigen::MatrixXd getWeights();
    void setWeights(Eigen::MatrixXd w, Eigen::VectorXd b, Eigen::Index rows, Eigen::Index cols, Eigen::Index bSize);
};

#endif // LAYER_H_