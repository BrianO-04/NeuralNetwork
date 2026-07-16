#ifndef LAYER_H_
#define LAYER_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <functional>

class Layer{
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd neuron_vals;
    Eigen::VectorXd neuron_activations;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_func;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_func_derivative;
    Eigen::VectorXd delta;
public:
    Layer(int input_size_n, int neurons, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> ac_func, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> ac_func_deriv);
    void forward(Eigen::VectorXd &in);
    Eigen::VectorXd getActiveNeurons();
    void setDelta(const Eigen::VectorXd& d);
    Eigen::VectorXd getDelta();
    Eigen::MatrixXd getWeights();
    void updateWeights(const Eigen::VectorXd& input, double learning_rate);
};

#endif // LAYER_H_