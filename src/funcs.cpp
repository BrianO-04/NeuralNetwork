#include "funcs.hpp"

Eigen::VectorXd ReLU(const Eigen::VectorXd &v){
    return v.array().max(0.0);
}

Eigen::VectorXd ReLU_Derivative(const Eigen::VectorXd &v){
    return (v.array() > 0.0).cast<double>();
}

Eigen::VectorXd Identity(const Eigen::VectorXd &v){
    return v;
}

Eigen::VectorXd Identity_Derivative(const Eigen::VectorXd &v){
    return Eigen::VectorXd::Ones(v.size());
}

Eigen::VectorXd SoftMax(const Eigen::VectorXd &logits){
    const double max = logits.maxCoeff();

    Eigen::VectorXd exponentials = (logits.array() - max).exp();

    return exponentials / exponentials.sum();
}

double CrossEntropy(const Eigen::VectorXd &probabilities, const Eigen::VectorXd &target){
    double epsilon = 1e-12;
    return -(target.array() * probabilities.array().max(epsilon).log()).sum();
}