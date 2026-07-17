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

double error_func(const Eigen::VectorXd &out, const Eigen::VectorXd &target){
    return 0.5 * (out - target).squaredNorm();
}

Eigen::VectorXd error_func_derivative(const Eigen::VectorXd &out, const Eigen::VectorXd &target){
    return out - target;
}