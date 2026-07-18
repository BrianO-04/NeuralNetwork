#ifndef FUNCS_H_
#define FUNCS_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>

Eigen::VectorXd ReLU(const Eigen::VectorXd &v);

Eigen::VectorXd ReLU_Derivative(const Eigen::VectorXd &v);

Eigen::VectorXd Identity(const Eigen::VectorXd &v);

Eigen::VectorXd Identity_Derivative(const Eigen::VectorXd &v);

double error_func(const Eigen::VectorXd &out, const Eigen::VectorXd &target);

Eigen::VectorXd error_func_derivative(const Eigen::VectorXd &out, const Eigen::VectorXd &target);

Eigen::VectorXd SoftMax(const Eigen::VectorXd& logits);

double CrossEntropy(const Eigen::VectorXd& probabilities, const Eigen::VectorXd& target);

#endif // FUNCS_H_