#pragma once

#include <Eigen/Dense>

Eigen::Matrix3d createRotationMatrix(const Eigen::Vector3d& axis);
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);
double GMLoss(double error, double sigma);
double HuberLoss(double error, double delta);
