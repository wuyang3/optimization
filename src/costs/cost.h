#pragma once

#include <eigen3/Eigen/Dense>

class Cost {
 public:
  virtual double Evaluate(const Eigen::Ref<const Eigen::VectorXd> &x) = 0;
  virtual void Gradient(const Eigen::Ref<const Eigen::VectorXd> &x,
                        Eigen::Ref<Eigen::VectorXd> gradient) = 0;
  virtual void Hessian(const Eigen::Ref<const Eigen::VectorXd> &x,
                       Eigen::Ref<Eigen::MatrixXd> hessian) = 0;
};
