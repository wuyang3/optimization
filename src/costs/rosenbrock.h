#pragma once

#include "src/costs/cost.h"

template <int NHalf>
class RosenBrock : public Cost {
 public:
  static constexpr int NDecisionVar = NHalf * 2;
  RosenBrock() = default;
  ~RosenBrock() = default;

  // All arguements are assumed to be well resized and zeroed.
  double Evaluate(const Eigen::Ref<const Eigen::VectorXd> &x) override {
    double cost = 0.0;

    for (int i = 0; i < NHalf; i++) {
      cost += 100 * (x(2 * i) * x(2 * i) - x(2 * i + 1)) *
                  (x(2 * i) * x(2 * i) - x(2 * i + 1)) +
              (x(2 * i) - 1) * (x(2 * i) - 1);
    }

    return cost;
  }

  void Gradient(const Eigen::Ref<const Eigen::VectorXd> &x,
                Eigen::Ref<Eigen::VectorXd> gradient) override {
    for (int i = 0; i < NHalf; i++) {
      gradient(2 * i) = 400 * (x(2 * i) * x(2 * i) - x(2 * i + 1)) * x(2 * i) +
                        2 * (x(2 * i) - 1);
      gradient(2 * i + 1) = -200 * (x(2 * i) * x(2 * i) - x(2 * i + 1));
    }
  }

  void Hessian(const Eigen::Ref<const Eigen::VectorXd> &x,
               Eigen::Ref<Eigen::MatrixXd> hessian) override {
    // currently not used.
    hessian.setZero();
  }

 private:
};
