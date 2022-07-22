#pragma once

#include <chrono>
#include <memory>

#include "src/optimizer/optimizer.h"

template <class CostType>
class SteepestDescent : public Optimizer {
 public:
  static constexpr int NEigen =
      CostType::NDecisionVar > 16 ? -1 : CostType::NDecisionVar;

  SteepestDescent() {
    if (NEigen == -1) {
      init_var_.resize(CostType::NDecisionVar);
      optimal_var_.resize(CostType::NDecisionVar);
      gradient_.resize(CostType::NDecisionVar);
    }
    init_var_.setZero();
    optimal_var_.setZero();
    gradient_.setZero();
  }
  ~SteepestDescent() = default;

  // class specific API.
  void set_cost(const std::shared_ptr<CostType>& cost) { cost_ = cost; }

  // common inherited API.
  bool Solve() override;
  void set_init_var(
      const Eigen::Ref<const Eigen::VectorXd>& init_var) override {
    init_var_ = init_var;
    optimal_var_ = init_var;
  }
  VectorRef init_var() override { return init_var_; }
  VectorRef optimal_var() override { return optimal_var_; }
  double solve_time() override {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               solve_end_time_ - solve_start_time_)
        .count();
  }
  SolverFlag solve_flag() override { return solve_flag_; }

 private:
  std::shared_ptr<CostType> cost_{nullptr};

  // solver intermediate variables.
  Eigen::Matrix<double, NEigen, 1> init_var_;
  Eigen::Matrix<double, NEigen, 1> optimal_var_;
  Eigen::Matrix<double, NEigen, 1> gradient_;

  // solver parameters.
  int max_iteration_ = 1e6;
  double gradient_tolerance_ = 1e-6;
  double c_ = 0.5;
  double tau_ = 1.0;

  // solver statistics.
  std::chrono::time_point<std::chrono::high_resolution_clock> solve_start_time_;
  std::chrono::time_point<std::chrono::high_resolution_clock> solve_end_time_;

  SolverFlag solve_flag_ = SolverFlag::SOLVER_UNSOLVED;
};

/********************** Definition ************************/
template <class CostType>
bool SteepestDescent<CostType>::Solve() {
  if (cost_ == nullptr) {
    std::cout << "SteepestDescent: cost not initialized.";

    return false;
  }

  solve_start_time_ = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < max_iteration_; iter++) {
    cost_->Gradient(optimal_var_, gradient_);

    // SOLVER_SOLVED.
    double gradient_norm = gradient_.norm();
    if (gradient_norm < gradient_tolerance_) {
      solve_flag_ = SolverFlag::SOLVER_SOLVED;

      std::cout << "Solver iter: " << iter << std::endl;
      break;
    }

    // Update.
    while (cost_->Evaluate(optimal_var_ - tau_ * gradient_) -
               cost_->Evaluate(optimal_var_) >
           -c_ * tau_ * gradient_norm * gradient_norm) {
      tau_ *= 0.5;
    }
    optimal_var_ = optimal_var_ - tau_ * gradient_;

    // SOLVER_MAX_ITERATION.
    solve_flag_ = (iter == max_iteration_ - 1)
                      ? SolverFlag::SOLVER_MAX_ITERATION
                      : solve_flag_;
  }

  solve_end_time_ = std::chrono::high_resolution_clock::now();

  return solve_flag_ == SolverFlag::SOLVER_SOLVED;
}
