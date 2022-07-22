#pragma once

#include <eigen3/Eigen/Dense>

enum class SolverFlag : int {
  SOLVER_UNSOLVED = 0,
  SOLVER_SOLVED = 1,
  SOLVER_MAX_ITERATION = 2
};

class Optimizer {
 public:
  using VectorRef = const Eigen::Ref<const Eigen::VectorXd>;

  virtual bool Solve() = 0;
  virtual void set_init_var(
      const Eigen::Ref<const Eigen::VectorXd>& init_var) = 0;

  virtual VectorRef init_var() = 0;
  virtual VectorRef optimal_var() = 0;

  // In millisecond.
  virtual double solve_time() = 0;
  virtual SolverFlag solve_flag() = 0;
};
