
#include <memory>

#include "gtest/gtest.h"
#include "src/costs/rosenbrock.h"
#include "src/optimizer/steepest_descent.h"

constexpr int kHalfVarSize = 1;

TEST(RosenbrockOptimizerTest, RosenbrockSteepestDescentTest) {
  const int NEigen = 2 * kHalfVarSize > 16 ? -1 : 2 * kHalfVarSize;
  Eigen::Matrix<double, NEigen, 1> init_var;
  if (NEigen == -1) {
    init_var.resize(2 * kHalfVarSize);
  }
  init_var.setZero();
  Eigen::Matrix<double, NEigen, 1> optimal_var = init_var;

  std::shared_ptr<RosenBrock<kHalfVarSize>> rosenbrock =
      std::make_shared<RosenBrock<kHalfVarSize>>();
  SteepestDescent<RosenBrock<kHalfVarSize>> steepest_descent_optimizer;

  steepest_descent_optimizer.set_cost(rosenbrock);
  steepest_descent_optimizer.set_init_var(init_var);

  bool solved = steepest_descent_optimizer.Solve();
  optimal_var = steepest_descent_optimizer.optimal_var();

  std::cout << "Solver flag: "
            << static_cast<int>(steepest_descent_optimizer.solve_flag())
            << std::endl;
  std::cout << "Solver time: " << steepest_descent_optimizer.solve_time()
            << "ms" << std::endl;
  std::cout << "Solution: " << std::endl
            << "[" << optimal_var << "]" << std::endl;

  EXPECT_TRUE(solved);

  for (size_t i = 0; i < 2 * kHalfVarSize; i++) {
    EXPECT_NEAR(1.0, optimal_var(i), 1e-4);
  }
}
