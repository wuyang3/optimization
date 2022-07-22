cc_test(
    name = "rosenbrock_optimizer_test",
    srcs = [
        "rosenbrock_optimizer_test.cc"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/costs:rosenbrock",
        "//src/optimizer:steepest_descent",
        "@googletest//:gtest_main",
    ],
)
