#### Prerequisite

Intall `bazel`, `eigen3`, `gdb`.

#### Build

Before building, change decision variable size `kHalfVarSize` in `tests/rosenbrock_optimizer_test.cc`

```sh
bazel build //...
```

#### Test

```sh
gdb bazel-bin/tests/rosenbrock_optimizer_test
run
```
