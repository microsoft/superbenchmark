# cuDNN Regression Checks

Run these commands from the repository root in an environment with the supported
CUDA toolkit, cuDNN development libraries, C++ compiler, CMake, and a compatible
NVIDIA GPU. Python parser checks require the normal SuperBench Python dependencies
but do not launch a GPU workload. CMake fetches the existing pinned JSON dependency
unless it is supplied through `FETCHCONTENT_SOURCE_DIR_JSON`.

## Build And Run

```bash
cmake -S superbench/benchmarks/micro_benchmarks/cudnn_function \
  -B build/cudnn-tests -DCMAKE_BUILD_TYPE=Release
cmake --build build/cudnn-tests \
  --target cudnn_benchmark cudnn_data_types_test --parallel 2
CUDA_VISIBLE_DEVICES=0 build/cudnn-tests/cudnn_data_types_test
CUDA_VISIBLE_DEVICES=0 \
  SB_CUDNN_TEST_BINARY="$PWD/build/cudnn-tests/cudnn_benchmark" \
  python3 tests/benchmarks/micro_benchmarks/test_cudnn_binary.py -v
python3 -m unittest discover -s tests/benchmarks/micro_benchmarks \
  -p test_cudnn_raw_result.py -v
```

With CUDA 13 installations that place CCCL headers under the toolkit's `include/cccl`
directory, add `-DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include/cccl` when configuring
(adjust the toolkit path). Use the toolkit and architecture flags appropriate to
the target GPU; a build on one architecture is not runtime qualification of another.

The native test target is excluded from the default build and installation. It is
available when the repository's test source is present, so native-source-only
packages can still configure and build without the tests. The CLI tests skip when
`SB_CUDNN_TEST_BINARY` is unset; set it explicitly for a real native gate.

## Coverage And Limits

- Native checks cover independent storage/compute fields, descriptor readback,
  seeded half initialization and device transfer, byte-exact workspace requests,
  cleanup, unprepared destruction, strided allocation, and invalid descriptor arrays.
- CLI checks exercise all three convolution directions and controlled failures.
  FP16 compute and noncontiguous layouts may report `CUDNN_STATUS_NOT_SUPPORTED`;
  those paths must exit nonzero without publishing timings. The FP32-storage/FP32-
  compute and FP16-storage/FP32-compute contiguous smoke cases must execute.
- Parser checks preserve names, units, raw output, and the failure marker while
  rejecting absent, duplicate, malformed, nonpositive, or non-finite measurements.
  They also cover the existing executor's aggregate failure and continuation behavior.
- These are execution and integrity checks, not a full numerical-accuracy test,
  performance acceptance, or certification of every default algorithm on every stack.

For a memory check, run:

```bash
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build/cudnn-tests/cudnn_data_types_test --memcheck
```

`--memcheck` skips only the deliberate query of a freed workspace address. The
ordinary test retains that negative check, while sanitizer leak detection checks
that allocations are released. Require zero memory errors and zero leaked bytes.

The repaired FP16-storage/FP32-compute cases need new baselines. See the
[precision and compatibility note](../../../../docs/user-tutorial/benchmarks/micro-benchmarks.md#precision-and-compatibility).
