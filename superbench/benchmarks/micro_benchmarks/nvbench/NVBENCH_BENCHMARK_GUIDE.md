# NVBench Benchmark Development Guide

Guide for GitHub Copilot to create new NVBench-based benchmarks in SuperBench.

## Architecture Overview

NVBench benchmarks follow a modular structure. To add a new benchmark `<name>`:

```
Files to Create/Modify:
├── superbench/benchmarks/micro_benchmarks/nvbench/
│   ├── <name>.cu                          # CUDA benchmark (CREATE)
│   └── CMakeLists.txt                     # Add .cu to NVBENCH_SOURCES (MODIFY)
├── superbench/benchmarks/micro_benchmarks/
│   ├── nvbench_<name>.py                  # Python wrapper (CREATE)
│   └── __init__.py                        # Add import (MODIFY)
├── tests/benchmarks/micro_benchmarks/
│   └── test_nvbench_<name>.py             # Test file (CREATE)
├── tests/data/
│   └── nvbench_<name>.log                 # Sample output for tests (CREATE)
├── examples/benchmarks/
│   └── nvbench_<name>.py                  # Example script (CREATE)
└── docs/user-tutorial/benchmarks/
    └── micro-benchmarks.md                # Add metrics documentation (MODIFY)
```

## Reference Files

When creating a new benchmark, examine these existing implementations:

| Component | Simple (no params) | Parameterized |
|-----------|-------------------|---------------|
| CUDA benchmark | `nvbench/kernel_launch.cu` | `nvbench/sleep_kernel.cu` |
| Python wrapper | `nvbench_kernel_launch.py` | `nvbench_sleep_kernel.py` |
| Test file | `test_nvbench_kernel_launch.py` | `test_nvbench_sleep_kernel.py` |
| Test data | `tests/data/nvbench_kernel_launch.log` | `tests/data/nvbench_sleep_kernel.log` |
| Example | `examples/benchmarks/nvbench_kernel_launch.py` | `examples/benchmarks/nvbench_sleep_kernel.py` |

## Key Components

### Base Class: `nvbench_base.py`
- `NvbenchBase` - Inherit from this class
- `_build_base_command()` - Builds command with common NVBench CLI args
- `parse_time_to_us(str)` - Parses "123.45 us", "678.9 ns", "0.12 ms", "1.5 s" → float µs
- `_handle_parsing_error()` - Consistent error handling (see Error Handling section below)

### CMakeLists.txt
Add new `.cu` file to `NVBENCH_SOURCES` list.

### Python Wrapper Pattern
1. Set `self._bin_name = 'nvbench_<name>'` (must match CMake target)
2. Override `add_parser_arguments()` if benchmark has custom parameters
3. Override `_preprocess()` if custom command building needed
4. Implement `_process_raw_result()` to parse NVBench output

### Registration
- Python: `BenchmarkRegistry.register_benchmark('nvbench-<name>', Nvbench<Name>, platform=Platform.CUDA)`
- Import in `__init__.py`

## Important Implementation Notes

### Error Handling Pattern
Always use this consistent error handling pattern in `_process_raw_result()`:
```python
def _process_raw_result(self, cmd_idx, raw_output):
    self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_output, self._args.log_raw_data)
    try:
        # ... parsing logic ...
        if not parsed_any:
            raise ValueError('No valid result rows parsed')
    except BaseException as e:
        self._handle_parsing_error(str(e), raw_output)
        return False
    return True
```
Key points:
- Use `BaseException` (not `Exception`) to match codebase convention
- Use `ValueError` for parsing failures (not `RuntimeError`)
- Always call `_handle_parsing_error()` from base class - don't implement custom error handling

### GPU ID Handling
**Do NOT track GPU IDs in result metric names.** SuperBench executes benchmarks with `CUDA_VISIBLE_DEVICES` set per GPU, so results are automatically stored in `metric_name:gpu_id` format by the framework. Simply parse results without GPU prefixes.

### Parsing Percentages
For percentage values like "12.34%", use simple string stripping:
```python
float(percent_str.rstrip('%'))
```
Do NOT use `parse_time_to_us()` for percentages - it only handles time values.

### Avoid Debug Logging
Do not add `logger.debug()` calls in `_process_raw_result()`. The raw output is already stored via `add_raw_data()` for debugging purposes.

### Documentation (`docs/user-tutorial/benchmarks/micro-benchmarks.md`)
Add a section under "## Computation Benchmarks" with:
1. `### \`nvbench-<name>\`` - Benchmark name header
2. `#### Introduction` - Brief description of what the benchmark measures
3. `#### Metrics` - Table with columns: Name, Unit, Description

Example format (see `nvbench-sleep-kernel` or `nvbench-kernel-launch` sections):
```markdown
### `nvbench-<name>`

#### Introduction
Description of what the benchmark measures and any configuration options.

#### Metrics
| Name                              | Unit      | Description                    |
|-----------------------------------|-----------|--------------------------------|
| nvbench-<name>/${param}_cpu_time  | time (μs) | CPU-measured execution time.   |
| nvbench-<name>/${param}_gpu_time  | time (μs) | GPU-measured execution time.   |
```

## NVBench Reference

For advanced NVBench features (axes, types, throughput calculations):
- Source: `third_party/nvbench/`
- Examples: `third_party/nvbench/examples/`

## Checklist

- [ ] `<name>.cu` - CUDA benchmark with `NVBENCH_BENCH` macro
- [ ] `CMakeLists.txt` - Add to `NVBENCH_SOURCES`
- [ ] `nvbench_<name>.py` - Python wrapper extending `NvbenchBase`
- [ ] `__init__.py` - Add import
- [ ] `test_nvbench_<name>.py` - Test file (use `self.assertAlmostEqual` for floats)
- [ ] `nvbench_<name>.log` - Test data
- [ ] `micro-benchmarks.md` - Add Introduction and Metrics documentation
- [ ] `nvbench_<name>.py` - Example script (follow format of other examples)
