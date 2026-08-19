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
│   └── nvbench_<name>.json                # Sample NVBench JSON output for tests (CREATE)
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
| Test data | `tests/data/nvbench_kernel_launch.json` | `tests/data/nvbench_sleep_kernel.json` |
| Example | `examples/benchmarks/nvbench_kernel_launch.py` | `examples/benchmarks/nvbench_sleep_kernel.py` |

## Key Components

### Base Class: `nvbench_base.py`
- `NvbenchBase` - Inherit from this class
- `_build_base_command()` - Builds command with common NVBench CLI args
- `_extend_command(parts)` - Override to append benchmark-specific args (e.g. axes); the base attaches `--json` automatically
- `_load_result_json(cmd_idx, raw_output)` - Reads the NVBench `--json` file (falls back to `raw_output` in tests) and returns the parsed dict
- `_iter_states(data)` - Yields `(axis_values, summaries_by_tag)` for each non-skipped state
- `_summary_value(summaries, tag)` - Returns the float value of a summary by tag
- `_handle_parsing_error()` - Consistent error handling (see Error Handling section below)
- `parse_time_to_us(str)` - Legacy string time parser; not needed when parsing JSON

### CMakeLists.txt
Add new `.cu` file to `NVBENCH_SOURCES` list.

### Python Wrapper Pattern
1. Set `self._bin_name = 'nvbench_<name>'` (must match CMake target)
2. Override `add_parser_arguments()` if benchmark has custom parameters
3. Override `_extend_command(parts)` to append axis args (do NOT override `_preprocess`; the base attaches `--json` and finalizes the command)
4. Implement `_process_raw_result()` to parse the NVBench JSON via `_load_result_json()`, `_iter_states()`, and `_summary_value()`

### Registration
- Python: `BenchmarkRegistry.register_benchmark('nvbench-<name>', Nvbench<Name>, platform=Platform.CUDA)`
- Import in `__init__.py`

## Important Implementation Notes

### Error Handling Pattern
Always use this consistent error handling pattern in `_process_raw_result()`:
```python
def _process_raw_result(self, cmd_idx, raw_output):
    try:
        data = self._load_result_json(cmd_idx, raw_output)
        parsed_any = False
        for axes, summaries in self._iter_states(data):
            self._result.add_result('gpu_time', self._summary_value(summaries, 'nv/cold/time/gpu/mean') * 1e6)
            parsed_any = True
        if not parsed_any:
            raise ValueError('No valid result states parsed')
    except BaseException as e:
        self._handle_parsing_error(str(e), raw_output)
        return False
    return True
```
Key points:
- `_load_result_json()` records the raw JSON via `add_raw_data()` - do NOT call `add_raw_data()` yourself
- Use `BaseException` (not `Exception`) to match codebase convention
- Use `ValueError` for parsing failures (not `RuntimeError`)
- Always call `_handle_parsing_error()` from base class - don't implement custom error handling

### GPU ID Handling
**Do NOT track GPU IDs in result metric names.** SuperBench executes benchmarks with `CUDA_VISIBLE_DEVICES` set per GPU, so results are automatically stored in `metric_name:gpu_id` format by the framework. Simply parse results without GPU prefixes.

### Parsing NVBench JSON Output
Benchmarks parse NVBench's machine-readable JSON, not the Markdown table. The base class forces `--json <path>` onto every command (see Result Output below); `_process_raw_result()` should:
1. `data = self._load_result_json(cmd_idx, raw_output)`
2. Iterate states with `for axes, summaries in self._iter_states(data):`
3. Read values by tag with `self._summary_value(summaries, '<tag>')`

Values are typed and unit-normalized in the JSON:
- **Durations are in seconds** (float64) - multiply by `1e6` to store microseconds.
- **CUPTI percentages are fractions** in `[0, 1]` - multiply by `100` to store percent.
- **Axis values** come from `axes` (e.g. `axes['Duration (us)']`, `axes['BlockSize']`), not from parsed columns.

Common tags:

| Metric | Tag | Conversion |
|--------|-----|------------|
| CPU time (mean) | `nv/cold/time/cpu/mean` | `× 1e6` → µs |
| GPU time (mean) | `nv/cold/time/gpu/mean` | `× 1e6` → µs |
| Batch GPU time | `nv/batch/time/gpu/mean` | `× 1e6` → µs |
| Element rate | `nv/cold/bw/item_rate` | elements/s |
| CUPTI counters | `nv/cupti/...` | `× 100` → percent |

### Result Output (JSON)
The base class writes NVBench results to a JSON file and reads them back:
- Default: a unique temp directory (e.g. `/tmp/nvbench_*`), removed after parsing.
- `--output_dir <dir>`: persist the raw JSON there instead (useful for debugging/artifacts).

Each command gets `--json <output_dir>/<bin_name>_<idx>.json` appended automatically, so you never add `--json` yourself.

### Parsing Percentages
CUPTI percentage summaries in the JSON are stored as fractions in `[0, 1]`; multiply by `100` to store percent (e.g. a `StoreEff` value of `1.0` → `100.0`).

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
| nvbench-<name>/${param}_cpu_time  | time (us) | CPU-measured execution time.   |
| nvbench-<name>/${param}_gpu_time  | time (us) | GPU-measured execution time.   |
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
- [ ] `nvbench_<name>.json` - Test data (NVBench JSON)
- [ ] `micro-benchmarks.md` - Add Introduction and Metrics documentation
- [ ] `nvbench_<name>.py` - Example script (follow format of other examples)
