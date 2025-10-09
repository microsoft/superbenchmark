# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the NVBench Kernel Launch benchmark."""

import re
from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform
from superbench.benchmarks.micro_benchmarks.nvbench_base import NvbenchBase


class NvbenchKernelLaunch(NvbenchBase):
    """The NVBench Kernel Launch benchmark class."""

    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._bin_name = 'nvbench_kernel_launch'

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to parse raw results and save the summarized results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_output, self._args.log_raw_data)
        
        try:
            gpu_section = r"### \[(\d+)\] NVIDIA"
            # Regex pattern to handle different time units and flexible spacing
            row_pat = (
                r"\|\s*([0-9]+)x\s*\|\s*"                            # Samples
                r"([\d.]+\s*[μmun]?s)\s*\|\s*"                       # CPU Time (μs, ns, ms, us, s)
                r"([\d.]+%)\s*\|\s*"                                 # CPU Noise percentage
                r"([\d.]+\s*[μmun]?s)\s*\|\s*"                       # GPU Time
                r"([\d.]+%)\s*\|\s*"                                 # GPU Noise percentage
                r"([0-9]+)x\s*\|\s*"                                 # Batch Samples
                r"([\d.]+\s*[μmun]?s)\s*\|"                          # Batch GPU Time
            )
            current = None
            parsed_any = False  # Track if any valid rows are parsed
            
            for line in raw_output.splitlines():
                line = line.strip()
                g = re.match(gpu_section, line)
                if g:
                    current = f"gpu_{g.group(1)}"
                    continue
                    
                r = re.match(row_pat, line)
                if r and current:
                    samples, cpu_time, cpu_noise, gpu_time, gpu_noise, batch_samples, batch_gpu = r.groups()
                    # self._result.add_result("samples", int(samples.replace('x', '')))
                    self._result.add_result("cpu_time", self._parse_time_value(cpu_time))
                    # self._result.add_result("cpu_noise", self._parse_percentage(cpu_noise))
                    self._result.add_result("gpu_time", self._parse_time_value(gpu_time))
                    # self._result.add_result("gpu_noise", self._parse_percentage(gpu_noise))
                    # self._result.add_result("batch_samples", int(batch_samples.replace('x', '')))
                    self._result.add_result("batch_gpu_time", self._parse_time_value(batch_gpu))
                    parsed_any = True
                    
            if not parsed_any:
                logger.error("No valid rows parsed from the raw output.")
                raise RuntimeError("No valid rows parsed")
                
        except Exception as e:
            self._handle_parsing_error(str(e), raw_output)
            return False
            
        return True

BenchmarkRegistry.register_benchmark('nvbench-kernel-launch', NvbenchKernelLaunch, platform=Platform.CUDA)