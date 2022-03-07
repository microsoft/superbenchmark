# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU-Burn Test."""


import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke



class GpuBurnBenchmark(MicroBenchmarkWithInvoke):
    """The GPU Burn Test benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.
        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'gpu_burn'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--doubles',
            action='store_true',
            default=False,
            help='Use doubles for the data type used in GPU-Burn',        
                
        )
        self._parser.add_argument(
            '--tensor_core',
            action='store_true',
            default=False,
            help='Use tensor cores in GPU-Burn',

        )
        self._parser.add_argument(
            '--time',
            type=int,
            default=10,
            help='Length of time to run GPU-Burn for(in seconds)',

        )        


    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.
        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False
        
        if not self._set_binary_path():
            return False
        
        command = os.path.join(self._args.bin_dir, self._bin_name)
        
        if self._args.doubles:
            command+=' -d'
        
        if self._args.tensor_core:
            command+=' -tc'
        command+= ' {} '.format(self._args.time)
        #copy compare.ptx which needs to be in the working directory
        compare_copy="cp " + self._args.bin_dir+ "/compare.ptx ./"
        #remove compare.ptx from working directory
        compare_rm="rm " + "compare.ptx"

        self._commands.append(compare_copy + " && " + command + " && " + compare_rm)

        return True

    def _process_raw_result(self, cmd_idx, raw_output):    
        """Function to parse raw results and save the summarized results.
          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.
        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.
        Return:
            True if the raw output string is valid and result can be extracted.
        """

        #self._result.add_raw_data('raw_output_gpuburn', raw_output)
        content = raw_output.splitlines()
        gpu_res=[]
        abort=False
        failure_msg="unknown failure"
        index=-1
        try:
            for idx, line in enumerate(content):
                if 'No clients are alive!' in line or 'Couldn\'t init a GPU' in line or 'Failure during compute' in line or 'Low mem for result' in line:
                    abort=True
                    failure_msg=line
                    break
                if 'done' in line:
                    index=idx
                    break

            if not abort:
                if 'done' not in content[index]:
                    abort=True
                    failure_msg= "The result format invalid"
                    raise failure_msg

                content = content[index + 2:len(content):]
                
                for line in content:
                    if 'Tested' in line:
                        continue;
                    if 'GPU' in line:
                        gpu_res.append(line.strip('\n').strip('\t'))

                self._result.add_result('GPU_Burn_Time',self._args.time)
                for res in gpu_res:
                    if 'OK' in res:
                        self._result.add_result(res.split(':')[0].replace(' ','_') + '_Pass', 1 )
                    else:
                        self._result.add_result(res.split(':')[0].replace(' ','_') + '_Fail', 1 )
                    self._result.add_raw_data('GPU-Burn_result',res)
            else:
                self._result.add_raw_data('GPU Burn Failure: ', failure_msg)
                return False 
        except BaseException as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True
BenchmarkRegistry.register_benchmark('gpu-burn', GpuBurnBenchmark, platform=Platform.CUDA)
