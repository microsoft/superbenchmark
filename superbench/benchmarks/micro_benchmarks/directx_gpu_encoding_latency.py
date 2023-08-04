# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the DirectXGPUEncodingLatency benchmarks."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


def create_nv12_file(file_name, num_frames, width, height):
    """Create a NV12 file with the specified name, number of frames, width, and height."""
    import numpy as np
    # Generate a Y plane of width x height with values from 0-255
    y_plane = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    # Generate a UV plane of width x height/2 with values from 0-255
    uv_plane = np.random.randint(0, 256, (height // 2, width), dtype=np.uint8)
    # Create the file
    with open(f'{file_name}', 'wb') as f:
        for _ in range(num_frames):
            # Write the Y plane and UV plane to the file
            f.write(y_plane.tobytes())
            f.write(uv_plane.tobytes())


class DirectXGPUEncodingLatency(MicroBenchmarkWithInvoke):
    """The DirectXGPUEncodingLatency benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor."""
        super().__init__(name, parameters)
        self._bin_name = 'EncoderLatency.exe'
        self._test_file = 'test_directx_gpu_encoding_latency.nv12'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--algo',
            type=str,
            choices=['ASAP', 'OneInOne'],
            default='ASAP',
            required=False,
            help='The algorithm to use for encoding'
        )
        self._parser.add_argument(
            '--codec',
            type=str,
            choices=['AVC', 'H264', 'HEVC', 'H265', 'AV1'],
            default='H265',
            required=False,
            help='The codec to use for encoding'
        )
        self._parser.add_argument(
            '--format',
            type=str,
            choices=['RGBA_F16', 'R10G10B10A2', 'NV12', 'P010'],
            default='NV12',
            required=False,
            help='The format to use for encoding'
        )
        self._parser.add_argument(
            '--frames', type=int, default=500, required=False, help='The number of frames to encode'
        )
        self._parser.add_argument(
            '--height', type=int, default=720, required=False, help='The height of the input video'
        )
        self._parser.add_argument(
            '--width', type=int, default=1080, required=False, help='The width of the input video'
        )
        self._parser.add_argument('--input_file', type=str, default=None, required=False, help='The input video file')
        self._parser.add_argument('--output_file', type=str, default=None, required=False, help='The output video file')
        self._parser.add_argument(
            '--output_height', type=int, default=720, required=False, help='The height of the output video'
        )
        self._parser.add_argument(
            '--output_width', type=int, default=1080, required=False, help='The width of the output video'
        )
        self._parser.add_argument(
            '--vcn', type=int, choices=[0, 1], default=0, required=False, help='The VCN instance to use for encoding'
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        command = os.path.join(self._args.bin_dir, self._bin_name)
        command += f' -ALGORITHM {self._args.algo}'
        command += f' -CODEC {self._args.codec}'
        command += f' -FORMAT {self._args.format}'
        command += f' -FRAMES {self._args.frames}'
        command += f' -HEIGHT {self._args.height}'
        command += f' -WIDTH {self._args.width}'
        if self._args.input_file is not None:
            command += f' -INPUT {os.path.abspath(self._args.input_file)}'
        else:
            if not os.path.exists(f'{self._test_file}'):
                create_nv12_file(self._test_file, self._args.frames, self._args.width, self._args.height)
            command += f' -INPUT {os.path.abspath(self._test_file)}'
        if self._args.output_file is not None:
            command += f' -OUTPUT {self._args.output_file}'
        command += f' -OUTPUT_HEIGHT {self._args.output_height}'
        command += f' -OUTPUT_WIDTH {self._args.output_width}'
        command += f' -VCNINSTANCE {self._args.vcn}'
        self._commands.append(command)

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
        self._result.add_raw_data('raw_output', raw_output, self._args.log_raw_data)

        content = raw_output.splitlines()
        metrics = {}

        try:
            for line in content:
                if 'Total' in line:
                    metrics['fps'] = float(line.split('=')[3].strip().strip('frames').split()[0])
                if 'Latency' in line and 'min' in line.lower():
                    metrics['min_lat'] = float(line.split('=')[1].split(',')[1].strip('ms').strip())
                    metrics['max_lat'] = float(line.split('=')[1].split(',')[2].strip('ms').strip())
                if 'Latency' in line and 'average' in line.lower():
                    metrics['avg_lat'] = float(line.split('=')[1].strip('ms').strip())
        except Exception as e:
            logger.error(
                'The result format is invalid - benchmark: {}, raw output: {}, error: {}'.format(
                    self._name, raw_output, str(e)
                )
            )
            return False

        for metric, value in metrics.items():
            self._result.add_result(metric, value)

        return True


BenchmarkRegistry.register_benchmark(
    'directx-gpu-encoding-latency', DirectXGPUEncodingLatency, platform=Platform.DIRECTX
)
