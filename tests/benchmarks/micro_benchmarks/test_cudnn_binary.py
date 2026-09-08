"""Run native cuDNN execution checks against SB_CUDNN_TEST_BINARY."""

import json
import math
import os
import subprocess
import unittest


@unittest.skipUnless(
    os.environ.get('SB_CUDNN_TEST_BINARY'), 'Set SB_CUDNN_TEST_BINARY to the compiled native benchmark.'
)
class CudnnBinaryTests(unittest.TestCase):
    """Exercise the actual native binary and its controlled error exits."""

    def configuration(self, name='cudnnConvolutionForward', input_type=0, conv_type=0):
        """Create a bounded convolution configuration."""
        return {
            'name': name,
            'algo': 0,
            'inputDims': [1, 2, 3, 3],
            'inputStride': [18, 9, 3, 1],
            'outputDims': [1, 2, 3, 3],
            'outputStride': [18, 9, 3, 1],
            'filterDims': [2, 2, 1, 1],
            'inputType': input_type,
            'convType': conv_type,
            'arrayLength': 2,
            'padA': [0, 0],
            'filterStrideA': [1, 1],
            'dilationA': [1, 1],
            'mode': 1,
            'tensorOp': False,
        }

    def run_binary(self, configuration):
        """Execute a small case with captured output and a finite deadline."""
        value = json.dumps(configuration) if isinstance(configuration, dict) else configuration
        return subprocess.run(
            [os.environ['SB_CUDNN_TEST_BINARY'], '--num_test', '2', '--warm_up', '1', '--num_in_step', '1',
             '--random_seed', '33931', '--config_json', value],
            capture_output=True, text=True, timeout=30
        )

    def test_convolution_storage_compute_combinations(self):
        """Execute supported storage and compute types in all directions."""
        for name in ('cudnnConvolutionForward', 'cudnnConvolutionBackwardData', 'cudnnConvolutionBackwardFilter'):
            for input_type, conv_type in ((0, 0), (2, 0)):
                with self.subTest(name=name, input_type=input_type, conv_type=conv_type):
                    result = self.run_binary(self.configuration(name, input_type, conv_type))
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn('Error', result.stdout + result.stderr)
                    rows = [line for line in result.stdout.splitlines() if line.startswith('[raw_data]:')]
                    self.assertEqual(len(rows), 1, result.stdout)
                    samples = [float(value) for value in rows[0].split(':', 1)[1].split(',') if value.strip()]
                    self.assertEqual(len(samples), 2)
                    self.assertTrue(all(math.isfinite(value) and value > 0 for value in samples))

    def test_half_compute_support_or_controlled_rejection(self):
        """Keep half-compute capability rejection distinct from an abort."""
        for name in ('cudnnConvolutionForward', 'cudnnConvolutionBackwardData', 'cudnnConvolutionBackwardFilter'):
            with self.subTest(name=name):
                result = self.run_binary(self.configuration(name, 2, 2))
                if 'CUDNN_STATUS_NOT_SUPPORTED' in result.stdout:
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn('[raw_data]:', result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('[raw_data]:', result.stdout)
                    self.assertNotIn('Error:', result.stdout)

    def test_invalid_json_is_a_controlled_failure(self):
        """Reject invalid JSON without timing output."""
        result = self.run_binary('{')
        self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Error:', result.stdout)
        self.assertNotIn('[raw_data]:', result.stdout)

    def test_invalid_type_pair_is_a_controlled_failure(self):
        """Reject an unsupported storage and compute type pair."""
        result = self.run_binary(self.configuration(input_type=0, conv_type=2))
        self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Error:', result.stdout)
        self.assertNotIn('[raw_data]:', result.stdout)

    def test_missing_required_field_is_a_controlled_failure(self):
        """Reject an incomplete configuration without a false successful exit."""
        config = self.configuration()
        del config['filterDims']
        result = self.run_binary(config)
        self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('[raw_data]:', result.stdout)

    def test_invalid_function_is_a_controlled_failure(self):
        """Reject unknown function names through the normal error handler."""
        result = self.run_binary(self.configuration(name='not-a-cudnn-function'))
        self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Error:', result.stdout)
        self.assertNotIn('[raw_data]:', result.stdout)

    def test_noncontiguous_tensor_strides_execute_or_reject_cleanly(self):
        """Respect strided storage when supported and retain capability failures."""
        for name in ('cudnnConvolutionForward', 'cudnnConvolutionBackwardData', 'cudnnConvolutionBackwardFilter'):
            with self.subTest(name=name):
                config = self.configuration(name)
                config['inputStride'] = [40, 20, 5, 1]
                config['outputStride'] = [40, 20, 5, 1]
                result = self.run_binary(config)
                if 'CUDNN_STATUS_NOT_SUPPORTED' in result.stdout:
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn('[raw_data]:', result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('[raw_data]:', result.stdout)

    def test_invalid_descriptor_lengths_are_controlled_failures(self):
        """Reject incomplete descriptor arrays before unsafe API access."""
        for field, value in (('inputStride', [18]), ('padA', [0]), ('filterDims', [])):
            with self.subTest(field=field):
                config = self.configuration()
                config[field] = value
                result = self.run_binary(config)
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('Error:', result.stdout)
                self.assertNotIn('[raw_data]:', result.stdout)


if __name__ == '__main__':
    unittest.main()
