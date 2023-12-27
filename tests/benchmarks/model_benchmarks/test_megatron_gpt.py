# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BERT model benchmarks."""

import os
from pathlib import Path
import statistics
from unittest import mock
import unittest
from superbench.benchmarks.context import ModelAction, Precision

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from tests.helper.testcase import BenchmarkTestCase


class MegatronGPTTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for IBBenchmark benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'megatron-gpt'
        cls.createMockEnvs(cls)
        cls.hostfile_path = os.path.join(cls._tmp_dir, 'hostfile')

    @classmethod
    def tearDownClass(cls):
        """Hook method for deconstructing the class fixture after running all tests in the class."""
        for p in [
            Path(cls._tmp_dir) / 'pretrain_gpt.py',
            Path(cls._tmp_dir) / 'customdataset_text_document.bin',
            Path(cls._tmp_dir) / 'customdataset_text_document.idx',
            Path(cls._tmp_dir) / 'hostfile'
        ]:
            if p.is_file():
                p.unlink()
        super().tearDownClass()

    @mock.patch('superbench.benchmarks.model_benchmarks.MegatronGPT._generate_dataset')
    def test_megatron_gpt_preprocess(self, mock_generate_dataset):
        """Test megatron-gpt benchmark."""
        # Check registry.
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)
        assert (benchmark_cls)
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--hostfile {self.hostfile_path} --batch_size 2048',
        )

        # Check init distribued setting.
        os.environ['OMPI_COMM_WORLD_SIZE'] = '2'
        os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'] = '1'
        os.environ['OMPI_COMM_WORLD_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        with open(self.hostfile_path, 'w') as f:
            f.write('host1\n')
            f.write('host2\n')
            f.write('host3\n')
        mock_generate_dataset.return_value = True
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.DISTRIBUTED_SETTING_INIT_FAILURE)

        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters='--hostfile xxx --batch_size 2048',
        )
        mock_generate_dataset.return_value = True
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.DISTRIBUTED_SETTING_INIT_FAILURE)

        os.environ['OMPI_COMM_WORLD_SIZE'] = '3'
        os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'] = '1'
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--hostfile {self.hostfile_path} --batch_size 2048',
        )
        mock_generate_dataset.return_value = True
        benchmark._preprocess()
        self.assertEqual(benchmark._num_nodes, 3)
        self.assertEqual(
            benchmark._distributed_args,
            '--nproc_per_node {0} --nnodes {1} --node_rank {2} --master_addr {3} --master_port {4}'.format(
                benchmark._args.num_gpus, benchmark._num_nodes, 0, 'localhost', '12345'
            )
        )

        # Check preprocessing.
        # Negative cases
        # no code_base
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --hostfile {self.hostfile_path} --batch_size 2048',
        )
        mock_generate_dataset.return_value = True
        ret = benchmark._preprocess()
        assert (ret is False)
        self.createMockFiles(['pretrain_gpt.py'])
        # invalid micro batch size
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --hostfile {self.hostfile_path} --micro_batch_size -1',
        )
        mock_generate_dataset.return_value = True
        ret = benchmark._preprocess()
        assert (ret is False)
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --hostfile {self.hostfile_path} --micro_batch_size 4096',
        )
        mock_generate_dataset.return_value = True
        ret = benchmark._preprocess()
        assert (ret is False)
        # invalid precision
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --hostfile {self.hostfile_path} \
                --batch_size 2048 --precision int8',
        )
        mock_generate_dataset.return_value = True
        ret = benchmark._preprocess()
        assert (ret is False)
        # Positive cases
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --hostfile {self.hostfile_path} --batch_size 2048',
        )
        mock_generate_dataset.return_value = True
        ret = benchmark._preprocess()
        assert (ret is True)

    def test_megatron_gpt_dataset(self):
        """Test dataset genreation."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)
        assert (benchmark_cls)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '1'
        os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'] = '1'
        os.environ['OMPI_COMM_WORLD_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        # use existing dataset
        self.createMockFiles(['customdataset_text_document.bin', 'customdataset_text_document.idx'])
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base /root/Megatron-DeepSpeed --data_home {self._tmp_dir} \
                --batch_size 2048 --data_prefix customdataset_text_document',
        )
        ret = benchmark._preprocess()
        ret = benchmark._generate_dataset()
        assert (ret is True)

    @mock.patch('superbench.benchmarks.model_benchmarks.MegatronGPT._generate_dataset')
    def test_megatron_gpt_command(self, mock_generate_dataset):
        """Test command generation."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)
        assert (benchmark_cls)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '2'
        os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'] = '1'
        os.environ['OMPI_COMM_WORLD_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        with open(self.hostfile_path, 'w') as f:
            f.write('host1\n')
            f.write('host2\n')
        # use url to process dataset
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --hostfile {self.hostfile_path} \
                --num_warmup 0 --num_steps 10 --batch_size 2048 --data_prefix dataset_text_document',
        )
        mock_generate_dataset.return_value = True
        benchmark._preprocess()
        benchmark._data_options = f'\
            --vocab-file {self._tmp_dir}/gpt2-vocab.json \
            --merge-file {self._tmp_dir}/gpt2-merges.txt \
            --data-path {self._tmp_dir}/dataset_text_document'

        script_path = str(Path(self._tmp_dir) / 'pretrain_gpt.py')
        expected_command = 'torchrun {distributed_args} {script_path} \
            --override-opt_param-scheduler \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --tensor-model-parallel-size 1 \
            --init-method-std 0.009 \
            --lr-decay-samples 43945312 \
            --lr-warmup-samples 0 \
            --lr-decay-style cosine \
            --micro-batch-size 2 \
            --global-batch-size 2048 \
            --num-layers 32 \
            --hidden-size 4096 \
            --num-attention-heads 32 \
            --seq-length 2048 \
            --max-position-embeddings 2048 \
            --train-samples 20480 \
            --lr 0.00012 \
            --min-lr 1e-06 \
            --split 949,50,1 \
            --log-interval 1 \
            --eval-interval 10 \
            --eval-iters 0 \
            --save-interval 10000 \
            --weight-decay 0.1 \
            --clip-grad 1.0 \
            --hysteresis 2 \
            --num-workers 8 \
            --attention-dropout 0.0 \
            --hidden-dropout 0.0 \
            --optimizer adam \
            --use-distributed-optimizer \
            {precision} \
            --seed 1234 \
            --log-throughput {data_options}'

        precision = Precision.FLOAT32
        command = benchmark._megatron_command(precision)
        self.assertEqual(
            command,
            expected_command.format(
                precision='',
                data_options=benchmark._data_options,
                distributed_args=benchmark._distributed_args,
                script_path=script_path
            )
        )
        precision = Precision.FLOAT16
        command = benchmark._megatron_command(precision)
        self.assertEqual(
            command,
            expected_command.format(
                precision='--fp16',
                data_options=benchmark._data_options,
                distributed_args=benchmark._distributed_args,
                script_path=script_path
            )
        )
        precision = Precision.BFLOAT16
        command = benchmark._megatron_command(precision)
        self.assertEqual(
            command,
            expected_command.format(
                precision='--bf16',
                data_options=benchmark._data_options,
                distributed_args=benchmark._distributed_args,
                script_path=script_path
            )
        )

        os.environ['OMPI_COMM_WORLD_SIZE'] = '1'
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --hostfile {self.hostfile_path} \
                --num_warmup 0 --num_steps 10 --batch_size 2048 --data_prefix dataset_text_document --deepspeed',
        )
        mock_generate_dataset.return_value = True
        benchmark._preprocess()
        benchmark._data_options = f'\
            --vocab-file {self._tmp_dir}/gpt2-vocab.json \
            --merge-file {self._tmp_dir}/gpt2-merges.txt \
            --data-path {self._tmp_dir}/dataset_text_document'

        command = benchmark._megatron_command(Precision.BFLOAT16)
        expected_command = 'deepspeed {script_path} --override-opt_param-scheduler \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --tensor-model-parallel-size 1 \
            --init-method-std 0.009 \
            --lr-decay-samples 43945312 \
            --lr-warmup-samples 0 \
            --lr-decay-style cosine \
            --micro-batch-size 2 \
            --global-batch-size 2048 \
            --num-layers 32 \
            --hidden-size 4096 \
            --num-attention-heads 32 \
            --seq-length 2048 \
            --max-position-embeddings 2048 \
            --train-samples 20480 \
            --lr 0.00012 \
            --min-lr 1e-06 \
            --split 949,50,1 \
            --log-interval 1 \
            --eval-interval 10 \
            --eval-iters 0 \
            --save-interval 10000 \
            --weight-decay 0.1 \
            --clip-grad 1.0 \
            --hysteresis 2 \
            --num-workers 8 \
            --attention-dropout 0.0 \
            --hidden-dropout 0.0 \
            --optimizer adam \
            --use-distributed-optimizer \
            {precision} \
            --seed 1234 {data_options} {deepseed_options}'

        expect_ds_options = f'\
            --deepspeed \
            --deepspeed_config {benchmark._config_json_path} \
            --zero-stage 1 \
            --pipeline-model-parallel-size 1 \
            --train-tokens 300000000000 \
            --data-impl mmap --no-pipeline-parallel'

        self.assertEqual(
            command,
            expected_command.format(
                precision='--bf16',
                data_options=benchmark._data_options,
                script_path=script_path,
                deepseed_options=expect_ds_options
            )
        )

    @decorator.load_data('tests/data/megatron_deepspeed.log')
    @mock.patch('superbench.benchmarks.model_benchmarks.MegatronGPT._generate_dataset')
    def test_megatron_parse_log(self, raw_output, mock_generate_dataset):
        """Test parse log function."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)
        assert (benchmark_cls)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '1'
        os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'] = '1'
        os.environ['OMPI_COMM_WORLD_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        # use url to process dataset
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --num_warmup 0 --num_steps 10 --batch_size 2048',
        )
        mock_generate_dataset.return_value = True
        benchmark._preprocess()
        benchmark._data_options = f'\
            --vocab-file {self._tmp_dir}/gpt2-vocab.json \
            --merge-file {self._tmp_dir}/gpt2-merges.txt \
            --data-path {self._tmp_dir}/dataset_text_document \
            --data-impl mmap'

        iteration_times, tflops, mem_allocated, max_mem_allocated = benchmark._parse_log(raw_output)
        assert (statistics.mean(iteration_times) == 75239.24)
        assert (statistics.mean(tflops) == 149.136)
        assert (statistics.mean(mem_allocated) == 17.535637855529785)
        assert (statistics.mean(max_mem_allocated) == 66.9744234085083)

        info = {'tflops': tflops, 'mem_allocated': mem_allocated, 'max_mem_allocated': max_mem_allocated}
        benchmark._process_info(ModelAction.TRAIN, Precision.FLOAT16, info)
        assert (benchmark.result is not None)
        assert (benchmark.result['fp16_train_tflops'][0] == 149.136)
        assert (benchmark.result['fp16_train_mem_allocated'][0] == 17.535637855529785)
        assert (benchmark.result['fp16_train_max_mem_allocated'][0] == 66.9744234085083)
