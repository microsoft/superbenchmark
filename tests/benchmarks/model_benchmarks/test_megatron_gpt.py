# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BERT model benchmarks."""

import os
from pathlib import Path
import shlex
import statistics
from unittest import mock
import unittest
from superbench.benchmarks.context import ModelAction, Precision

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from tests.helper.testcase import BenchmarkTestCase


def normalize_command(cmd):
    """Convert a CLI string into a list of meaningful argument units (key-value or flag)."""
    tokens = shlex.split(cmd)
    units = []
    i = 0
    while i < len(tokens):
        if tokens[i].startswith('--'):
            if i + 1 >= len(tokens) or tokens[i + 1].startswith('--'):
                units.append(tokens[i])    # flag-only
                i += 1
            else:
                units.append(f'{tokens[i]} {tokens[i + 1]}')    # key-value pair
                i += 2
        else:
            # Include positional args like torchrun, script path, etc.
            units.append(tokens[i])
            i += 1
    return sorted(units)


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
                --num_warmup 0 --num_steps 10 --batch_size 2048 --data_prefix dataset_text_document --override_opt_param_scheduler',
        )
        mock_generate_dataset.return_value = True
        benchmark._preprocess()
        benchmark._data_options = f'\
            --vocab-file {self._tmp_dir}/gpt2-vocab.json \
            --merge-file {self._tmp_dir}/gpt2-merges.txt \
            --data-path {self._tmp_dir}/dataset_text_document \
            --split 949,50,1'

        script_path = str(Path(self._tmp_dir) / 'pretrain_gpt.py')
        expected_command_template = 'torchrun {distributed_args} {script_path} \
            --tokenizer-type GPT2BPETokenizer \
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
        expected_command = expected_command_template.format(
            precision='',
            data_options=benchmark._data_options,
            distributed_args=benchmark._distributed_args,
            script_path=script_path
        )
        command = benchmark._megatron_command(precision)
        actual_units = normalize_command(command)
        expected_units = normalize_command(expected_command)
        self.assertEqual(actual_units, expected_units)

        precision = Precision.FLOAT16
        expected_command = expected_command_template.format(
            precision='--fp16',
            data_options=benchmark._data_options,
            distributed_args=benchmark._distributed_args,
            script_path=script_path
        )
        command = benchmark._megatron_command(precision)
        actual_units = normalize_command(command)
        expected_units = normalize_command(expected_command)
        self.assertEqual(actual_units, expected_units)

        precision = Precision.BFLOAT16
        expected_command = expected_command_template.format(
            precision='--bf16',
            data_options=benchmark._data_options,
            distributed_args=benchmark._distributed_args,
            script_path=script_path
        )
        command = benchmark._megatron_command(precision)
        actual_units = normalize_command(command)
        expected_units = normalize_command(expected_command)
        self.assertEqual(actual_units, expected_units)

        os.environ['OMPI_COMM_WORLD_SIZE'] = '1'
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --hostfile {self.hostfile_path} \
                --num_warmup 0 --num_steps 10 --batch_size 2048 --data_prefix dataset_text_document --deepspeed --override_opt_param_scheduler',
        )
        benchmark._preprocess()
        benchmark._data_options = f'\
            --vocab-file {self._tmp_dir}/gpt2-vocab.json \
            --merge-file {self._tmp_dir}/gpt2-merges.txt \
            --data-path {self._tmp_dir}/dataset_text_document \
            --split 949,50,1'

        command = benchmark._megatron_command(Precision.BFLOAT16)
        expected_command = 'deepspeed {script_path} --override-opt_param-scheduler \
            --tokenizer-type GPT2BPETokenizer \
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

        expected_command = expected_command.format(
            precision='--bf16',
            data_options=benchmark._data_options,
            deepseed_options=expect_ds_options,
            script_path=script_path
        )
        command = benchmark._megatron_command(Precision.BFLOAT16)
        actual_units = normalize_command(command)
        expected_units = normalize_command(expected_command)
        self.assertEqual(actual_units, expected_units)

    def test_deepseek_v2_command(self):
        # test deepspeed with megatron
        os.environ['OMPI_COMM_WORLD_SIZE'] = '1'
        os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'] = '1'
        os.environ['OMPI_COMM_WORLD_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        with open(self.hostfile_path, 'w') as f:
            f.write('host1\n')

        benchmark_name = 'megatron-deepseek-v2'
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.ROCM)
        assert (benchmark_cls)
        benchmark = benchmark_cls(
            benchmark_name,
            parameters=f'--code_base {self._tmp_dir} --hostfile {self.hostfile_path} '
            '--num_warmup 0 '
            '--num_steps 10 '
            '--batch_size 256 '
            '--expert_model_parallel_size 8 '
            '--micro_batch_size 2 '
            '--mock_data '
            '--model=deepseek '
            '--tokenizer_type=DeepSeekV2Tokenizer '
            '--transformer_impl=transformer_engine '
            '--num_layers=27 '
            '--hidden_size=1024 '
            '--seq_len=4096 '
            '--ffn_hidden_size=10944 '
            '--num_attn_heads=16 '
            '--moe_ffn_hidden_size=1408 '
            '--enable_shared_expert '
            '--moe_layer_freq=1 '
            '--num_shared_experts=2 '
            '--moe_router_topk=6 '
            '--moe_aux_loss_coeff=0.01 '
            '--moe_router_load_balancing_type=aux_loss '
            '--num_experts=64 '
            '--patch_tokenizer_type=DeepSeekV2Tokenizer '
            '--position_embedding_type=rope '
            '--no_rope_fusion '
            '--rotary_base=10000 '
            '--rotary_scaling_factor=40 '
            '--qk_nope_head_dim=128 '
            '--qk_rope_head_dim=64 '
            '--v_head_dim=128 '
            '--ffn_hidden_size=10944 '
            '--swiglu '
            '--normalization=RMSNorm '
            '--norm_epsilon=1e-06 '
            '--no_bias_swiglu_fusion '
            '--disable_bias_linear '
            '--untie_embeddings_and_output_weights '
            '--extra_vocab_size=2400 '
            '--load=deepseek-ai/DeepSeek-V2-Lite '
            '--no_load_optim '
            '--no_load_rng '
            '--ckpt_format=torch '
            '--eod_mask_loss '
            '--train_mode=pretrain '
            '--data_cache_path=/root/cache '
            '--max_padding_length=4096 '
            '--kv_lora_rank=512 '
            '--dataloader_type=cyclic '
        )

        benchmark._preprocess()
        benchmark._data_options = f'\
            --mock-data \
            --dataloader-type cyclic \
            --data-cache-path /root/cache \
            --dataset LLama-Pretrain-Idxmap'

        precision = Precision.BFLOAT16
        command = benchmark._megatron_command(precision)

        expected_command = (
            'torchrun {script_path} --bf16 \
            --init-method-std 0.009 \
            --adam-beta1 0.9 \
            --hidden-dropout 0.0 \
            --min-lr 1e-06 \
            --lr 0.00012 \
            --optimizer adam \
            --log-interval 1 \
            --eval-interval 10 \
            --seed 1234 \
            --eval-iters 0 \
            --max-position-embeddings 4096 \
            --hysteresis 2 \
            --lr-decay-style cosine \
            --lr-decay-samples 43945312 \
            --clip-grad 1.0 \
            --save-interval 10000 \
            --adam-beta2 0.95 \
            --moe-aux-loss-coeff 0.01 \
            --log-throughput \
            --num-workers 8 \
            --use-distributed-optimizer \
            --attention-dropout 0.0 \
            --tensor-model-parallel-size 1 \
            --lr-warmup-samples 0 \
            --weight-decay 0.1 \
            --train-samples 2560 \
            --no-load-optim \
            --load deepseek-ai/DeepSeek-V2-Lite \
            --no-load-rng \
            --ffn-hidden-size 10944 \
            --patch-tokenizer-type DeepSeekV2Tokenizer \
            --swiglu \
            --normalization RMSNorm \
            --norm-epsilon 1e-06 \
            --no-bias-swiglu-fusion \
            --no-rope-fusion \
            --position-embedding-type rope \
            --untie-embeddings-and-output-weights \
            --disable-bias-linear \
            --ckpt-format torch \
            --rotary-base 10000 \
            --rotary-scaling-factor 40 \
            --eod-mask-loss \
            --moe-ffn-hidden-size 1408 \
            --enable-shared-expert \
            --moe-layer-freq 1 \
            --num-shared-experts 2 \
            --moe-router-topk 6 \
            --kv-lora-rank 512 \
            --qk-nope-head-dim 128 \
            --qk-rope-head-dim 64 \
            --v-head-dim 128 \
            --moe-router-load-balancing-type aux_loss \
            --train-mode pretrain \
            --extra-vocab-size 2400 \
            --global-batch-size 256 \
            --micro-batch-size 2 \
            --num-layers 27 \
            --hidden-size 1024 \
            --seq-length 4096 \
            --num-attention-heads 16 \
            --tokenizer-type DeepSeekV2Tokenizer \
            --transformer-impl transformer_engine \
            --num-experts 64 \
            --expert-model-parallel-size 8 \
            --max-padding-length 4096 \
            {data_options} \
            {disitributed_args}'
        ).format(
            script_path=str(Path(self._tmp_dir) / 'pretrain_deepseek.py'),
            data_options=benchmark._data_options,
            disitributed_args=benchmark._distributed_args
        )
        actual_units = normalize_command(command)
        expected_units = normalize_command(expected_command)

        self.assertEqual(actual_units, expected_units)

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
