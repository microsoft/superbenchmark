# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the megatron deepspeed GPT pretrain class."""

import json
import os
import statistics
import numpy as np
import requests
import torch
from pathlib import Path
import re

from superbench.benchmarks import BenchmarkRegistry
from superbench.benchmarks.context import Platform, Precision
from superbench.benchmarks.model_benchmarks.model_base import ModelBenchmark
from superbench.benchmarks.return_code import ReturnCode
from superbench.common.utils import logger, run_command


def download_file(url, path):
    """Download file from url to path."""
    response = requests.get(url)
    with open(path, 'wb') as file:
        file.write(response.content)


class MegatronGPT(ModelBenchmark):
    """The Megatron DeepSpeed GPT pretrain benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): parameters of the benchmark.
        """
        super().__init__(name, parameters)
        self._supported_precision = [Precision.FLOAT32, Precision.FLOAT16, Precision.BFLOAT16]

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument('--code_base', type=str, required=False, default='', help='Code base.')
        self._parser.add_argument('--dataset_url', type=str, required=False, default=None, help='Dataset URL.')
        self._parser.add_argument(
            '--vocab_url',
            type=str,
            required=False,
            default='https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json',
            help='Vocab URL.'
        )
        self._parser.add_argument(
            '--merges_url',
            type=str,
            required=False,
            default='https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt',
            help='Merges URL.'
        )
        self._parser.add_argument(
            '--tokenizer_type', type=str, required=False, default='GPT2BPETokenizer', help='Tokenizer type.'
        )
        self._parser.add_argument('--model_size', type=int, required=False, default=6.7, help='Model size.')
        self._parser.add_argument('--num_layers', type=int, required=False, default=32, help='Number of layers.')
        self._parser.add_argument('--hidden_size', type=int, required=False, default=4096, help='Hidden size.')
        self._parser.add_argument(
            '--num_attn_heads', type=int, required=False, default=32, help='Number of attention heads.'
        )
        self._parser.add_argument('--micro_batch_size', type=int, required=False, default=2, help='micro batch size.')
        self._parser.add_argument('--lr', type=float, required=False, default=1.2e-4, help='Learning rate.')
        self._parser.add_argument('--min_lr', type=float, required=False, default=1.0e-6, help='Minimum learning rate.')
        self._parser.add_argument('--init_std', type=float, required=False, default=0.009, help='Init std.')
        self._parser.add_argument('--seq_len', type=int, required=False, default=2048, help='Sequence length.')
        self._parser.add_argument(
            '--tensor_model_parallel_size', type=int, required=False, default=1, help='Tensor model parallel size.'
        )
        self._parser.add_argument(
            '--pipeline_model_parallel_size', type=int, required=False, default=1, help='Pipeline model parallel size.'
        )
        self._parser.add_argument(
            '--num_gpus', type=int, required=False, default=8, help='Number of GPUs per node to run the benchmark.'
        )
        self._parser.add_argument(
            '--num_nodes', type=int, required=False, default=1, help='Number of nodes to run the benchmark.'
        )
        self._parser.add_argument('--sequence_parallel', action='store_true', help='Enable Sequence parallel.')
        self._parser.add_argument(
            '--no_async_tensor_model_parallel_allreduce',
            action='store_true',
            help='No async tensor model parallel allreduce.'
        )
        self._parser.add_argument(
            '--use_rotary_position_embeddings', action='store_true', help='Use rotary position embeddings.'
        )
        self._parser.add_argument(
            '--no_gradient_accumulation_fusion', action='store_true', help='No gradient accumulation fusion.'
        )
        self._parser.add_argument('--use_flash_attn', action='store_true', help='Use flash attention.')
        self._parser.add_argument('--no_masked_softmax_fusion', action='store_true', help='No masked softmax fusion.')
        self._parser.add_argument('--no_bias_gelu_fusion', action='store_true', help='No bias gelu fusion.')
        self._parser.add_argument('--no_bias_dropout_fusion', action='store_true', help='No bias dropout fusion.')
        self._parser.add_argument(
            '--train_tokens', type=int, required=False, default=300000000000, help='Train tokens.'
        )
        # lr configs
        # Parallelism configs
        self._parser.add_argument('--zero_stage', type=int, default=1, help='Zero stage.')
        # Misc configs
        self._parser.add_argument('--log-interval', type=int, required=False, default=1, help='Log interval.')
        self._parser.add_argument('--eval_iters', type=int, default=0, help='Eval iters.')
        self._parser.add_argument('--eval_interval', type=int, default=10, help='Eval interval.')
        self._parser.add_argument('--num_save', type=int, default=10000, help='Num save.')
        self._parser.add_argument('--save_interval', type=int, default=10000, help='Save interval.')
        # Output and data configs
        self._parser.add_argument('--seed', type=int, default=1234, help='Seed.')
        self._parser.add_argument('--data_home', type=str, default='/tmp', help='Data home.')
        self._parser.add_argument('--vocab_path', type=str, default='/tmp/gpt2-vocab.json', help='Vocab path.')
        self._parser.add_argument('--merge_path', type=str, default='/tmp/gpt2-merges.txt', help='Merge path.')
        self._parser.add_argument(
            '--split', type=str, default='949,50,1', help='Split dataset ratio for train/val/test.'
        )
        self._parser.add_argument('--prescale_grad', action='store_true', help='Prescale grad.')
        self._parser.add_argument(
            '--hostfile', type=str, default=None, help='Hostfile to run the mutli-node benchmark.'
        )
        self._parser.add_argument('--data_impl', type=str, default='mmap', help='Data impl.')
        self._parser.add_argument('--data_prefix', type=str, default='dataset_text_document', help='Data prefix.')
        self._parser.add_argument('--deepspeed', action='store_true', help='Use deepspeed.')
        self._parser.add_argument('--extra', type=str, default=None, help='Extra options for Megatron.')

    def _preprocess(self):
        if not super()._preprocess():
            return False
        if not self._args.code_base:
            if self._args.deepspeed:
                self._args.code_base = os.path.join(
                    os.getenv('SB_MICRO_PATH'), 'third_party/Megatron/Megatron-DeepSpeed/'
                )
            else:
                self._args.code_base = os.path.join(os.getenv('SB_MICRO_PATH'), 'third_party/Megatron/Megatron-LM')

        if not os.path.exists(self._args.code_base) or \
                not os.path.exists(os.path.join(self._args.code_base, 'pretrain_gpt.py')):
            logger.error('Code base is not valid.')
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            return False

        data_parallel_size = self._args.num_gpus * self._num_nodes \
            // self._args.pipeline_model_parallel_size // self._args.tensor_model_parallel_size
        if self._args.micro_batch_size < 1 or \
                self._args.micro_batch_size > (self._args.batch_size // data_parallel_size):
            logger.error('Micro Batch size * data parallel size is larger than global batch size.')
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            return False

        for precision in self._args.precision:
            if precision not in self._supported_precision:
                logger.error('Precision %s is not supported.' % precision)
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                return False

        if not os.path.exists(self._args.data_home):
            os.makedirs(self._args.data_home)

        return True

    def _parse_log(self, output):
        """Parse log output and get the performance."""
        tflops_pattern = re.compile(r'(TFLOPs|TFLOP/s/GPU\)): (\d+\.\d+)')
        elapsed_time_pattern = re.compile(r'elapsed time per iteration \(ms\): (\d+\.\d+)')
        mem_allocated_pattern = re.compile(r'allocated: (\d+\.\d+)')
        max_mem_allocated_pattern = re.compile(r'max allocated: (\d+\.\d+)')
        lines = output.splitlines()
        tflops = []
        mem_allocated = []
        max_mem_allocated = []
        iteration_times = []
        for line in lines:
            if 'elapsed time per iteration' in line:
                tflops_matches = tflops_pattern.search(line)
                elapsed_time_match = elapsed_time_pattern.search(line)
                if tflops_matches:
                    tflops_values = float(tflops_matches.group(2))
                    tflops.append(tflops_values)
                if elapsed_time_match:
                    elapsed_time_value = float(elapsed_time_match.group(1))
                    iteration_times.append(elapsed_time_value)

            if 'max allocated' in line:
                mem_allocated_match = mem_allocated_pattern.search(line)
                max_mem_allocated_match = max_mem_allocated_pattern.search(line)
                if mem_allocated_match:
                    mem_allocated_value = float(mem_allocated_match.group(1)) / 1024
                    mem_allocated.append(mem_allocated_value)

                if max_mem_allocated_match:
                    max_mem_allocated_value = float(max_mem_allocated_match.group(1)) / 1024
                    max_mem_allocated.append(max_mem_allocated_value)

        return iteration_times, tflops, mem_allocated, max_mem_allocated

    def __prepare_deespeed_config(self, precision_megatron):
        """Prepare deepspeed configs."""
        self._config_json_path = os.path.join(self._args.data_home, 'ds_config_gpt.json')
        # Load deepspeed config template json file
        precision_template = {
            'enabled': True,
            'loss_scale': 0,
            'loss_scale_window': 500,
            'hysteresis': 2,
            'min_loss_scale': 1,
            'initial_scale_power': 11
        }

        ds_config_template = {
            'train_batch_size': self._args.batch_size,
            'train_micro_batch_size_per_gpu': self._args.micro_batch_size,
            'steps_per_print': self._args.log_interval,
            'zero_optimization': {
                'stage': self._args.zero_stage
            },
            'gradient_clipping': 1.0,
            'prescale_gradients': self._args.prescale_grad,
        }

        if len(precision_megatron) > 0:
            ds_config_template[precision_megatron] = precision_template

        # Write to config json file
        with open(self._config_json_path, 'w') as file:
            json.dump(ds_config_template, file, indent=4)

        deepspeed_options = f'\
            --deepspeed \
            --deepspeed_config {self._config_json_path} \
            --zero-stage {self._args.zero_stage} \
            --pipeline-model-parallel-size {self._args.pipeline_model_parallel_size} \
            --train-tokens {self._args.train_tokens} \
            --data-impl {self._args.data_impl}'

        if self._args.pipeline_model_parallel_size <= 1:
            deepspeed_options = f'{deepspeed_options} --no-pipeline-parallel'
        return deepspeed_options

    def _megatron_command(self, precision):    # noqa: C901
        """Generate megatron command."""
        if precision == Precision.FLOAT32:
            precision_megatron = ''
        elif precision == Precision.FLOAT16:
            precision_megatron = '--fp16'
        elif precision == Precision.BFLOAT16:
            precision_megatron = '--bf16'

        megatron_options = f'\
            --override-opt_param-scheduler \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --tensor-model-parallel-size {self._args.tensor_model_parallel_size} \
            --init-method-std {self._args.init_std} \
            --lr-decay-samples 43945312 \
            --lr-warmup-samples {self._args.num_warmup * self._args.batch_size} \
            --lr-decay-style cosine \
            --micro-batch-size {self._args.micro_batch_size} \
            --global-batch-size {self._args.batch_size} \
            --num-layers {self._args.num_layers} \
            --hidden-size {self._args.hidden_size} \
            --num-attention-heads {self._args.num_attn_heads} \
            --seq-length {self._args.seq_len} \
            --max-position-embeddings {self._args.seq_len} \
            --train-samples {self._args.num_steps * self._args.batch_size} \
            --lr {self._args.lr} \
            --min-lr {self._args.min_lr} \
            --split {self._args.split} \
            --log-interval {self._args.log_interval} \
            --eval-interval {self._args.eval_interval} \
            --eval-iters {self._args.eval_iters} \
            --save-interval {self._args.save_interval} \
            --weight-decay 0.1 \
            --clip-grad 1.0 \
            --hysteresis 2 \
            --num-workers {self._args.num_workers} \
            --attention-dropout 0.0 \
            --hidden-dropout 0.0 \
            --optimizer adam \
            --use-distributed-optimizer \
            {precision_megatron} \
            --seed {self._args.seed} \
            --log-throughput'

        if self._args.sequence_parallel:
            megatron_options = f'{megatron_options} --sequence-parallel'
        if self._args.no_async_tensor_model_parallel_allreduce:
            megatron_options = f'{megatron_options} --no-async-tensor-model-parallel-allreduce'
        if self._args.use_rotary_position_embeddings:
            megatron_options = f'{megatron_options} --use-rotary-position-embeddings'
        if self._args.no_gradient_accumulation_fusion:
            megatron_options = f'{megatron_options} --no-gradient-accumulation-fusion'
        if self._args.use_flash_attn:
            megatron_options = f'{megatron_options} --use-flash-attn'
        if self._args.no_masked_softmax_fusion:
            megatron_options = f'{megatron_options} --no-masked-softmax-fusion'
        if self._args.no_bias_gelu_fusion:
            megatron_options = f'{megatron_options} --no-bias-gelu-fusion'
        if self._args.no_bias_dropout_fusion:
            megatron_options = f'{megatron_options} --no-bias-dropout-fusion'
        if self._args.extra:
            megatron_options = f'{megatron_options} {self._args.extra}'

        command = ''
        script_path = os.path.join(self._args.code_base, 'pretrain_gpt.py')
        if self._args.deepspeed:
            deepspeed_option = self.__prepare_deespeed_config(precision_megatron.lstrip('--'))
            # No --log-throughput in Megatron-DeepSpeed by 20231219
            megatron_options = megatron_options.replace('--log-throughput', '').strip()
            if self._num_nodes > 1:
                command = f'torchrun {self._distributed_args} ' + \
                    f'{script_path} {megatron_options} {self._data_options} {deepspeed_option}'
            else:
                command = f'deepspeed {script_path} {megatron_options} {self._data_options} {deepspeed_option}'

        else:
            command = f'torchrun {self._distributed_args} {script_path} {megatron_options} {self._data_options}'

        return command

    def _train_step(self, precision):    # noqa: E501
        """Train the model and get the performance."""
        command = self._megatron_command(precision)
        local_rank = os.environ.pop('OMPI_COMM_WORLD_LOCAL_RANK', None)
        logger.info('Running command: {}.'.format(command))
        output = run_command(command, flush_output=True)
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = local_rank

        iteration_times = []
        info = {}
        # last rank will print the result, first rank will print the memory usage
        if self._num_nodes == 1 or \
            int(os.environ['OMPI_COMM_WORLD_RANK']) == int(os.environ['OMPI_COMM_WORLD_SIZE']) - 1 \
                or int(os.environ['OMPI_COMM_WORLD_RANK']) == 0:
            iteration_times, tflops, mem_allocated, max_mem_allocated = self._parse_log(output.stdout)
            if len(tflops) > 0:
                info['tflops'] = tflops
            if len(mem_allocated) > 0:
                info['mem_allocated'] = mem_allocated
            if len(max_mem_allocated) > 0:
                info['max_mem_allocated'] = max_mem_allocated
        if not iteration_times:
            iteration_times = [-1 for i in range(self._args.num_steps)]

        return iteration_times, info

    def _sync_result(self, data):
        """Sync the result of model benchmarking.

        Args:
            data (list): the data to be reduced.
        """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        data = np.array(data, dtype=np.float64)
        # Reduce the data to a single value on rank 0
        result = np.zeros_like(data)
        comm.Allreduce([data, MPI.DOUBLE], [result, MPI.DOUBLE], op=MPI.MAX)
        return result.tolist()

    def _process_info(self, model_action, precision, info):
        """Process the result of model benchmarking."""
        precision_metric = {'float16': 'fp16', 'float32': 'fp32', 'bfloat16': 'bf16'}
        if precision.value in precision_metric.keys():
            precision = precision_metric[precision.value]
        for key, values in info.items():
            metric = '{}_{}_{}'.format(precision, model_action, key)
            self._result.add_raw_data(metric, values, self._args.log_raw_data)
            self._result.add_result(metric, statistics.mean(values))
            logger.info(
                'Average {} - round: {}, model: {}, precision: {}, value: {:.6f}.'.format(
                    key, self._curr_run_index, self._name, precision, statistics.mean(values)
                )
            )

    def _judge_gpu_availability(self):
        """Judge GPUs' availability according to arguments and running environment."""
        self._gpu_available = not self._args.no_gpu and torch.cuda.is_available()

    def _init_distributed_setting(self):
        """Initialize the distributed library and bind the worker to GPU.

        Return:
            True if distributed library is initialized successfully.
        """
        if not os.getenv('OMPI_COMM_WORLD_SIZE'):
            logger.error('MPI is not enabled.')

            return False
        self._num_nodes = int(os.getenv('OMPI_COMM_WORLD_SIZE')) // int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))
        master_addr = 'localhost'
        if self._num_nodes > 1:
            if not self._args.hostfile:
                sb_hostfile = os.path.join(os.environ.get('SB_WORKSPACE', '.'), 'hostfile')
                if os.path.exists(sb_hostfile):
                    hosts = open(sb_hostfile).read().split('\n')
                    hosts = [f'{host} slots={self._args.num_gpus}' for host in hosts if host != '']
                    self._args.hostfile = os.path.join(self._args.data_home, 'hostfile')
                    with open(self._args.hostfile, 'w') as file:
                        file.write('\n'.join(hosts))
            if not os.path.exists(self._args.hostfile):
                logger.error('Hostfile not found.')
                return False
            hosts = open(self._args.hostfile, 'r').readlines()
            if self._num_nodes != len(hosts):
                logger.error('MPI init failed since hostfile not match the MPI setting.')
                return False
            master_addr = hosts[0].split()[0]

        addr = os.getenv('MASTER_ADDR', master_addr)
        port = os.getenv('MASTER_PORT', '29500')
        node_rank = int(os.environ['OMPI_COMM_WORLD_RANK']) // int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        self._distributed_args = f'--nproc_per_node {self._args.num_gpus} --nnodes {self._num_nodes} ' + \
            f'--node_rank {node_rank} --master_addr {addr} --master_port {port}'
        return True

    def _generate_dataset(self):
        """Generate dataset for benchmarking.

        Return:
            True if dataset is created successfully.
        """
        self._vocab_path = str(Path(self._args.data_home) / 'gpt2-vocab.json')
        download_file(self._args.vocab_url, self._vocab_path)
        self._merges_path = str(Path(self._args.data_home) / 'gpt2-merges.txt')
        download_file(self._args.merges_url, self._merges_path)

        if not os.path.exists(os.path.join(self._args.data_home, f'{self._args.data_prefix}.bin')) \
                or not os.path.exists(os.path.join(self._args.data_home, f'{self._args.data_prefix}.idx')):
            if self._args.dataset_url:
                self._raw_data_path = str(Path(self._args.data_home) / 'data.json')
                download_file(self._args.dataset_url, self._raw_data_path)
                command = (
                    'python3 '
                    f'{os.path.join(self._args.code_base, "tools/preprocess_data.py")} '
                    f'--input {self._raw_data_path} '
                    f'--tokenizer-type {self._args.tokenizer_type} '
                    f'--output-prefix {os.path.join(self._args.data_home, "dataset")} '
                    f'--workers {str(self._args.num_workers)} '
                    f'--vocab-file {self._vocab_path} '
                    f'--merge-file {self._merges_path}'
                )

                # split documents
                run_command(command, flush_output=True)
                # binarize dataset
                run_command(command, flush_output=True)
                if not os.path.exists(os.path.join(self._args.data_home, f'{self._args.data_prefix}.bin')) \
                        or not os.path.exists(os.path.join(self._args.data_home, f'{self._args.data_prefix}.idx')):
                    logger.error('Dataset failed to generate.')
                    self._result.set_return_code(ReturnCode.DATASET_GENERATION_FAILURE)
                    return False
            else:
                logger.error('No dataset or dataset url provided.')
                self._result.set_return_code(ReturnCode.DATASET_GENERATION_FAILURE)
                return False

        self._data_path = os.path.join(self._args.data_home, f'{self._args.data_prefix}')
        self._data_options = f'\
            --vocab-file {self._vocab_path} \
            --merge-file {self._merges_path} \
            --data-path {self._data_path}'

        logger.info('Dataset preparation successfully.')
        return True

    def _set_force_fp32(self):
        """Set force FP32."""
        pass

    def _init_dataloader(self):
        """Initialize the dataloader.

        Return:
            True if dataloader is created successfully.
        """
        return True

    def _create_optimizer(self):
        """Create the optimzier instance used for training and wrap with distributed library if need.

        Return:
            True if optimizer instance is created successfully.
        """
        return True

    def _create_model(self, precision):
        """Construct the model for benchmarking.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.
        """
        return True

    def _inference_step(self, precision):
        """Define the inference process.

        Args:
            precision (Precision): precision of model and input data,
              such as float32, float16.

        Return:
            The latency list of every inference operation.
        """
        pass

    def _cal_params_count(self):
        """Calculate the parameters scale of the model.

        Return:
            The count of trainable parameters.
        """
        pass


# Register GPT3 benchmark.
BenchmarkRegistry.register_benchmark('megatron-gpt', MegatronGPT, parameters='', platform=Platform.CUDA)
BenchmarkRegistry.register_benchmark('megatron-gpt', MegatronGPT, parameters='', platform=Platform.ROCM)
