# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the megatron deepspeed GPT pretrain class."""

import json
import os
import statistics
import requests
import torch
from pathlib import Path
import re

from superbench.benchmarks import BenchmarkRegistry
from superbench.benchmarks.context import ModelAction, Precision
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
        self._parser.add_argument(
            '--global_batch_size', type=int, required=False, default=2048, help='Global batch size.'
        )
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

        if not os.path.exists(self._args.code_base) or \
                not os.path.exists(os.path.join(self._args.code_base, 'pretrain_gpt.py')):
            logger.error('Code base is not valid.')
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            return False

        data_parallel_size = self._args.num_gpus * self._num_nodes \
            // self._args.pipeline_model_parallel_size // self._args.tensor_model_parallel_size
        if self._args.batch_size < 1 or \
                self._args.batch_size > (self._args.global_batch_size // data_parallel_size):
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

    def _is_rank_0(self):
        """Check if the rank is 0."""
        # If it's invoked by MPI and rank is not 0, empty content is expected
        if os.getenv('OMPI_COMM_WORLD_RANK'):
            rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            if rank == 0:
                return True
        return False

    def _parse_log(self, output):
        """Parse log output and get the performance."""
        tflops_pattern = re.compile(r'TFLOPs: (\d+\.\d+)')
        samples_per_second_pattern = re.compile(r'samples per second: (\d+\.\d+)')
        elapsed_time_pattern = re.compile(r'elapsed time per iteration \(ms\): (\d+\.\d+)')
        mem_allocated_pattern = re.compile(r'MemAllocated=([\d.]+)[KMGTPEZY]?B')
        max_mem_allocated_pattern = re.compile(r'MaxMemAllocated=([\d.]+)[KMGTPEZY]?B')
        lines = output.splitlines()
        tflops = []
        samples_per_seconds = []
        mem_allocated = []
        max_mem_allocated = []
        iteration_times = []
        for line in lines:
            if 'TFLOPs' in line:
                tflops_matches = tflops_pattern.search(line)
                samples_per_second_match = samples_per_second_pattern.search(line)
                elapsed_time_match = elapsed_time_pattern.search(line)
                if tflops_matches:
                    tflops_values = float(tflops_matches.group(1))
                    tflops.append(tflops_values)
                if samples_per_second_match:
                    samples_per_second_value = float(samples_per_second_match.group(1))
                    samples_per_seconds.append(samples_per_second_value)
                if elapsed_time_match:
                    elapsed_time_value = float(elapsed_time_match.group(1))
                    iteration_times.append(elapsed_time_value)

            if 'MaxMemAllocated' in line:
                mem_allocated_match = mem_allocated_pattern.search(line)
                max_mem_allocated_match = max_mem_allocated_pattern.search(line)
                if mem_allocated_match:
                    mem_allocated_value = float(mem_allocated_match.group(1))
                    mem_allocated.append(mem_allocated_value)

                if max_mem_allocated_match:
                    max_mem_allocated_value = float(max_mem_allocated_match.group(1))
                    max_mem_allocated.append(max_mem_allocated_value)

        return iteration_times, samples_per_seconds, tflops, mem_allocated, max_mem_allocated

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
            'train_batch_size': self._args.global_batch_size,
            'train_micro_batch_size_per_gpu': self._args.batch_size,
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
            --pipeline-model-parallel-size {self._args.pipeline_model_parallel_size}'

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
            --lr-warmup-samples {self._args.num_warmup * self._args.global_batch_size} \
            --lr-decay-style cosine \
            --micro-batch-size {self._args.batch_size} \
            --global-batch-size {self._args.global_batch_size} \
            --num-layers {self._args.num_layers} \
            --hidden-size {self._args.hidden_size} \
            --num-attention-heads {self._args.num_attn_heads} \
            --seq-length {self._args.seq_len} \
            --max-position-embeddings {self._args.seq_len} \
            --train-tokens {self._args.train_tokens} \
            --train-samples {self._args.num_steps * self._args.global_batch_size} \
            --lr {self._args.lr} \
            --min-lr {self._args.min_lr} \
            --split 949,50,1 \
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
            --seed {self._args.seed}'

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
            if self._num_nodes > 1:
                command = f'torchrun {self._distributed_args} ' + \
                    f'{script_path} {megatron_options} {self._data_options} {deepspeed_option}'
            else:
                command = f'deepspeed {script_path} {megatron_options} {self._data_options} {deepspeed_option}'

        else:
            command = f'torchrun {self._distributed_args} {script_path} {megatron_options} {self._data_options}'

        return command

    def _train(self, precision):    # noqa: E501
        """Train the model and get the performance."""
        command = self._megatron_command(precision)
        local_rank = os.environ.pop('OMPI_COMM_WORLD_LOCAL_RANK', None)
        logger.info('Running command: {}.'.format(command))
        output = run_command(command, flush_output=True)
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = local_rank

        # last rank will print the result, first rank will print the memory usage
        if self._num_nodes == 1 or \
            int(os.environ['OMPI_COMM_WORLD_RANK']) == int(os.environ['OMPI_COMM_WORLD_SIZE']) - 1 \
                or self._is_rank_0():
            iteration_times, samples_per_seconds, tflops, mem_allocated, max_mem_allocated = self._parse_log(
                output.stdout
            )

            iteration_times, samples_per_seconds, tflops, mem_allocated, max_mem_allocated = self.__process_model_result(
                ModelAction.TRAIN, precision, iteration_times, samples_per_seconds, tflops, mem_allocated,
                max_mem_allocated
            )

            if not iteration_times and not mem_allocated:
                self._result.set_return_code(ReturnCode.INVALID_BENCHMARK_RESULT)
                return False

            if iteration_times:
                logger.info(
                    'Average train time - round: {}, model: {}, precision: {}, \
                    iteration time: {:.6f} ms, samples per second: {:.6f}, tflops: {:.6f}.'.format(
                        self._curr_run_index, self._name, precision, statistics.mean(iteration_times),
                        statistics.mean(samples_per_seconds), statistics.mean(tflops)
                    )
                )
            if mem_allocated:
                logger.info(
                    'Average train time - round: {}, model: {}, precision: {}, \
                    allocated mem: {:.6f}, max allocated mem: {:.6f}.'.format(
                        self._curr_run_index, self._name, precision, statistics.mean(mem_allocated),
                        statistics.mean(max_mem_allocated)
                    )
                )

        return True

    def __process_model_result(
        self, model_action, precision, iteration_times, samples_per_seconds, tflops, mem_allocated, max_mem_allocated
    ):
        """Process the result of model benchmarking."""
        if len(tflops) == 0 and len(mem_allocated) == 0:
            logger.error(
                'Step time list is empty - round: {}, model: {}, model_action: {}, precision: {}.'.format(
                    self._curr_run_index, self._name, model_action, precision
                )
            )
            self._result.set_return_code(ReturnCode.INVALID_BENCHMARK_RESULT)
            return None, None, None, None, None

        precision_metric = {'float16': 'fp16', 'float32': 'fp32', 'bfloat16': 'bf16'}
        if precision.value in precision_metric.keys():
            precision = precision_metric[precision.value]

        metric_d = '{}_{}_iteration_time'.format(precision, model_action)
        metric_s = '{}_{}_samples_per_second'.format(precision, model_action)
        metric_t = '{}_{}_tflops'.format(precision, model_action)
        metric_m = '{}_{}_mem_allocated'.format(precision, model_action)
        metric_m_max = '{}_{}_max_mem_allocated'.format(precision, model_action)
        if iteration_times:
            self._result.add_raw_data(
                metric_d.format(precision, model_action), iteration_times, self._args.log_raw_data
            )
            self._result.add_raw_data(
                metric_s.format(precision, model_action), samples_per_seconds, self._args.log_raw_data
            )
            self._result.add_raw_data(metric_t.format(precision, model_action), tflops, self._args.log_raw_data)
            self._result.add_result(metric_d, statistics.mean(iteration_times))
            self._result.add_result(metric_s, statistics.mean(samples_per_seconds))
            self._result.add_result(metric_t, statistics.mean(tflops))

        if mem_allocated:
            self._result.add_raw_data(metric_m.format(precision, model_action), mem_allocated, self._args.log_raw_data)
            self._result.add_raw_data(
                metric_m_max.format(precision, model_action), max_mem_allocated, self._args.log_raw_data
            )
            self._result.add_result(metric_m, statistics.mean(mem_allocated))
            self._result.add_result(metric_m_max, statistics.mean(max_mem_allocated))

        return iteration_times, samples_per_seconds, tflops, mem_allocated, max_mem_allocated

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

            addr = os.getenv('MASTER_ADDR', hosts[0].split()[0])
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
            --data-path {self._data_path} \
            --data-impl {self._args.data_impl}'

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

    def _train_step(self, precision):
        """Define the training process.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Return:
            The step-time list of every training step.
        """
        pass

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
BenchmarkRegistry.register_benchmark('megatron-gpt', MegatronGPT, parameters='')
