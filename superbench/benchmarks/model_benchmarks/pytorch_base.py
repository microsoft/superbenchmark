# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch model-benchmark base class."""

import os
from datetime import timedelta
import time

import torch
import transformers
try:
    import transformer_engine.pytorch as te
except ImportError:
    te = None
from torch.utils.data import DataLoader
from torch.distributed import TCPStore, PrefixStore

from superbench.common.utils import logger
from superbench.benchmarks import Framework, ReturnCode, DistributedBackend, DistributedImpl
from superbench.benchmarks.model_benchmarks.model_base import Optimizer, ModelBenchmark
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig
from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import HuggingFaceModelLoader


class PytorchBase(ModelBenchmark):
    """The base class of Pytorch model benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._framework = Framework.PYTORCH
        torch.backends.cudnn.benchmark = True

    def _judge_gpu_availability(self):
        """Judge GPUs' availability according to arguments and running environment."""
        self._gpu_available = not self._args.no_gpu and torch.cuda.is_available()

    def _create_model_source_config(self, precision=None):
        """Create ModelSourceConfig from benchmark arguments.

        Args:
            precision: Optional precision override for torch_dtype.

        Returns:
            ModelSourceConfig if model_source is specified, None otherwise.
        """
        if not hasattr(self._args, 'model_source'):
            return None

        # Determine torch_dtype from precision if not explicitly set
        torch_dtype = 'float32'
        if precision is not None:
            if precision.value == 'float16':
                torch_dtype = 'float16'
            elif precision.value == 'bfloat16':
                torch_dtype = 'bfloat16'

        config = ModelSourceConfig(
            source=self._args.model_source,
            identifier=self._args.model_identifier or self._name,
            torch_dtype=torch_dtype,
            device_map='auto' if not self._gpu_available else None,
        )

        return config

    def _estimate_param_count_from_config(self, hf_config):
        """Estimate parameter count from a HuggingFace config without instantiating the model.

        Delegates to HuggingFaceModelLoader.estimate_param_count_from_config().

        Args:
            hf_config: A HuggingFace PretrainedConfig object.

        Returns:
            Optional[int]: Estimated number of parameters, or None if estimation is not possible.
        """
        return HuggingFaceModelLoader.estimate_param_count_from_config(hf_config)

    def _estimate_training_memory(self, param_count, precision):
        """Estimate GPU memory required for training a model.

        Delegates to HuggingFaceModelLoader.estimate_memory() with mode='training'.

        Args:
            param_count (int): Number of model parameters.
            precision (Precision): Model precision (float32, float16, etc.).

        Returns:
            tuple: (estimated_bytes, gpu_total_bytes, fits) where fits is True if
                   the model is estimated to fit in available memory.
        """
        return HuggingFaceModelLoader.estimate_memory(param_count, precision.value, mode='training')

    def _customize_hf_config(self, hf_config):
        """Hook for subclasses to customize the HF config after download.

        Override this in subclasses to modify config before memory estimation
        and model loading (e.g., to override num_hidden_layers).

        Args:
            hf_config: The downloaded HuggingFace model configuration.

        Returns:
            The (possibly modified) config.
        """
        return hf_config

    def _create_huggingface_model(self, model_config, precision):
        """Load model from HuggingFace Hub and prepare for benchmarking.

        First estimates memory from the HF config (small download). If the model
        fits on the GPU, downloads pretrained weights. Otherwise, uses random
        weight initialization and logs a warning.

        Args:
            model_config (ModelSourceConfig): Configuration for model loading.
            precision (Precision): Model precision (float32, float16, etc.).

        Returns:
            bool: True if model is created successfully, False otherwise.
        """
        try:
            logger.info(f'Loading HuggingFace model: {model_config.identifier}')

            # Step 1: Download config only (few KB) to estimate memory
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            load_kwargs = {}
            if hf_token:
                load_kwargs['token'] = hf_token
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(
                model_config.identifier, trust_remote_code=True, **load_kwargs
            )

            # Allow subclasses to customize config (e.g., override num_hidden_layers)
            hf_config = self._customize_hf_config(hf_config)

            # Step 2: Estimate param count from config (no model instantiation — avoids
            # allocating hundreds of GB of CPU RAM for large models like 70B)
            param_count_raw = self._estimate_param_count_from_config(hf_config)
            if param_count_raw is None:
                logger.warning(
                    f'Could not estimate param count from config for {model_config.identifier}. '
                    f'Proceeding with download — memory check skipped.'
                )
                fits = True
                estimated_bytes, gpu_mem = 0, 0
                param_count = 0
            else:
                estimated_bytes, gpu_mem, fits = self._estimate_training_memory(param_count_raw, precision)
                param_count = param_count_raw / 1e6

            # Step 3: If model doesn't fit, fail gracefully without downloading weights
            if not fits:
                mem_type = 'GPU memory' if self._gpu_available else 'system RAM'
                logger.error(
                    f'Model {model_config.identifier} ({param_count:.1f}M params) estimated to need '
                    f'~{estimated_bytes / 1e9:.1f}GB for training (weights + gradients + optimizer states), '
                    f'which exceeds available {mem_type} ({gpu_mem / 1e9:.1f}GB). '
                    f'Skipping benchmark. To fix this, either: '
                    f'(1) reduce num_hidden_layers to use a smaller model, '
                    f'(2) use a smaller model variant, or '
                    f'(3) use a {"GPU" if self._gpu_available else "machine"} with more memory.'
                )
                return False

            logger.info(
                f'Model {model_config.identifier} ({param_count:.1f}M params) estimated to need '
                f'~{estimated_bytes / 1e9:.1f}GB for training'
                f'{f", fits in available memory ({gpu_mem / 1e9:.1f}GB)" if gpu_mem > 0 else ""}. '
                f'Downloading pretrained weights...'
            )
            loader = HuggingFaceModelLoader(
                cache_dir=None,
                token=hf_token
            )
            hf_model, _, tokenizer = loader.load_model_from_config(
                model_config, device='cpu', config_pretrained=hf_config
            )
            self._tokenizer = tokenizer

            # Step 4: Wrap model for benchmark
            if hasattr(self, '_create_model_wrapper'):
                self._model = self._create_model_wrapper(hf_model, hf_config)
            else:
                self._model = hf_model
                logger.warning(
                    f'No model wrapper defined for {self._name}. Using raw HuggingFace model. '
                    'Consider implementing _create_model_wrapper() for custom head.'
                )

            param_count = sum(p.numel() for p in self._model.parameters()) / 1e6
            logger.info(
                f'Created HuggingFace model - identifier: {model_config.identifier}, '
                f'precision: {precision.value}, parameters: {param_count:.2f}M'
            )

            # Set precision
            self._model = self._model.to(dtype=getattr(torch, precision.value))

            # Ensure model is in training mode (from_pretrained loads in eval mode)
            self._model.train()

            # Move to GPU if available
            if self._gpu_available:
                self._model = self._model.cuda()

            # Create target tensor for training
            if hasattr(self._args, 'num_classes'):
                self._target = torch.LongTensor(self._args.batch_size).random_(self._args.num_classes)
                if self._gpu_available:
                    self._target = self._target.cuda()

            return True

        except Exception as e:
            logger.error(
                f'Failed to load HuggingFace model: {str(e)}'
            )
            import traceback
            logger.error(traceback.format_exc())
            return False

    def add_parser_arguments(self):
        """Add PyTorch model benchmark-specific arguments to the argument parser."""
        super().add_parser_arguments()

        # HuggingFace model loading parameters
        self._parser.add_argument(
            '--model_source',
            type=str,
            default='in-house',
            choices=['in-house', 'huggingface'],
            help='Source of the model: in-house (default) or huggingface.',
        )
        self._parser.add_argument(
            '--model_identifier',
            type=str,
            default=None,
            help='Model identifier: for HuggingFace, use format "org/model-name" (e.g., "bert-base-uncased").',
        )

    def _set_force_fp32(self):
        """Set the config that controls whether full float32 precision will be used.

        On Ampere or newer GPUs, pytorch and tensorflow will use TF32 instead of FP32 by default.
        We can disable TF32 execution by setting force_fp32 as True.
        """
        torch.backends.cuda.matmul.allow_tf32 = not self._args.force_fp32
        torch.backends.cudnn.allow_tf32 = not self._args.force_fp32

    @torch.no_grad()
    def _to_te_model(self, model):
        """Convert the input model to Transformer Engine model.

        Replace all Linear/LayerNorm layers.
        Modified based on Huggingface's utils `accelerate.accelerator.convert_model`, reference:
        https://github.com/huggingface/accelerate/blob/v0.17.1/src/accelerate/utils/transformer_engine.py#L24

        Args:
            model (torch.nn.Module): Torch model.
        """
        if not te:
            return
        for name, m in model.named_children():
            if isinstance(m, torch.nn.Linear):
                # check 16-byte alignment
                if any(p % 16 != 0 for p in m.weight.shape):
                    return
                te_m = te.Linear(m.in_features, m.out_features, bias=(m.bias is not None), params_dtype=m.weight.dtype)
                te_m.weight.copy_(m.weight)
                if m.bias is not None:
                    te_m.bias.copy_(m.bias)
                setattr(model, name, te_m)
            elif isinstance(m, torch.nn.LayerNorm):
                te_m = te.LayerNorm(m.normalized_shape[0], eps=m.eps, params_dtype=m.weight.dtype)
                if hasattr(te_m, 'weight'):
                    te_m.weight.copy_(m.weight)
                    te_m.bias.copy_(m.bias)
                else:
                    te_m.layer_norm_weight.copy_(m.weight)
                    te_m.layer_norm_bias.copy_(m.bias)
                setattr(model, name, te_m)
            else:
                self._to_te_model(m)

    def _init_distributed_setting(self):
        """Initialize the distributed library and bind the worker to GPU.

        Return:
            True if distributed library is initialized successfully.
        """
        if self._args.distributed_impl:
            logger.info(
                'Distributed training is enabled - model: {}, distributed implementation: {}.'.format(
                    self._name, self._args.distributed_impl
                )
            )
            if self._args.distributed_impl == DistributedImpl.HOROVOD:
                import horovod.torch as hvd
                hvd.init()
                self._world_size = int(hvd.size())
                self._local_rank = int(hvd.local_rank())
                self._global_rank = int(hvd.rank())
            elif self._args.distributed_impl == DistributedImpl.DDP:
                if os.environ.get('WORLD_SIZE') is None or os.environ.get('LOCAL_RANK') is None:
                    logger.error(
                        'Can not find WORLD_SIZE or LOCAL_RANK in env variables - model: {},'
                        ' distributed implementation: {}.'.format(self._name, self._args.distributed_impl)
                    )
                    return False
                # torch >= 1.9.0a0 torch.distributed.elastic is used by default
                port = int(os.environ.get('MASTER_PORT', '29500')) + 1
                os.environ['MASTER_PORT'] = str(port)
                addr = os.environ['MASTER_ADDR']
                self._global_rank = int(os.environ['RANK'])
                self._local_rank = int(os.environ['LOCAL_RANK'])
                self._world_size = int(os.environ['WORLD_SIZE'])
                logger.debug('ip:{},port:{},rank:{},world:{}'.format(addr, port, self._global_rank, self._world_size))
                store = PrefixStore(
                    self._name, TCPStore(addr, port, self._world_size, self._global_rank == 0, timedelta(seconds=300))
                )
                torch.distributed.init_process_group(
                    backend=self._args.distributed_backend.value,
                    timeout=timedelta(seconds=300),
                    rank=self._global_rank,
                    world_size=self._world_size,
                    store=store
                )

            else:
                logger.error(
                    'Unsupported distributed implementation - model: {}, distributed implementation: {}.'.format(
                        self._name, self._args.distributed_impl
                    )
                )
                return False

            if self._gpu_available:
                torch.cuda.set_device(self._local_rank)

        return True

    def _init_dataloader(self):
        """Initialize the dataloader.

        Return:
            True if dataloader is created successfully.
        """
        train_sampler = None
        if self._args.distributed_impl:
            if self._args.distributed_impl == DistributedImpl.HOROVOD:
                import horovod.torch as hvd
                train_sampler = \
                    torch.utils.data.distributed.DistributedSampler(
                        self._dataset,
                        num_replicas=hvd.size(),
                        rank=hvd.rank()
                    )
            elif self._args.distributed_impl == DistributedImpl.DDP:
                try:
                    train_sampler = \
                        torch.utils.data.distributed.DistributedSampler(
                            self._dataset
                        )
                except BaseException as e:
                    logger.error(
                        'Init dataloader failed - model: {}, distributed implementation: {}, message: {}.'.format(
                            self._name, self._args.distributed_impl, str(e)
                        )
                    )
                    return False
            else:
                logger.error(
                    'Unsupported distributed implementation - model: {}, distributed implementation: {}.'.format(
                        self._name, self._args.distributed_impl
                    )
                )
                return False

        self._dataloader = DataLoader(
            dataset=self._dataset,
            batch_size=self._args.batch_size,
            shuffle=False,
            num_workers=self._args.num_workers,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=self._args.pin_memory
        )

        return True

    def _create_optimizer(self):
        """Create the optimzier instance used for training and wrap with distributed library if need.

        Return:
            True if optimizer instance is created successfully.
        """
        if self._args.distributed_impl == DistributedImpl.DDP:
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._model, device_ids=[self._local_rank], output_device=self._local_rank
            )

        if self._optimizer_type == Optimizer.SGD:
            self._optimizer = torch.optim.SGD(
                self._model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4, nesterov=True
            )
        elif self._optimizer_type == Optimizer.ADAM:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
        elif self._optimizer_type == Optimizer.ADAMW:
            if hasattr(torch.optim, 'AdamW'):
                self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
            else:
                self._optimizer = transformers.AdamW(self._model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
        else:
            self._optimizer = None

        if not self._optimizer:
            logger.error(
                'Create optimizer failed - model: {}, optimizer type: {}.'.format(self._name, self._optimizer_type)
            )
            return False

        if self._args.distributed_impl == DistributedImpl.HOROVOD:
            import horovod.torch as hvd
            self._optimizer = hvd.DistributedOptimizer(
                self._optimizer,
                named_parameters=self._model.named_parameters(),
                compression=hvd.Compression.none,
                op=hvd.Average
            )
            hvd.broadcast_parameters(self._model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self._optimizer, root_rank=0)

        return True

    def _is_finished(self, curr_step, curr_time, check_frequency=100):
        """Judge whether the benchmarking should be stopped early or not.

        Args:
            curr_step (int): the current benchmarking step.
            curr_time (float): the current time in seconds got from time.time().
            check_frequency (int): the frequency (step numbers) to check if benchmark should be stopped.

        Return:
            True if the benchmarking should be stopped.
        """
        is_finished = int(super()._is_finished(curr_step, curr_time))
        if self._args.duration > 0:
            if curr_step % check_frequency == 0:
                # sync is_finished in distributed mode
                # if any rank is_finished is True, all ranks should be finished
                if self._args.distributed_impl == DistributedImpl.DDP:
                    tensor = torch.IntTensor([is_finished])
                    if self._args.distributed_backend == DistributedBackend.NCCL:
                        tensor = tensor.cuda()
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
                    is_finished = tensor.tolist()[0]
            else:
                is_finished = 0

        return (is_finished == 1)

    def _sync_result(self, result):
        """Function to reduce the result to rank 0.

        Args:
            result (list): The result data to sync.

        Return:
            Result if reduce result data successfully, otherwise None.
        """
        result = super()._sync_result(result)
        if not result:
            return None

        try:
            if self._args.distributed_impl == DistributedImpl.DDP:
                if self._args.distributed_backend == DistributedBackend.NCCL:
                    tensor = torch.as_tensor(result).cuda()
                else:
                    tensor = torch.as_tensor(result)
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
                result = tensor.tolist()
        except BaseException as e:
            logger.error(
                'Sync train result failed - model: {}, distributed implementation: {}, message: {}.'.format(
                    self._name, self._args.distributed_impl, str(e)
                )
            )
            return None

        return result

    def _postprocess(self):
        """Postprocess/cleanup operations after the benchmarking.

        Return:
            True if _postprocess() succeed.
        """
        if not super()._postprocess():
            return False

        try:
            if self._args.distributed_impl == DistributedImpl.DDP:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
        except BaseException as e:
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_DESTROY_FAILURE)
            logger.error(
                'Post process failed - model: {}, distributed implementation: {}, message: {}.'.format(
                    self._name, self._args.distributed_impl, str(e)
                )
            )
            return False

        if self._gpu_available:
            torch.cuda.synchronize()
        del self._target
        del self._optimizer
        del self._model
        if hasattr(self, '_tokenizer'):
            del self._tokenizer
        if self._gpu_available:
            torch.cuda.empty_cache()

        return True

    def _cal_params_count(self):
        """Calculate the parameters scale of the model.

        Return:
            The count of trainable parameters.
        """
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def _timer(self):
        """Returns the current time which ensures all previous CUDA events have been finished.

        If there is no GPU present, this defaults to `time.time()`; otherwise it will
        synchronize CUDA before measuring the time.

        Returns:
            Current time in second.
        """
        if self._gpu_available:
            torch.cuda.synchronize()
        return time.time()

    def _benchmark(self):
        """Wrap super._benchmark with profiler context if enabled by environment variable.

        Set SB_ENABLE_PYTORCH_PROFILER='1' to enable profiling.
        """
        # Check if this is a Nvidia GPU
        if not (torch.cuda.is_available() and torch.version.cuda is not None):
            return super()._benchmark()

        # Check if profiling is enabled via environment variable
        enable_profiler = os.environ.get('SB_ENABLE_PYTORCH_PROFILER', '0') == '1'

        if not enable_profiler:
            # Run without profiling
            return super()._benchmark()

        # Run with profiling enabled
        logger.info('PyTorch profiler enabled for model: {}'.format(self._name))
        ret = None

        from torch.profiler import profile, ProfilerActivity
        from torch.autograd import DeviceType
        import json

        if self._local_rank is None:
            local_rank = 0
        else:
            local_rank = self._local_rank

        diag_agent_prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
        dump_file_dir = os.environ.get('SB_TORCH_PROFILER_TRACE_DIR', '.')
        diag_agent_dump_file_path = f'{dump_file_dir}/torch-profiler-sb-{self._name}-{local_rank}.json'
        diag_agent_prof.__enter__()

        ret = super()._benchmark()

        diag_agent_prof.__exit__(None, None, None)
        diag_agent_events = []
        for event in diag_agent_prof.events():
            if event.device_type != DeviceType.CPU:
                continue
            diag_agent_event = {
                'name': event.name,
                'input_shapes': event.input_shapes,
                'input_values': event.concrete_inputs,
            }
            diag_agent_event['cpu_time'] = event.cpu_time
            diag_agent_event['gpu_time'] = event.cuda_time
            diag_agent_event['start_time'] = event.time_range.start
            diag_agent_events.append(diag_agent_event)
        with open(diag_agent_dump_file_path, 'w') as f:
            json.dump(diag_agent_events, f, sort_keys=True)

        return ret
