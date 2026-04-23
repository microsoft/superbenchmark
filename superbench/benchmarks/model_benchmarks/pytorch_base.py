# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch model-benchmark base class."""

import os
import statistics
import time
from datetime import timedelta

import torch
import transformers
try:
    import transformer_engine.pytorch as te
except ImportError:
    te = None
from torch.distributed import TCPStore, PrefixStore
from torch.utils.data import DataLoader

from superbench.common.utils import logger
from superbench.common import model_log_utils
from superbench.benchmarks import (
    Framework,
    ReturnCode,
    DistributedBackend,
    DistributedImpl,
)
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
        # Set CUBLAS_WORKSPACE_CONFIG early, before parent init which might parse args
        # This ensures it's set before any CUDA operations if determinism is enabled
        if 'enable_determinism' in parameters:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

        super().__init__(name, parameters)

        self._framework = Framework.PYTORCH
        torch.backends.cudnn.benchmark = True

        self._model_run_losses = []
        self._model_run_periodic = {}

    def _judge_gpu_availability(self):
        """Judge GPUs' availability according to arguments and running environment."""
        self._gpu_available = not self._args.no_gpu and torch.cuda.is_available()

    def _enable_deterministic_training(self):
        """Enable deterministic training settings for reproducible results."""
        # Set CUBLAS_WORKSPACE_CONFIG (should already be set in __init__, but ensure it's set as backup)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

        if hasattr(self._args, 'deterministic_seed'):
            import random
            torch.manual_seed(self._args.deterministic_seed)
            random.seed(self._args.deterministic_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._args.deterministic_seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Disable TF32 to remove potential numerical variability
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            logger.warning('Failed to disable TF32 in cuda matmul')

        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            logger.warning('Failed to disable TF32 in cuDNN')

        # Force Scaled Dot-Product Attention to use deterministic math kernel
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception:
            logger.warning('SDP kernel backend configuration not available')
            # Older PyTorch versions may not expose these APIs; ignore in that case

    def record_determinism_fingerprint(self, curr_step, loss, logits, periodic, check_frequency):
        """Centralized logic for recording per-step loss and periodic fingerprints for deterministic runs.

        Args:
            curr_step (int): Current training step.
            loss (torch.Tensor or float): Loss value for this step.
            logits (torch.Tensor or float): Logits output for this step (sample 0).
            periodic (dict): Dictionary to store periodic fingerprints ('loss', 'act_mean', 'step').
            check_frequency (int): Frequency for fingerprint logging.
        """
        enable_determinism = getattr(self._args, 'enable_determinism', False)
        # If determinism is not enabled, skip determinism-specific logging to avoid unnecessary GPU syncs.
        if not enable_determinism:
            return

        # Record per-step loss for determinism checks
        loss_value = model_log_utils.record_step_loss(loss, curr_step, self._model_run_losses, logger)

        # Record periodic fingerprint (loss and activation mean)
        model_log_utils.record_periodic_fingerprint(
            curr_step,
            loss_value,
            logits,
            periodic,
            check_frequency,
            enable_determinism,
            logger,
        )

    def _finalize_periodic_logging(self, periodic, info_key='loss'):
        """Finalize periodic logging and return info dict for training step."""
        info = {info_key: periodic.get(info_key, [])}
        if self._model_run_periodic and getattr(self._args, 'enable_determinism', False):
            logger.warning(
                'Deterministic periodic data is being overwritten by a subsequent precision/action run. '
                "Only the last run's deterministic metrics will be reported. "
                'Consider using a single precision when enable_determinism is set.'
            )
        self._model_run_periodic = dict(periodic)
        return info

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
            hf_config = AutoConfig.from_pretrained(model_config.identifier, trust_remote_code=True, **load_kwargs)

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
            loader = HuggingFaceModelLoader(cache_dir=None, token=hf_token)
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
            logger.error(f'Failed to load HuggingFace model: {str(e)}')
            import traceback
            logger.error(traceback.format_exc())
            return False

    def add_parser_arguments(self):
        """Add PyTorch model benchmark-specific arguments to the argument parser."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--deterministic_seed',
            type=int,
            default=42,
            required=False,
            help='Random seed for deterministic training.',
        )
        self._parser.add_argument(
            '--enable_determinism',
            action='store_true',
            default=False,
            help='Enable deterministic training for reproducible results.',
        )
        self._parser.add_argument(
            '--check_frequency',
            type=int,
            default=100,
            required=False,
            help='How often (in steps) to run lightweight periodic checks/logs and evaluate early-stop conditions.',
        )
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

    def _post_run_model_log(self):
        """Add deterministic metrics to results.

        Deterministic metrics (loss, activation mean) are stored in the results file alongside
        other benchmark metrics. These can later be compared using `sb result diagnosis`.
        """
        # Add deterministic metrics to result system (all ranks add their own metrics)
        if getattr(self._args, 'enable_determinism', False):
            self._add_deterministic_metrics_to_result()

    def _add_deterministic_metrics_to_result(self):
        """Add deterministic fingerprints and losses to the benchmark result system.

        This makes deterministic metrics visible in results-summary.json alongside
        other benchmark metrics. In distributed training, metrics include rank information.
        """
        # Add periodic fingerprints (loss, activation mean) to results
        if self._model_run_periodic:
            for key, values in self._model_run_periodic.items():
                if isinstance(values, list) and values:
                    # Include rank in metric name for distributed training
                    if self._global_rank is not None:
                        metric_name = f'deterministic_{key}_rank{self._global_rank}'
                    else:
                        metric_name = f'deterministic_{key}'

                    # Add summarized result (mean of checkpointed values)
                    filtered_values = [v for v in values if v is not None]
                    if filtered_values:
                        self._result.add_result(metric_name, statistics.mean(filtered_values))
                    else:
                        # No valid (non-None) values recorded; record NaN to avoid StatisticsError
                        self._result.add_result(metric_name, float('nan'))

        # Add count of deterministic checks performed
        if self._model_run_periodic.get('step'):
            if self._global_rank is not None:
                metric_name = f'deterministic_check_count_rank{self._global_rank}'
            else:
                metric_name = 'deterministic_check_count'
            self._result.add_result(metric_name, len(self._model_run_periodic['step']))

        # Add configuration parameters for validation
        self._add_determinism_config_to_result()

    def _add_determinism_config_to_result(self):
        """Add benchmark configuration parameters as metrics for determinism validation.

        These parameters are included in the results file so they can be compared
        between runs using diagnosis rules. This ensures runs being compared used
        identical configurations.
        """
        # Configuration parameters to include in results for validation
        config_params = {
            'batch_size': getattr(self._args, 'batch_size', None),
            'num_steps': getattr(self._args, 'num_steps', None),
            'num_warmup': getattr(self._args, 'num_warmup', None),
            'deterministic_seed': getattr(self._args, 'deterministic_seed', None),
            'check_frequency': getattr(self._args, 'check_frequency', None),
            'seq_len': getattr(self._args, 'seq_len', None),
            'hidden_size': getattr(self._args, 'hidden_size', None),
            'num_classes': getattr(self._args, 'num_classes', None),
            'input_size': getattr(self._args, 'input_size', None),
            'num_layers': getattr(self._args, 'num_layers', None),
            'num_hidden_layers': getattr(self._args, 'num_hidden_layers', None),
            'num_attention_heads': getattr(self._args, 'num_attention_heads', None),
            'intermediate_size': getattr(self._args, 'intermediate_size', None),
        }

        for param_name, value in config_params.items():
            if value is not None:
                metric_name = f'deterministic_config_{param_name}'
                self._result.add_result(metric_name, value)

    def _create_target(self, num_classes):
        """Create target tensor for training, using a deterministic generator when determinism is enabled.

        Args:
            num_classes (int): Number of classes for random target generation.

        Return:
            torch.LongTensor: Target tensor of shape (batch_size,).
        """
        generator = None
        if getattr(self._args, 'enable_determinism', False) and hasattr(self._args, 'deterministic_seed'):
            generator = torch.Generator()
            generator.manual_seed(self._args.deterministic_seed + 1)
        if generator is not None:
            target = torch.LongTensor(self._args.batch_size).random_(num_classes, generator=generator)
        else:
            target = torch.LongTensor(self._args.batch_size).random_(num_classes)
        if self._gpu_available:
            target = target.cuda()
        return target

    def _preprocess(self):
        """Preprocess and apply PyTorch-specific defaults."""
        preprocess_ok = super()._preprocess()
        if not preprocess_ok:
            return False
        return True

    def set_deterministic_seed(self):
        """Set deterministic RNGs centrally for PyTorch benchmarks.

        This will set the seeds and deterministic flags prior to dataset generation
        so per-model dataset generation is reproducible without each model needing
        to call torch.manual_seed().
        """
        if getattr(self._args, 'enable_determinism', False):
            # Validate check_frequency before any deterministic operations
            check_freq = getattr(self._args, 'check_frequency', 100)
            if not isinstance(check_freq, int) or check_freq <= 0:
                logger.error(
                    f'Invalid check_frequency={check_freq}. Must be a positive integer >= 1. '
                    'Defaulting to 100.'
                )
                self._args.check_frequency = 100
            try:
                self._enable_deterministic_training()
            except Exception:
                logger.error(
                    'Failed to enable deterministic training. '
                    'Disabling enable_determinism to avoid silently non-deterministic results.'
                )
                self._args.enable_determinism = False

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

        Run the benchmark then handle post-run model log save/compare.
        Set SB_ENABLE_PYTORCH_PROFILER='1' to enable profiling.
        """
        # Check if this is a Nvidia GPU
        if not (torch.cuda.is_available() and torch.version.cuda is not None):
            ok = super()._benchmark()
            self._post_run_model_log()
            return ok

        # Check if profiling is enabled via environment variable
        enable_profiler = os.environ.get('SB_ENABLE_PYTORCH_PROFILER', '0') == '1'

        if not enable_profiler:
            # Run without profiling
            ok = super()._benchmark()
            self._post_run_model_log()
            return ok

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

        # Handle post-run model log save/compare regardless of profiling
        self._post_run_model_log()
        return ret
