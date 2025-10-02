# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch model-benchmark base class."""

import os
from datetime import timedelta
import random
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
from superbench.benchmarks import (
    Framework,
    ReturnCode,
    DistributedBackend,
    DistributedImpl,
)
from superbench.benchmarks.model_benchmarks.model_base import Optimizer, ModelBenchmark
from torch.backends.cuda import sdp_kernel
from superbench.common import model_log_utils


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

        self._generate_log = False
        self._compare_log = None
        self._model_run_metadata = {}
        self._model_run_losses = []
        self._model_run_periodic = {}

    def _judge_gpu_availability(self):
        """Judge GPUs' availability according to arguments and running environment."""
        self._gpu_available = not self._args.no_gpu and torch.cuda.is_available()

    def _enable_deterministic_training(self):
        """Enable deterministic training settings for reproducible results."""
        if hasattr(self._args, 'deterministic_seed'):
            torch.manual_seed(self._args.deterministic_seed)
            random.seed(self._args.deterministic_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self._args.deterministic_seed)
                torch.cuda.manual_seed_all(self._args.deterministic_seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Disable TF32 to remove potential numerical variability
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            logger.info('Failed to disable TF32 in cuda matmul')
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            logger.info('Failed to disable TF32 in cuDNN')
            pass
        # Force Scaled Dot-Product Attention to use deterministic math kernel
        try:
            sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        except Exception:
            logger.info('SDP kernel not available')
            # Older PyTorch versions may not expose sdp_kernel; ignore in that case
            pass

    def _assign_model_run_metadata(self, precision, extra_keys=None):
        """Assign model_run_metadata for determinism fingerprinting/logging.

        Args:
            precision: Model precision (can be enum or string).
            extra_keys: List of additional argument keys to include in metadata.
            self._args: Benchmark arguments containing model configuration.

        Returns:
            None
        """
        # Common metadata keys
        metadata = {
            'model_name': self._name,
            'precision': (precision.value if hasattr(precision, 'value') else str(precision)),
            'seed': getattr(self._args, 'deterministic_seed', None),
            'deterministic_seed': getattr(self._args, 'deterministic_seed', None),
            'batch_size': getattr(self._args, 'batch_size', None),
            'seq_len': getattr(self._args, 'seq_len', None),
            'num_steps': getattr(self._args, 'num_steps', None),
            'check_frequency': getattr(self._args, 'check_frequency', None),
            'num_classes': getattr(self._args, 'num_classes', None),
        }
        # Add any extra keys present in args (for model-specific fields)
        keys = [
            'hidden_size',
            'num_hidden_layers',
            'num_attention_heads',
            'intermediate_size',
            'input_size',
            'num_layers',
            'bidirectional',
        ]
        if extra_keys:
            keys += extra_keys
        for key in keys:
            metadata[key] = getattr(self._args, key, None)
        self._model_run_metadata = metadata
        return None

    def record_determinism_fingerprint(self, curr_step, loss, logits, periodic, check_frequency):
        """Centralized logic for recording per-step loss and periodic fingerprints for deterministic runs.

        Args:
            curr_step (int): Current training step.
            loss (torch.Tensor or float): Loss value for this step.
            logits (torch.Tensor or float): Logits output for this step (sample 0).
            periodic (dict): Dictionary to store periodic fingerprints ('loss', 'act_mean', 'step').
            check_frequency (int): Frequency for fingerprint logging.
        """
        # Record per-step loss for determinism checks (for full history)
        try:
            v = float(loss.detach().item()) if hasattr(loss, 'detach') else float(loss)
        except Exception:
            v = None
        # Periodic fingerprint logging
        if getattr(self._args, 'deterministic', False) and (curr_step % check_frequency == 0):
            # 1) Loss fingerprint (only at fingerprinting frequency)
            try:
                # Ensure the lists exist and remain index-aligned by appending
                # a placeholder (None) when a measurement is unavailable.
                if 'loss' in periodic and isinstance(periodic['loss'], list):
                    periodic['loss'].append(v if v is not None else None)
                else:
                    periodic['loss'] = [v if v is not None else None]

                logger.info(f'Loss at step {curr_step}: {v}')
                periodic.setdefault('step', []).append(curr_step)
            except Exception:
                pass
            # 2) Tiny activation fingerprint: mean over logits for sample 0
            try:
                if logits is not None:
                    act_mean = (
                        float(logits[0].detach().float().mean().item())
                        if hasattr(logits[0], 'detach') else float(logits[0])
                    )
                    logger.info(f'ActMean at step {curr_step}: {act_mean}')
                    periodic.setdefault('act_mean', []).append(act_mean)
                else:
                    # Keep lists aligned by appending None when activation not available
                    periodic.setdefault('act_mean', []).append(None)
            except Exception:
                # On exception preserve alignment by ensuring keys exist
                periodic.setdefault('act_mean', []).append(None)
                pass

    def _finalize_periodic_logging(self, duration, periodic, info_key='loss'):
        """Finalize periodic logging and return results tuple for training step."""
        info = {info_key: periodic.get(info_key, [])}
        self._model_run_losses = list(periodic.get(info_key, []))
        self._model_run_periodic = dict(periodic)
        return (duration, info)

    def _benchmark(self):
        """Run the benchmark then handle post-run model log save/compare."""
        ok = super()._benchmark()
        self._post_run_model_log()
        return ok

    def add_parser_arguments(self):
        """Add PyTorch model benchmark-specific arguments to the argument parser."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--generate-log',
            nargs='?',
            const=True,
            default=False,
            type=str,
            help='Save fingerprint log to file. Optionally specify a path to save the log.'
        )
        self._parser.add_argument(
            '--compare-log',
            '--compare_log',
            dest='compare_log',
            type=str,
            default=None,
            help='Compare this run to a reference fingerprint log.',
        )
        self._parser.add_argument(
            '--deterministic_seed',
            type=int,
            default=42,
            required=False,
            help='Random seed for deterministic training.',
        )
        self._parser.add_argument(
            '--deterministic',
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

    def _post_run_model_log(self):
        """Save or compare model run logs after run, if requested."""
        gen_arg = getattr(self._args, 'generate_log', None)
        if gen_arg:
            # gen_arg can be True (const) or a string path if user provided it
            log_path = None
            if isinstance(gen_arg, str):
                log_path = gen_arg
            if not log_path:
                model = getattr(
                    self._args,
                    'model_name',
                    self._name if hasattr(self, '_name') else 'model',
                )
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                os.makedirs('./outputs', exist_ok=True)
                log_path = f'./outputs/model_run_{model}_{timestamp}.json'
            else:
                # Ensure destination directory exists when a custom path is provided
                try:
                    dirpath = os.path.dirname(log_path) or '.'
                    os.makedirs(dirpath, exist_ok=True)
                except Exception:
                    logger.info(f'Failed to create directory for log path: {log_path}')
                    pass
            model_log_utils.save_model_log(
                log_path,
                self._model_run_metadata,
                self._model_run_losses,
                self._model_run_periodic,
            )
            logger.info(f'Saved model log to {log_path}')
        if getattr(self._args, 'compare_log', None):
            logger.info(f'Comparing model log to {self._args.compare_log}')
            ref = model_log_utils.load_model_log(self._args.compare_log)
            curr = {
                'metadata': self._model_run_metadata,
                'per_step_fp32_loss': self._model_run_losses,
                'fingerprints': self._model_run_periodic,
            }
            compare_ok = model_log_utils.compare_model_logs(curr, ref)
            if not compare_ok:
                raise RuntimeError(
                    f'Determinism check failed: this run does not match reference log {self._args.compare_log}'
                )
            logger.info(f'Determinism check PASSED against {self._args.compare_log}')

    def _preprocess(self):
        """Preprocess and apply PyTorch-specific defaults."""
        preprocess_ok = super()._preprocess()
        if not preprocess_ok:
            return False
        # Enable deterministic training centrally so individual model files don't need to call it.
        if getattr(self._args, 'deterministic', False):
            try:
                self._enable_deterministic_training()
            except Exception:
                logger.info('Failed to enable deterministic training in centralized preprocess')
        if getattr(self._args, 'deterministic', False):
            self._handle_deterministic_log_options()
        return True

    def _handle_deterministic_log_options(self):
        """Set generate_log if deterministic and no log options are set."""
        has_gen = getattr(self._args, 'generate_log', None)
        has_cmp = getattr(self._args, 'compare_log', None)
        if not has_gen and not has_cmp:
            setattr(self._args, 'generate_log', True)
            logger.info('Deterministic run detected with no log options; defaulting to --generate-log.')

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
