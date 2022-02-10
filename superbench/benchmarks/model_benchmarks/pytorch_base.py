# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch model-benchmark base class."""

import os
from datetime import timedelta

import torch
import transformers
from torch.utils.data import DataLoader
from torch.distributed import TCPStore, PrefixStore

from superbench.common.utils import logger
from superbench.benchmarks import Framework, ReturnCode, DistributedBackend, DistributedImpl
from superbench.benchmarks.model_benchmarks.model_base import Optimizer, ModelBenchmark


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

    def _set_force_fp32(self):
        """Set the config that controls whether full float32 precision will be used.

        On Ampere or newer GPUs, pytorch and tensorflow will use TF32 instead of FP32 by default.
        We can disable TF32 execution by setting force_fp32 as True.
        """
        torch.backends.cuda.matmul.allow_tf32 = not self._args.force_fp32
        torch.backends.cudnn.allow_tf32 = not self._args.force_fp32

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
            elif self._args.distributed_impl == DistributedImpl.DDP:
                if os.environ.get('WORLD_SIZE') is None or os.environ.get('LOCAL_RANK') is None:
                    logger.error(
                        'Can not find WORLD_SIZE or LOCAL_RANK in env variables - model: {},'
                        ' distributed implementation: {}.'.format(self._name, self._args.distributed_impl)
                    )
                    return False
                # torch >= 1.9.0a0 torch.distributed.elastic is used by default
                port = int(os.environ['MASTER_PORT']) + 1
                addr = os.environ['MASTER_ADDR']
                global_rank = int(os.environ['RANK'])
                self._local_rank = int(os.environ['LOCAL_RANK'])
                self._world_size = int(os.environ['WORLD_SIZE'])
                logger.debug('ip:{},port:{},rank:{},world:{}'.format(addr, port, global_rank, self._world_size))
                store = PrefixStore(
                    self._name, TCPStore(addr, port, self._world_size, global_rank == 0, timedelta(seconds=300))
                )
                torch.distributed.init_process_group(
                    backend=self._args.distributed_backend.value,
                    timeout=timedelta(seconds=300),
                    rank=global_rank,
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
            num_workers=8,
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

    def _sync_result(self, result):
        """Function to reduce the result to rank 0.

        Args:
            result (list): The result data to sync.

        Return:
            True if reduce result data successfully.
        """
        if not super()._sync_result(result):
            return False

        try:
            if self._args.distributed_impl == DistributedImpl.DDP:
                if self._args.distributed_backend == DistributedBackend.NCCL:
                    tensor = torch.as_tensor(result).cuda()
                else:
                    tensor = torch.as_tensor(result)
                torch.distributed.reduce(tensor, 0, op=torch.distributed.ReduceOp.MAX)
                result = tensor.tolist()
        except BaseException as e:
            logger.error(
                'Sync train result failed - model: {}, distributed implementation: {}, message: {}.'.format(
                    self._name, self._args.distributed_impl, str(e)
                )
            )
            return False

        return True

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
