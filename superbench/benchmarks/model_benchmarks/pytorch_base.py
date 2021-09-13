# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch model-benchmark base class."""

import os

import torch
import transformers
from torch.utils.data import DataLoader

from superbench.common.utils import logger
from superbench.benchmarks import Framework, ReturnCode
from superbench.benchmarks.model_benchmarks.model_base import Optimizer, DistributedImpl, ModelBenchmark


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

                torch.distributed.init_process_group(backend=self._args.distributed_backend.value)
                self._world_size = int(os.environ['WORLD_SIZE'])
                self._local_rank = int(os.environ['LOCAL_RANK'])
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
