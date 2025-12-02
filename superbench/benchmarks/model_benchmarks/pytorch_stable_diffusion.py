# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch Stable Diffusion model benchmark."""

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Precision
from superbench.benchmarks.model_benchmarks.model_base import Optimizer
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase
from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


class PytorchStableDiffusion(PytorchBase):
    """The Stable Diffusion benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._supported_precision = [Precision.FLOAT32, Precision.FLOAT16, Precision.BFLOAT16]
        self._optimizer_type = Optimizer.ADAMW
        self._pipe = None

    def add_parser_arguments(self):
        """Add the Stable Diffusion-specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--model_id',
            type=str,
            default='stabilityai/stable-diffusion-xl-base-1.0',
            help='Model ID or path for Stable Diffusion model.'
        )
        self._parser.add_argument(
            '--guidance_scale',
            type=float,
            default=8.0,
            help='Guidance scale for diffusion process.'
        )
        self._parser.add_argument(
            '--num_inference_steps',
            type=int,
            default=20,
            help='Number of inference steps.'
        )
        self._parser.add_argument(
            '--height',
            type=int,
            default=1024,
            help='Image height.'
        )
        self._parser.add_argument(
            '--width',
            type=int,
            default=1024,
            help='Image width.'
        )
        self._parser.add_argument(
            '--seq_len',
            type=int,
            default=77,
            help='Text sequence length.'
        )
        self._parser.add_argument(
            '--hidden_size',
            type=int,
            default=2048,
            help='Text encoder hidden size.'
        )
        self._parser.add_argument(
            '--pooled_embeds_dim',
            type=int,
            default=1280,
            help='Pooled text embeddings dimension.'
        )
        self._parser.add_argument(
            '--latent_channels',
            type=int,
            default=4,
            help='Number of latent space channels.'
        )
        self._parser.add_argument(
            '--latent_downsample_factor',
            type=int,
            default=8,
            help='Latent space downsampling factor from image space.'
        )
        self._parser.add_argument(
            '--crop_coords_top_left_h',
            type=int,
            default=0,
            help='Crop coordinates top left height.'
        )
        self._parser.add_argument(
            '--crop_coords_top_left_w',
            type=int,
            default=0,
            help='Crop coordinates top left width.'
        )

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info.

        Return:
            True if dataset is created successfully.
        """
        self._dataset = TorchRandomDataset(
            [
                self._args.sample_count,
                self._args.latent_channels,
                self._args.height // self._args.latent_downsample_factor,
                self._args.width // self._args.latent_downsample_factor
            ],
            self._world_size,
            dtype=torch.float32
        )
        if len(self._dataset) == 0:
            logger.error('Generate random dataset failed - model: {}'.format(self._name))
            return False

        return True

    def _create_model(self, precision):
        """Construct the model for benchmarking.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.
        """
        try:
            # Map precision to torch dtype
            if precision == Precision.FLOAT16:
                dtype = torch.float16
                variant = 'fp16'
            elif precision == Precision.BFLOAT16:
                dtype = torch.bfloat16
                variant = None
            else:
                dtype = torch.float32
                variant = None

            # Load scheduler
            self._scheduler = EulerDiscreteScheduler.from_pretrained(
                self._args.model_id,
                subfolder='scheduler'
            )

            # Load pipeline
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                self._args.model_id,
                scheduler=self._scheduler,
                add_watermarker=False,
                variant=variant,
                torch_dtype=dtype,
            )

            if self._gpu_available:
                self._pipe.to('cuda')

            # Disable progress bar for cleaner output
            self._pipe.set_progress_bar_config(disable=True)

            # Extract UNet model for training
            self._model = self._pipe.unet

        except BaseException as e:
            logger.error(
                'Create model with specified precision failed - model: {}, precision: {}, message: {}.'.format(
                    self._name, precision, str(e)
                )
            )
            return False

        # Create random target latents for training
        self._target = torch.randn(
            self._args.batch_size,
            self._args.latent_channels,
            self._args.height // self._args.latent_downsample_factor,
            self._args.width // self._args.latent_downsample_factor,
            dtype=getattr(torch, precision.value)
        )
        if self._gpu_available:
            self._target = self._target.cuda()

        # Generate random text embeddings to bypass text encoder
        prompt_embeds = torch.randn(
            self._args.batch_size, self._args.seq_len, self._args.hidden_size,
            dtype=getattr(torch, precision.value)
        )
        pooled_prompt_embeds = torch.randn(
            self._args.batch_size, self._args.pooled_embeds_dim,
            dtype=getattr(torch, precision.value)
        )
        if self._gpu_available:
            prompt_embeds = prompt_embeds.cuda()
            pooled_prompt_embeds = pooled_prompt_embeds.cuda()

        self._encoder_hidden_states = prompt_embeds
        self._added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": self._get_add_time_ids()}

        return True

    def _get_add_time_ids(self):
        """Get additional conditioning time IDs for SDXL.

        Return:
            Tensor of time IDs with shape (batch_size, 6).
        """
        add_time_ids = torch.tensor([
            [self._args.height, self._args.width,  # original size
             self._args.crop_coords_top_left_h, self._args.crop_coords_top_left_w,  # crops coords top left
             self._args.height, self._args.width]  # target size
        ], dtype=self._encoder_hidden_states.dtype)
        add_time_ids = add_time_ids.repeat(self._args.batch_size, 1)
        if self._gpu_available:
            add_time_ids = add_time_ids.cuda()
        return add_time_ids

    def _train_step(self, precision):
        """Define the training process.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Return:
            The step-time list of every training step.
        """
        duration = []
        curr_step = 0
        check_frequency = 100
        while True:
            for idx, sample in enumerate(self._dataloader):
                sample = sample.to(dtype=getattr(torch, precision.value))
                start = self._timer()
                if self._gpu_available:
                    sample = sample.cuda()
                if self._args.exclude_copy_time:
                    start = self._timer()
                self._optimizer.zero_grad()
                # Simulate timestep
                timestep = torch.randint(0, self._scheduler.config.num_train_timesteps, (self._args.batch_size,))
                if self._gpu_available:
                    timestep = timestep.cuda()
                output = self._model(
                    sample,
                    timestep,
                    encoder_hidden_states=self._encoder_hidden_states,
                    added_cond_kwargs=self._added_cond_kwargs
                ).sample
                loss = torch.nn.functional.mse_loss(output, self._target)
                loss.backward()
                self._optimizer.step()
                end = self._timer()
                curr_step += 1
                if curr_step > self._args.num_warmup:
                    # Save the step time of every training/inference step, unit is millisecond.
                    duration.append((end - start) * 1000)
                    self._log_step_time(curr_step, precision, duration)
                if self._is_finished(curr_step, end, check_frequency):
                    return duration

    def _inference_step(self, precision):
        """Define the inference process.

        Args:
            precision (Precision): precision of model and input data,
          such as float32, float16.

        Return:
            The latency list of every inference operation.
        """
        duration = []
        curr_step = 0
        with torch.no_grad():
            self._model.eval()
            while True:
                for idx, sample in enumerate(self._dataloader):
                    sample = sample.to(dtype=getattr(torch, precision.value))
                    start = self._timer()
                    if self._gpu_available:
                        sample = sample.cuda()
                    if self._args.exclude_copy_time:
                        start = self._timer()
                    # Simulate timestep
                    timestep = torch.randint(0, self._scheduler.config.num_train_timesteps, (self._args.batch_size,))
                    if self._gpu_available:
                        timestep = timestep.cuda()
                    self._model(
                        sample,
                        timestep,
                        encoder_hidden_states=self._encoder_hidden_states,
                        added_cond_kwargs=self._added_cond_kwargs
                    )
                    end = self._timer()
                    curr_step += 1
                    if curr_step > self._args.num_warmup:
                        # Save the step time of every training/inference step, unit is millisecond.
                        duration.append((end - start) * 1000)
                        self._log_step_time(curr_step, precision, duration)
                    if self._is_finished(curr_step, end):
                        return duration


# Register Stable Diffusion benchmark
BenchmarkRegistry.register_benchmark(
    'pytorch-stable-diffusion-xl',
    PytorchStableDiffusion,
    parameters='--model_id stabilityai/stable-diffusion-xl-base-1.0'
)
