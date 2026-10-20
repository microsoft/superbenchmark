# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Full-parameter BF16 V2-Lite training, launched once through torchrun."""

import argparse
import gc
import os
import signal
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

from deepseek_v2_lite_reference import fixed_documents, load_model, training_reference
from deepseek_v2_lite_utils import (
    configure_cache, download_checkpoint, identity_report, interrupt_worker, packed_tokens, read_json,
    set_cache_environment, write_json
)


def compare_gradients(model, reference):
    """Unshard one layer at a time to check sampled dense, MLA and MoE gradients."""
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    groups = {}
    for name, probe in reference['gradients'].items():
        if name.startswith('model.layers.'):
            layer = int(name.split('.')[2])
            unit = model.module.model.layers[layer]
            relative_name = '.'.join(name.split('.')[3:])
            key = layer
        else:
            unit = model
            relative_name = name
            key = -1
        groups.setdefault(key, (unit, []))[1].append((relative_name, probe))
    errors = []
    for unit, probes in groups.values():
        with FSDP.summon_full_params(unit, recurse=False, writeback=False, with_grads=True):
            for name, probe in probes:
                parameter = unit.module.get_parameter(name)
                if list(parameter.shape) != probe['shape'] or parameter.grad is None:
                    raise RuntimeError('Missing or incorrectly shaped full gradient: {}'.format(name))
                indices = torch.tensor(probe['indices'], device='cuda', dtype=torch.long)
                actual = parameter.grad.detach().flatten()[indices].float()
                expected = torch.tensor(probe['values'], device='cuda', dtype=torch.float32)
                torch.testing.assert_close(actual, expected, rtol=0.025, atol=1e-5)
                errors.append((actual - expected).abs().max().item())
    return max(errors)


def training_step(model, optimizer, inputs, verify_update=False):
    """Run real loss, backward, gradient clipping and an AdamW parameter update."""
    import torch
    import torch.distributed as dist

    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().to('cpu', copy=True) for parameter in model.parameters()] if verify_update else []
    torch.cuda.synchronize()
    start = time.monotonic()
    output = model(input_ids=inputs, labels=inputs, use_cache=False)
    loss = output.loss
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError('Missing or nonfinite causal-LM loss.')
    loss.backward()
    gradient_norm = model.clip_grad_norm_(1.0)
    if not torch.isfinite(gradient_norm).item() or gradient_norm.item() <= 0:
        raise RuntimeError('Missing, zero or nonfinite full-model gradient norm.')
    optimizer.step()
    torch.cuda.synchronize()
    elapsed_ms = (time.monotonic() - start) * 1000
    changed = 0
    if verify_update:
        for previous, parameter in zip(before, model.parameters()):
            difference = parameter.detach().cpu() - previous
            if not torch.isfinite(difference).all().item():
                raise RuntimeError('Optimizer produced nonfinite model parameters.')
            changed += torch.count_nonzero(difference).item()
        count = torch.tensor(changed, device='cuda', dtype=torch.int64)
        dist.all_reduce(count)
        changed = count.item()
        if changed <= 0:
            raise RuntimeError('AdamW did not update model parameters.')
    average_loss = loss.detach().float()
    dist.all_reduce(average_loss)
    average_loss /= dist.get_world_size()
    return elapsed_ms, average_loss.item(), gradient_norm.item(), changed


def run_training(args, reference, checkpoint):
    """Train the complete checkpoint with per-layer FSDP and FP32 optimizer state."""
    import torch
    import torch.distributed as dist
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    model = load_model(str(checkpoint), torch.float32)
    decoder_type = type(model.model.layers[0])
    model = FSDP(
        model,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={decoder_type}),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        limit_all_gathers=True,
    )
    model.train()
    correctness_inputs = torch.tensor(reference['tokens'], device='cuda', dtype=torch.long)
    output = model(input_ids=correctness_inputs, labels=correctness_inputs, use_cache=False)
    torch.testing.assert_close(
        output.loss.float(), torch.tensor(reference['loss'], device='cuda'), rtol=0.005, atol=0.01
    )
    output.loss.backward()
    gradient_error = compare_gradients(model, reference)
    model.zero_grad(set_to_none=True)
    del output, correctness_inputs
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, decoder_type),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1, foreach=False
    )
    documents, eos = fixed_documents(str(checkpoint))
    local_rank = int(os.environ['LOCAL_RANK'])
    step_times, losses, norms = [], [], []
    updated_parameters = 0
    measurement_start = None
    measurement_end = None
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(args.num_warmup + args.num_steps):
            offset = (step * dist.get_world_size() + dist.get_rank()) * args.batch_size * args.seq_len
            tokens = packed_tokens(documents, eos, args.seq_len, args.batch_size, offset)
            inputs = torch.tensor(tokens, device='cuda', dtype=torch.long)
            if step == args.num_warmup:
                torch.cuda.synchronize()
                measurement_start = time.time()
            elapsed, loss, norm, changed = training_step(model, optimizer, inputs, verify_update=step == 0)
            updated_parameters += changed
            maximum_time = torch.tensor(elapsed, device='cuda', dtype=torch.float64)
            dist.all_reduce(maximum_time, op=dist.ReduceOp.MAX)
            if step >= args.num_warmup:
                step_times.append(maximum_time.item())
                losses.append(loss)
                norms.append(norm)
                measurement_end = time.time()
            if dist.get_rank() == 0:
                print(
                    'step={} global_loss={:.6f} max_rank_ms={:.3f}'.format(step, loss, maximum_time.item()), flush=True
                )
        memory = torch.tensor(torch.cuda.max_memory_allocated(), device='cuda', dtype=torch.int64)
        all_memory = [torch.zeros_like(memory) for _ in range(dist.get_world_size())]
        dist.all_gather(all_memory, memory)
        return {
            'step_ms': step_times,
            'loss': losses,
            'gradient_norm': norms,
            'updated_parameter_elements': updated_parameters,
            'gradient_probe_max_abs_error': gradient_error,
            'gradient_probe_rtol': 0.025,
            'gradient_probe_atol': 1e-5,
            'loss_token_count': args.num_steps * args.batch_size * dist.get_world_size() * (args.seq_len - 1),
            'peak_memory_bytes_per_gpu': [value.item() for value in all_memory],
            'world_size': dist.get_world_size(),
            'local_rank': local_rank,
            'optimizer': 'AdamW, FP32 master parameters and moments',
            'optimizer_resumed': False,
            'training_parallelism': 'FSDP full shard; no expert parallelism',
            'measurement_start_unix_seconds': measurement_start,
            'measurement_end_unix_seconds': measurement_end,
        }
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    """Include download, identity, reference, warmup and teardown in one job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--num_warmup', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--duration', type=int, default=3300)
    args = parser.parse_args()
    start = time.monotonic()
    signal.signal(signal.SIGTERM, interrupt_worker)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    set_cache_environment(args.cache_dir)
    import torch
    import torch.distributed as dist

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != args.num_gpus or torch.cuda.device_count() < args.num_gpus:
        raise ValueError('torchrun world size and visible GPU count must match num_gpus.')
    if args.num_gpus < 2 or args.batch_size < 1 or args.seq_len < 32 or args.num_warmup < 0 or args.num_steps < 1:
        raise ValueError('Invalid full-model training dimensions.')
    if not 0 < args.learning_rate < 1 or not 30 < args.duration < 3600:
        raise ValueError('Invalid learning rate or download-inclusive duration.')
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261015)
    dist.init_process_group(
        backend='nccl', timeout=timedelta(seconds=args.duration), device_id=torch.device('cuda', local_rank)
    )
    control_group = dist.new_group(backend='gloo', timeout=timedelta(seconds=args.duration))
    report = identity_report()
    checkpoint = args.cache_dir / 'checkpoint'
    reference_path = args.output.parent / 'training-reference.json'
    try:
        download_start = time.monotonic()
        if rank == 0:
            configure_cache(args.cache_dir)
            checkpoint = download_checkpoint(args.cache_dir)
        dist.barrier(group=control_group)
        report['download_verify_seconds'] = time.monotonic() - download_start
        reference_start = time.monotonic()
        if rank == 0:
            reference_model = load_model(str(checkpoint), torch.bfloat16).cuda()
            try:
                write_json(reference_path, training_reference(reference_model, str(checkpoint)))
            finally:
                del reference_model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        dist.barrier(group=control_group)
        reference = read_json(reference_path)
        report['reference_seconds'] = time.monotonic() - reference_start
        training_start = time.monotonic()
        report.update(run_training(args, reference, checkpoint))
        report['training_seconds'] = time.monotonic() - training_start
        report['correctness_passed'] = True
        report['torch_version'] = torch.__version__
        report['gpu_names'] = [torch.cuda.get_device_name(index) for index in range(args.num_gpus)]
    finally:
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()
    if rank == 0:
        report['teardown_complete'] = True
        report['total_seconds'] = time.monotonic() - start
        if report['total_seconds'] >= args.duration:
            raise TimeoutError('Training exceeded the download-inclusive duration.')
        report['status'] = 'success'
        write_json(args.output, report)


if __name__ == '__main__':
    main()
