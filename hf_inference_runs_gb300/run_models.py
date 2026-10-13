#!/usr/bin/env python3
"""Run the Hugging Face inference matrix on GB300 GPUs and persist summaries."""

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import queue
import shutil
import socket
import statistics
import subprocess
import threading

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / 'hf_inference_runs_gb300'
CACHE_DIR = Path.home() / '.cache' / 'huggingface'
IMAGE = 'superbench-dev:cuda13.0-arm64'
LOCK = threading.Lock()
PRECISION = 'float16'


def metric_precision():
    return {'float16': 'fp16', 'float32': 'fp32', 'bfloat16': 'bf16'}[PRECISION]


def load_models():
    with (OUTPUT_ROOT / 'models.tsv').open(newline='') as file:
        return list(csv.DictReader(file, delimiter='\t'))


def write_config(model, output_dir):
    config = f"""version: v0.11
superbench:
  enable:
  - model-benchmarks:{model['slug']}
  monitor:
    enable: false
  benchmarks:
    model-benchmarks:{model['slug']}:
      modes:
      - name: local
        proc_num: 1
        parallel: no
      frameworks: [pytorch]
      timeout: 14400
      models:
      - hf-lm
      parameters:
        duration: 0
        num_warmup: 4
        num_steps: 16
        sample_count: 32
        batch_size: 1
        precision: [{PRECISION}]
        model_action: [inference]
        pin_memory: yes
        num_workers: 0
        seq_len: 128
        model_source: huggingface
        model_identifier: {model['model_id']}
"""
    (output_dir / 'config.yaml').write_text(config)


def successful_result(output_dir):
    path = output_dir / 'results.json'
    if not path.exists():
        return False
    try:
        records = json.loads(path.read_text())
        return bool(records and records[0]['result']['return_code'][0] == 0)
    except Exception:
        return False


def summarize(model, output_dir):
    output_relative = output_dir.relative_to(ROOT)
    source_files = list((output_dir / 'work' / 'benchmarks').glob('**/rank0/results.json'))
    if not source_files:
        raise RuntimeError('SuperBench did not produce rank0/results.json')
    shutil.copy2(source_files[0], output_dir / 'results.json')
    record = json.loads((output_dir / 'results.json').read_text())[0]
    result = record['result']
    benchmark = record['name']
    metric_prefix = metric_precision()
    metric_names = [f'{metric_prefix}_inference_step_time', f'{metric_prefix}_inference_throughput']
    flattened = {
        f'{benchmark}/{metric}': result[metric][0]
        for metric in metric_names
    }
    flattened['node'] = socket.gethostname()
    (output_dir / 'results-summary.jsonl').write_text(json.dumps(flattened) + '\n')

    rules = f"""version: v0.12
superbench:
  rules:
    hf-inference:
      statistics:
      - mean
      categories: {model['family']} Inference
      metrics:
            - {benchmark}/{metric_prefix}_inference_step_time
            - {benchmark}/{metric_prefix}_inference_throughput
"""
    (output_dir / 'summary-rules.yaml').write_text(rules)

    command = [
        'docker', 'run', '--rm',
        '-e', 'PYTHONPATH=/root/superbenchmark',
        '-v', f'{ROOT}:/root/superbenchmark',
        '-w', '/root/superbenchmark', IMAGE,
        'sb', 'result', 'summary',
        '--data-file', f'/root/superbenchmark/{output_relative}/results-summary.jsonl',
        '--rule-file', f'/root/superbenchmark/{output_relative}/summary-rules.yaml',
        '--output-file-format', 'md', '--decimal-place-value', '3',
        '--output-dir', f'/root/superbenchmark/{output_relative}',
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    raw_times = record.get('raw_data', {}).get(f'{metric_prefix}_inference_step_time', [[]])[0]
    latency = result[f'{metric_prefix}_inference_step_time'][0]
    throughput = result[f'{metric_prefix}_inference_throughput'][0]
    flags = []
    if not math.isfinite(latency) or latency <= 0:
        flags.append('non-positive or non-finite latency')
    if not math.isfinite(throughput) or throughput <= 0:
        flags.append('non-positive or non-finite throughput')
    if latency > 0 and abs(throughput - 1000 / latency) / throughput > 0.10:
        flags.append('latency/throughput differ by more than 10%')
    if len(raw_times) > 1:
        coefficient_of_variation = statistics.stdev(raw_times) / statistics.mean(raw_times)
        if coefficient_of_variation > 0.10:
            flags.append(f'high step-time variation ({coefficient_of_variation:.1%})')
    analysis = 'No automatic anomaly detected.' if not flags else '; '.join(flags)
    (output_dir / 'analysis.txt').write_text(analysis + '\n')
    return latency, throughput, analysis


def run_one(model, gpu):
    output_dir = OUTPUT_ROOT / model['slug']
    output_dir.mkdir(parents=True, exist_ok=True)
    if successful_result(output_dir):
        latency, throughput, analysis = summarize(model, output_dir)
        return {**model, 'status': 'SUCCESS', 'latency': latency, 'throughput': throughput, 'analysis': analysis}

    write_config(model, output_dir)
    command = [
        'docker', 'run', '--rm', '--init', '--gpus', f'device={gpu}', '--ipc=host',
        '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
        '-e', 'PYTHONPATH=/root/superbenchmark', '-e', 'HF_HOME=/hf-cache',
        '-v', f'{ROOT}:/root/superbenchmark',
        '-v', f'{CACHE_DIR}:/hf-cache',
        '-v', f'{output_dir}:/results',
        '-w', '/root/superbenchmark', IMAGE,
        'sb', 'exec', '-c', '/results/config.yaml', '--output-dir', '/results/work',
    ]
    with (output_dir / 'run.log').open('a') as log:
        log.write(f'GPU {gpu}: {" ".join(command)}\n')
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        return {**model, 'status': 'FAILED', 'analysis': f'docker/sb exit code {completed.returncode}'}
    try:
        latency, throughput, analysis = summarize(model, output_dir)
        return {**model, 'status': 'SUCCESS', 'latency': latency, 'throughput': throughput, 'analysis': analysis}
    except Exception as error:
        return {**model, 'status': 'FAILED', 'analysis': str(error)}


def write_report(results):
    results = sorted(results, key=lambda item: item['family'].lower())
    lines = [
        '# Hugging Face Inference Runs on GB300', '',
        f'| Family | Model | Status | {PRECISION} latency (ms) | Throughput (samples/s) | Analysis |',
        '| --- | --- | --- | ---: | ---: | --- |',
    ]
    for item in results:
        latency = f"{item['latency']:.3f}" if 'latency' in item else '-'
        throughput = f"{item['throughput']:.3f}" if 'throughput' in item else '-'
        analysis = item.get('analysis', '').replace('|', '\\|')
        lines.append(
            f"| {item['family']} | `{item['model_id']}` | {item['status']} | "
            f"{latency} | {throughput} | {analysis} |"
        )
    (OUTPUT_ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    (OUTPUT_ROOT / 'run-status.json').write_text(json.dumps(results, indent=2) + '\n')


def main():
    global OUTPUT_ROOT, PRECISION
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--precision', choices=['float16', 'float32', 'bfloat16'], default=PRECISION)
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_root.resolve()
    PRECISION = args.precision
    models = load_models()
    results = [
        {
            **model,
            'status': 'SKIPPED',
            'analysis': 'Official checkpoint access is pending Hugging Face approval.',
        }
        for model in models if model.get('action') == 'skip'
    ]
    write_report(results)
    model_queue = queue.Queue()
    for model in models:
        if model.get('action') != 'skip':
            model_queue.put(model)

    def gpu_worker(gpu):
        while True:
            try:
                model = model_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_one(model, gpu)
            except Exception as error:
                result = {**model, 'status': 'FAILED', 'analysis': str(error)}
            finally:
                model_queue.task_done()
            with LOCK:
                results.append(result)
                write_report(results)
                print(f"{result['status']}: {model['family']} ({model['model_id']})", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(gpu_worker, gpu) for gpu in range(args.workers)]
        for future in futures:
            future.result()
    return 1 if any(item['status'] == 'FAILED' for item in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
