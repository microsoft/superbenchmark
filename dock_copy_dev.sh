#!/bin/bash
sudo docker cp /home/azureuser/superbenchmark/tests/benchmarks/micro_benchmarks/test_cpu_stream_performance.py  sb-workspace:/opt/superbench/tests/benchmarks/micro_benchmarks/
sudo docker cp /home/azureuser/superbenchmark/tests/data/streamResult.log  sb-workspace:/opt/superbench/tests/data/
sudo docker cp /home/azureuser/superbenchmark/superbench/benchmarks/micro_benchmarks/__init__.py sb-workspace:/opt/superbench/superbench/benchmarks/micro_benchmarks/
sudo docker cp /home/azureuser/superbenchmark/superbench/benchmarks/micro_benchmarks/cpu_stream_performance.py sb-workspace:/opt/superbench/superbench/benchmarks/micro_benchmarks/
sudo docker cp /home/azureuser/streamZen3.exe sb-workspace:/usr/local/bin