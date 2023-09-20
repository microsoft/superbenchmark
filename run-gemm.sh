#!/bin/bash

sudo docker cp $1 sb-workspace:/root/run-config.yaml

sudo docker exec sb-workspace sudo nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=1000 --trace-fork-before-exec=true -f true -o test sb run --no-docker -l localhost -c run-config.yaml

sudo docker cp sb-workspace:/root/test.nsys-rep reports/test.nsys-rep
