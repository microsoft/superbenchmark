#!/bin/bash

# Set the CUDA visible devices
export CUDA_VISIBLE_DEVICES=1

# Set the number of instances to run
NUM_INSTANCES=10


# Run the instances in parallel
for ((i=0; i<$NUM_INSTANCES; i++)); do
    ./cublaslt_gemm -m 1024 -n 1024 -k 1024 -b 0 -w 50 -i 1000 -t fp32 & 
done

# Wait for all instances to finish
wait
