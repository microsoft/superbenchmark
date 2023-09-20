#!/bin/bash

# set to GPU 1
export CUDA_VISIBLE_DEVICES=3

# Matrix A: Regular, Matrix B: Tall and skinny. No strided batch
echo "Matrix A: Regular, Matrix B: Tall and skinny. No strided batch"
./cublaslt_gemm -m 49152 -n 4096 -k 49152 -b 4 -w 50 -i 2000 -t fp16
sleep 30

# Matrix A: Tall and skinny, Matrix B: Regular. No strided batch
echo "Matrix A: Tall and skinny, Matrix B: Regular. No strided batch"
./cublaslt_gemm -m 4096 -n 49152 -k 49152 -b 4 -w 50 -i 2000 -t fp16
sleep 30

# Matrix A: Regular, Matrix B: Regular. No strided batch
echo "Matrix A: Regular, Matrix B: Regular. No strided batch"
./cublaslt_gemm -m 24576 -n 24576 -k 24576 -b 4 -w 50 -i 2000 -t fp16
sleep 30

# Matrix A: Regular, Matrix B: Tall and skinny. Strided batch
#echo "Matrix A: Regular, Matrix B: Tall and skinny. Strided batch"
#./cublaslt_gemm -m 8192 -n 512 -k 8192 -b 256 -w 50 -i 1000 -t fp16
#sleep 30

# Matrix A: Tall and skinny, Matrix B: Regular. Strided batch
#echo "Matrix A: Tall and skinny, Matrix B: Regular. Strided batch"
#./cublaslt_gemm -m 512 -n 8192 -k 8192 -b 256 -w 50 -i 1000 -t fp16
#sleep 30

# Matrix A: Regular, Matrix B: Regular. Strided batch
#echo "Matrix A: Regular, Matrix B: Regular. Strided batch"
#./cublaslt_gemm -m 8192 -n 8192 -k 8192 -b 256 -w 50 -i 1000 -t fp16
