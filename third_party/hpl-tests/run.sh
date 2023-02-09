#!/bin/bash

# On the off chance OMPI_MCA is set to UCX-only, disable that
unset OMPI_MCA_osc


ldd ./xhpl > ldd_output.log
NT=60
NR=2
MAP_BY=socket
set -x

mpirun --map-by ${MAP_BY}:PE=$NT -np $NR --bind-to core \
    -x OMP_NUM_THREADS=$NT -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
    ./bindmem.sh ./xhpl
