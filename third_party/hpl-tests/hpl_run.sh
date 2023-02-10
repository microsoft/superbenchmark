#!/bin/bash

source /opt/hpcx/hpcx-init.sh
hpcx_load

# On the off chance OMPI_MCA is set to UCX-only, disable that
unset OMPI_MCA_osc

BIN_PATH="$(dirname "$0")"
XHPL_EXE=$1
NCORES=$2

NT=${NCORES}
NR=2
MAP_BY=socket
set -x

mpirun --map-by ${MAP_BY}:PE=$NT -np $NR --bind-to core \
    -x OMP_NUM_THREADS=$NT -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
    ${BIN_PATH}/bindmem.sh ${BIN_PATH}/${XHPL_EXE} 
