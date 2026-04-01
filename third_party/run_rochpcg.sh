#!/bin/bash

# =================================================
# Helper functions
# =================================================
help() {
    cat << EOF
rocHPCG helper script
Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    -h, --help    Show this help message and exit
    --npx         Number of processes in x dimension of process grid (default: ${npx})
    --npy         Number of processes in y dimension of process grid (default: ${npy})
    --npz         Number of processes in z dimension of process grid (default: ${npz})
    --nx          Problem size in x dimension (default: ${nx})
    --ny          Problem size in y dimension (default: ${ny})
    --nz          Problem size in z dimension (default: ${nz})
    --rt          Benchmarking time in seconds (> 1800s for official runs) (default: ${runtime})
    --tol         Residual tolerance, skip reference verification if set (default: ${tol})
    --pz          Partition boundary in z process dimension (default: 0, uniform grid)
    --zl          Local nz value for processes with z rank < pz (default: equal to ${nz})
    --zu          Local nz value for processes with z rank >= pz (default: equal to ${nz})
EOF
}

# =================================================
# Global variables
# =================================================
npx=1
npy=1
npz=1
nx=560
ny=280
nz=280
runtime=60
tol=1
pz=0
zl=${nz}
zu=${nz}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rochpcg_bin="${SCRIPT_DIR}/rochpcg"

if [[ ! -x "${rochpcg_bin}" ]]; then
  echo "Cannot find rochpcg binary at ${rochpcg_bin}"
  exit 1
fi

# =================================================
# Parameter parsing
# =================================================
GETOPT_PARSE=$(getopt --name "${0}" --options h --longoptions help,npx:,npy:,npz:,nx:,ny:,nz:,rt:,tol:,pz:,zl:,zu: -- "$@") \
  || { echo "getopt invocation failed; could not parse the command line"; exit 1; }

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help) help; exit 0 ;;
    --npx) npx=${2}; shift 2 ;;
    --npy) npy=${2}; shift 2 ;;
    --npz) npz=${2}; shift 2 ;;
    --nx) nx=${2}; shift 2 ;;
    --ny) ny=${2}; shift 2 ;;
    --nz)
        nz=${2}
        zl=${nz}
        zu=${nz}
        shift 2 ;;
    --rt) runtime=${2}; shift 2 ;;
    --tol) tol=${2}; shift 2 ;;
    --pz) pz=${2}; shift 2 ;;
    --zl) zl=${2}; shift 2 ;;
    --zu) zu=${2}; shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

# Build rochpcg arguments
rochpcg_args="--npx=${npx} --npy=${npy} --npz=${npz}"
rochpcg_args+=" --nx=${nx} --ny=${ny} --nz=${nz}"
rochpcg_args+=" --rt=${runtime}"
rochpcg_args+=" --tol=${tol}"
rochpcg_args+=" --pz=${pz}"
rochpcg_args+=" --zl=${zl}"
rochpcg_args+=" --zu=${zu}"

# =================================================
# Affinity setup
# =================================================
globalRank=${OMPI_COMM_WORLD_RANK:-0}
rank=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
size=${OMPI_COMM_WORLD_LOCAL_SIZE:-1}

#construct a list of all cpus, sorted by core
cpulist=$(lscpu --parse=CPU,CORE,NODE | awk '!/#/' | tr ',' "\t" | sort -k 2 -g -s)

#construct list of devices and their numa affinities
devicelist=$(hy-smi --csv --showtoponuma | tail -n +2 | tr ',' "\t")

#count the cpus per core
threads_per_core=$(echo "${cpulist}" | grep -c ".*	0	.*")

#remove the extra cpus on each core to make a list of just physical cores, then sort by numa domain
corelist=$(echo "$cpulist" | awk -v tpc=${threads_per_core} '(NR-1)%tpc==0' | sort -k 3 -g -s)

#count numa domains
line=($(echo "$cpulist" | tail -n 1))
n_numa=$((line[2]+1))

numa_core_counts=()
numa_proc_counts=()
for i in $(seq 1 ${n_numa}); do numa_core_counts+=(0); numa_proc_counts+=(0); done

#parse the list of cpus to array and count cpus in each numa
cpus=()
while read -a line; do
  cpus+=(${line[0]})
  ((numa_core_counts[${line[2]}]++)) || true
done <<< "${corelist}"

numa_core_offsets=(0)
for i in $(seq 1 $((n_numa-1))); do numa_core_offsets+=($((numa_core_offsets[$((i-1))] + numa_core_counts[$i]))); done

#parse device to numa mapping
device_to_numa=()
while read -a line; do
  device_to_numa+=(${line[1]})
done <<< "${devicelist}"

rank_to_device=()
n_devices=$(echo "${devicelist}" | grep -c "card")
for i in $(seq 0 $((size-1))); do
  rank_to_device+=($((i%n_devices)))
done

mygpu=${rank_to_device[rank]}
mynuma=${device_to_numa[mygpu]}

rank_to_numa=()
for i in $(seq 0 $((size-1))); do
  rank_to_numa+=(${device_to_numa[${rank_to_device[$((i%n_devices))]}]})
done

for i in $(seq 0 $((size-1))); do
  numa=${rank_to_numa[$i]}
  ((numa_proc_counts[numa]++)) || true
done

omp_num_threads=$((numa_core_counts[mynuma]/numa_proc_counts[mynuma]))

core_offset=${numa_core_offsets[mynuma]}
for i in $(seq 0 $((rank-1))); do
  numa=${rank_to_numa[$i]}
  if [[ $numa -eq $mynuma ]]; then
    core_offset=$((core_offset + omp_num_threads))
  fi
done

omp_places="{${cpus[core_offset]}}"
for c in $(seq 1 $((omp_num_threads-1))); do
  omp_places+=",{${cpus[core_offset+c]}}"
done

if [[ $omp_num_threads -gt 1 ]]; then
  places="{${cpus[core_offset]}-${cpus[core_offset+$((omp_num_threads-1))]}}"
else
  places="{${cpus[core_offset]}}"
fi

# Export OpenMP config
export OMP_NUM_THREADS=${omp_num_threads}
export OMP_PLACES=${omp_places}
export OMP_PROC_BIND=true

if [[ $globalRank -lt $size ]]; then
  echo "Node Binding: Process $rank [(nx,ny,nz)=(${nx},${ny},${nz})] GPU: $mygpu, NUMA: $mynuma, CPU Cores: $omp_num_threads - $places"
fi

# Run
numactl -N ${mynuma} -m ${mynuma} ${rochpcg_bin} ${rochpcg_args}
