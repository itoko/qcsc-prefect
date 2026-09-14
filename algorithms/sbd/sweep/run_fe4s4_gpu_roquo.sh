#!/bin/bash
#SBATCH --job-name=fe4s4-gpu
#SBATCH --account=q0000219
#SBATCH --partition=roquo
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# 4Fe-4S 72q GPU recovery on ROQUO GB200 (1 GPU; solver OOMs multi-rank). RHF by default.
# Runs the nvc++-built diag-gpu directly in this allocation (local target, no nested sbatch).
#   METHOD=rhf FE4S4_POOL=<npz> sbatch run_fe4s4_gpu_roquo.sh
#   Interactive: srun --account=q0000219 --partition=roquo --gres=gpu:4 --time=03:00:00 \
#                    bash run_fe4s4_gpu_roquo.sh
set -euo pipefail

REPO="$HOME/qcsc-prefect"
SBD="$REPO/algorithms/sbd"
SWEEP="$SBD/sweep"
export MY_PROJECT="$REPO"

# GPU compiler/runtime libs for diag-gpu (nvc++/CUDA runtime) + host BLAS.
NV=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5
module load cuda/13.2 nvhpc/26.5 2>/dev/null || true
export PATH="$NV/compilers/bin:$NV/comm_libs/hpcx/bin:$PATH"
export LD_LIBRARY_PATH="$NV/compilers/lib:$NV/cuda/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"

export UV_CACHE_DIR="/tmp/uvcache_$USER"
export TMPDIR="/tmp/uvtmp_$USER"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
export PATH="$SBD/.venv/bin:$PATH"

export FE_MOL="${FE_MOL:-fe4s4}"
export FE4S4_FCIDUMP="${FE4S4_FCIDUMP:-$SWEEP/fcidump_Fe4S4_MO.txt}"
export FE2S2_FCIDUMP="$SWEEP/fe2s2_40q.fcidump"
export FE4S4_METHOD="${METHOD:-rhf}"
export FE4S4_SQD_DIM="${FE4S4_SQD_DIM:-300000000}"
export FE4S4_RECSTEPS="${FE4S4_RECSTEPS:-5}"
export FE4S4_NBATCH="${FE4S4_NBATCH:-5}"
export ROQUO_OMPTHREADS="${ROQUO_OMPTHREADS:-140}"
# FE4S4_POOL must be set (the persisted sample pool npz).
export FE4S4_POOL="${FE4S4_POOL:?set FE4S4_POOL to the persisted raw_samples npz}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$SBD"
"$SBD/.venv/bin/python" "$SWEEP/run_fe4s4_gpu_roquo.py"
echo "EXIT_fe4s4_gpu=$?"
