#!/bin/bash
#SBATCH --job-name=fe4s4-rhf-sample
#SBATCH --account=q0000219
#SBATCH --partition=roquo
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# 4Fe-4S 72q sampling on ROQUO (IBM kobe from compute node) -> save pool. RHF by default.
#   METHOD=rhf sbatch run_fe4s4_sample_roquo.sh
#   FE2S2_QSRC=random METHOD=rhf sbatch run_fe4s4_sample_roquo.sh   # dry-run
set -euo pipefail

REPO="$HOME/qcsc-prefect"; SBD="$REPO/algorithms/sbd"; SWEEP="$SBD/sweep"
export MY_PROJECT="$REPO"
NV=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5
module load cuda/13.2 hpcx/2.50 2>/dev/null || true
export PATH="$NV/compilers/bin:$NV/comm_libs/hpcx/bin:$SBD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$NV/compilers/lib:$NV/cuda/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"
export UV_CACHE_DIR="/tmp/uvcache_$USER"; export TMPDIR="/tmp/uvtmp_$USER"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

export FE_MOL="${FE_MOL:-fe4s4}"
export FE4S4_FCIDUMP="$SWEEP/fcidump_Fe4S4_MO.txt"
export FE2S2_FCIDUMP="$SWEEP/fe2s2_40q.fcidump"
export FE4S4_METHOD="${METHOD:-rhf}"
export ROQUO_OMPTHREADS="${ROQUO_OMPTHREADS:-140}"

# IBM creds (kobe) from gitignored .env.local.
if [[ -f "$SWEEP/.env.local" ]]; then source "$SWEEP/.env.local"; else
  echo "ERROR: $SWEEP/.env.local missing (IBM_API_KEY/IBM_CRN/IBM_BACKEND)"; exit 1; fi
export FE2S2_QSRC="${FE2S2_QSRC:-real-device}"

cd "$SBD"
"$SBD/.venv/bin/python" "$SWEEP/run_fe4s4_sample_roquo.py"
echo "EXIT_fe4s4_sample_roquo=$?"
