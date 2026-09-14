#!/bin/bash
#SBATCH --job-name=fe2s2-rhf
#SBATCH --account=q0000219
#SBATCH --partition=roquo
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# Fe2S2 RHF SQD on ROQUO CPU (aarch64). Orchestrator + solver run in THIS one allocation
# (solver uses launcher=single, no nested sbatch). ROQUO is a separate cluster from Fugaku, so the
# Fugaku one-job-at-a-time limit does not apply here.
#
# Real device:   sbatch run_fe2s2_rhf_roquo.sh
# Dry-run (no IBM):  FE2S2_QSRC=random sbatch run_fe2s2_rhf_roquo.sh
# Interactive:   srun --account=q0000219 --partition=roquo --gres=gpu:1 --time=02:00:00 \
#                    bash run_fe2s2_rhf_roquo.sh
set -euo pipefail

REPO="$HOME/qcsc-prefect"
SBD="$REPO/algorithms/sbd"
SWEEP="$SBD/sweep"
export MY_PROJECT="$REPO"

# Runtime libs for the aarch64 diag binary (OpenBLAS + hpcx MPI).
module load gcc/14 hpcx/2.50 2>/dev/null || true

# Node-local uv cache/tmp: Lustre home breaks uv's build-tempdir rmtree (see roquo memory).
export UV_CACHE_DIR="/tmp/uvcache_$USER"
export TMPDIR="/tmp/uvtmp_$USER"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

export PATH="$SBD/.venv/bin:$PATH"

# IBM credentials from the gitignored .env.local (never committed).
if [[ -f "$SWEEP/.env.local" ]]; then
    # shellcheck disable=SC1091
    source "$SWEEP/.env.local"
fi

export FE2S2_QSRC="${FE2S2_QSRC:-real-device}"
export ROQUO_OMPTHREADS="${ROQUO_OMPTHREADS:-140}"
# FCIDUMP lives here on ROQUO (transferred from Fugaku); override the Fugaku-path default.
export FE2S2_FCIDUMP="${FE2S2_FCIDUMP:-$SWEEP/fe2s2_40q.fcidump}"

cd "$SBD"
"$SBD/.venv/bin/python" "$SWEEP/run_fe2s2_rhf_roquo.py"
echo "EXIT_fe2s2_rhf_roquo=$?"
