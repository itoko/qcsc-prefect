#!/bin/bash
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.err
# Offline multi-iteration recovery sweep from the persisted pool. NO IBM call.
#
# CONSTRAINT: one Fugaku job at a time -- do not submit while another is running.
#
#   METHOD=uhf MAX_RECOVERY=10 sbatch --partition=mem2 --time=360 run_fe2s2_recover.sh
set -euo pipefail

source ~/load_env.sh
export UV_PYTHON_INSTALL_DIR="$MY_SPACE/uv_python"

SWEEP="$MY_PROJECT/algorithms/sbd/sweep"
cd "$MY_PROJECT/algorithms/sbd"
export PATH="$MY_PROJECT/algorithms/sbd/.venv/bin:$PATH"

METHOD="${METHOD:-uhf}"
MAX_RECOVERY="${MAX_RECOVERY:-10}"
NBATCH="${NBATCH:-5}"

python "$SWEEP/fe2s2_recover.py" \
    --method "$METHOD" --max-recovery "$MAX_RECOVERY" --n-batches "$NBATCH"
echo "EXIT_fe2s2_recover_${METHOD}=$?"
