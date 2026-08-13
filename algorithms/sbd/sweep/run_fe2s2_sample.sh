#!/bin/bash
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.err
# Sample the Fe2S2 LUCJ circuit ONCE for a method and persist the pool.
#
# CONSTRAINT: only ONE Fugaku job may be in flight at a time. Do NOT submit while another job
# (e.g. a recovery run) is queued/running. Check with `pjstat` first.
#
# Real device (default):   METHOD=uhf sbatch --partition=mem2 --time=360 run_fe2s2_sample.sh
# Cheap dry-run (no IBM):   FE2S2_QSRC=random METHOD=uhf sbatch --partition=mem2 --time=60 run_fe2s2_sample.sh
set -euo pipefail

source ~/load_env.sh
export UV_PYTHON_INSTALL_DIR="$MY_SPACE/uv_python"

SWEEP="$MY_PROJECT/algorithms/sbd/sweep"
cd "$MY_PROJECT/algorithms/sbd"
export PATH="$MY_PROJECT/algorithms/sbd/.venv/bin:$PATH"

# IBM credentials from the gitignored .env.local (not committed).
if [[ -f "$SWEEP/.env.local" ]]; then
    # shellcheck disable=SC1091
    source "$SWEEP/.env.local"
fi

export FE2S2_QSRC="${FE2S2_QSRC:-real-device}"
METHOD="${METHOD:-uhf}"

python "$SWEEP/fe2s2_sample.py" --method "$METHOD"
echo "EXIT_fe2s2_sample_${METHOD}=$?"
