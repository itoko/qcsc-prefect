#!/bin/bash
#SBATCH --output=/vol0206/data/ra010014/u14924/u14924_space/sweep/run_fe4s4_sample.%j.log
#SBATCH --error=/vol0206/data/ra010014/u14924/u14924_space/sweep/run_fe4s4_sample.%j.err
# 4Fe-4S 72q UHF sampling ORCHESTRATOR on Fugaku. Runs on the mem2 pre/post partition (x86_64,
# matches the venv python arch); it submits the SBD solver as its own large-node PJM job
# (FE4S4_NODES, default 3600 = 60x60 grid, queue "large") for a fast turnaround.
#   Real:    sbatch --partition=mem2 --time=360 run_fe4s4_sample.sh
#   Dry-run: FE2S2_QSRC=random sbatch --partition=mem2 --time=120 run_fe4s4_sample.sh
# ONE Fugaku job at a time (this orchestrator + its child solver job).
source ~/load_env.sh
export UV_PYTHON_INSTALL_DIR="$MY_SPACE/uv_python"
cd "$MY_PROJECT/algorithms/sbd"
export PATH="$MY_PROJECT/algorithms/sbd/.venv/bin:$PATH"

export FE_MOL="fe4s4"
export FE4S4_FCIDUMP="$MY_SPACE/sweep/fcidump_Fe4S4_MO.txt"
export FE4S4_METHOD="${FE4S4_METHOD:-uhf}"
export FE4S4_NODES="${FE4S4_NODES:-3600}"
export FE4S4_ADET="${FE4S4_ADET:-60}"
export FE4S4_BDET="${FE4S4_BDET:-60}"
export FE4S4_QUEUE="${FE4S4_QUEUE:-large}"

# IBM credentials (kobe) from the gitignored sweep/.env.local (never committed).
if [[ -f "$MY_SPACE/sweep/.env.local" ]]; then
    # shellcheck disable=SC1091
    source "$MY_SPACE/sweep/.env.local"
else
    echo "ERROR: $MY_SPACE/sweep/.env.local not found (IBM_API_KEY/IBM_CRN/IBM_BACKEND)." >&2
    exit 1
fi

export FE2S2_QSRC="${FE2S2_QSRC:-real-device}"
export PREFECT_HOME="$MY_SPACE/sweep/runs/fe4s4_uhf/prefect_home"
export PREFECT_LOCAL_STORAGE_PATH="$PREFECT_HOME/storage"
rm -rf "$PREFECT_HOME"; mkdir -p "$PREFECT_HOME"

"$MY_PROJECT/algorithms/sbd/.venv/bin/python" \
    "$MY_PROJECT/algorithms/sbd/sweep/run_fe4s4_sample.py"
rc=$?
echo "EXIT_fe4s4_sample=$rc"
exit $rc
