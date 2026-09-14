#!/bin/bash
#SBATCH --output=/vol0206/data/ra010014/u14924/u14924_space/sweep/run_fe4s4_recover.%j.log
#SBATCH --error=/vol0206/data/ra010014/u14924/u14924_space/sweep/run_fe4s4_recover.%j.err
# 4Fe-4S 72q UHF multi-iteration recovery from the SAVED kobe pool. NO hardware.
# Orchestrator on mem2 (x86_64); spawns the 3600-node aarch64 solver via PJM.
#   sbatch --partition=mem2 --time=360 run_fe4s4_recover.sh
source ~/load_env.sh
export UV_PYTHON_INSTALL_DIR="$MY_SPACE/uv_python"
cd "$MY_PROJECT/algorithms/sbd"
export PATH="$MY_PROJECT/algorithms/sbd/.venv/bin:$PATH"

export FE_MOL="fe4s4"
export FE4S4_FCIDUMP="$MY_SPACE/sweep/fcidump_Fe4S4_MO.txt"
export FE4S4_METHOD="${FE4S4_METHOD:-uhf}"

# The persisted 5M-shot kobe pool (backed up to a data area). Override FE4S4_POOL if needed.
export FE4S4_POOL="${FE4S4_POOL:-$MY_SPACE/sweep/fe4s4_pools/raw_samples_t00_w0_853cf4ae7e284baeac95de5d7c14c36e.npz}"

export FE4S4_SQD_DIM="${FE4S4_SQD_DIM:-300000000}"   # 3e8
export FE4S4_RECSTEPS="${FE4S4_RECSTEPS:-5}"
export FE4S4_NBATCH="${FE4S4_NBATCH:-5}"
export FE4S4_NODES="${FE4S4_NODES:-3600}"
export FE4S4_ADET="${FE4S4_ADET:-60}"
export FE4S4_BDET="${FE4S4_BDET:-60}"
export FE4S4_QUEUE="${FE4S4_QUEUE:-large}"

"$MY_PROJECT/algorithms/sbd/.venv/bin/python" \
    "$MY_PROJECT/algorithms/sbd/sweep/run_fe4s4_recover.py"
rc=$?
echo "EXIT_fe4s4_recover=$rc"
exit $rc
