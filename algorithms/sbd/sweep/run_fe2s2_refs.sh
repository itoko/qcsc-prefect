#!/bin/bash
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.err
# Compute reference energies (UHF/UCCSD/CCSD(T)/HCI/DMRG) -> runs/refs.json.
# Runs in the reference venv (refenv: pyscf + block2). DEFERRED until the live mem2 job ends.
#
#   sbatch --partition=mem2 --time=360 run_fe2s2_refs.sh
set -euo pipefail

source ~/load_env.sh
SWEEP="$MY_PROJECT/algorithms/sbd/sweep"
REFENV="$MY_SPACE/sweep/refenv"

export FE2S2_FCIDUMP="/2ndfs/ra010014/u14924_space/sweep/fe2s2_40q.fcidump"
export DMRG_M="${DMRG_M:-100,200,400,800,1200}"
export DMRG_SCRATCH="$MY_SPACE/sweep/dmrg_scratch_fe2s2"
export DMRG_THREADS="${DMRG_THREADS:-8}"
export OMP_NUM_THREADS="$DMRG_THREADS"
export REFS_OUT="$SWEEP/runs/refs.json"

mkdir -p "$SWEEP/runs" "$DMRG_SCRATCH"
"$REFENV/bin/python" "$SWEEP/build_fe2s2_refs.py"
echo "EXIT_fe2s2_refs=$?"
