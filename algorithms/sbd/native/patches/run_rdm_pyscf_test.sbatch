#!/usr/bin/env bash
#SBATCH -J rdm_pyscf
#SBATCH -A qc-prj-other02
#SBATCH -p roquo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH -o rdm_pyscf_%j.out
#SBATCH -e rdm_pyscf_%j.out
# Standalone batch runner for the opposite-spin 2-RDM sign proof.
#   sbatch native/patches/run_rdm_pyscf_test.sbatch
# Requires billing points on account qc-prj-other02. If you have none, piggyback on an
# already-running allocation instead (no points/account needed):
#   srun --jobid=<your running jobid> --overlap -N1 -n1 \
#        .venv/bin/python native/patches/test_rdm_vs_pyscf.py
set -euo pipefail
cd "$HOME/qcsc-prefect/algorithms/sbd"
echo "host=$(hostname) arch=$(uname -m)"
PY="$HOME/qcsc-prefect/algorithms/sbd/.venv/bin/python"
export OMP_NUM_THREADS=4
"$PY" native/patches/test_rdm_vs_pyscf.py
