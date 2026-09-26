#!/bin/bash -l
#SBATCH --account=qc-prj-other02
#SBATCH --partition=roquo
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=36
#SBATCH --time=00:30:00
#SBATCH --job-name=qcsc-uhf-fixed-build
#SBATCH --output=/home/nfs1/q0000230/qcsc-prefect/algorithms/sbd/native/build_uhf_fixed.%j.out
#SBATCH --error=/home/nfs1/q0000230/qcsc-prefect/algorithms/sbd/native/build_uhf_fixed.%j.err

set -euo pipefail
case "${SLURM_JOB_PARTITION:-}" in
  *large*) echo "Refusing large partition." >&2; exit 2;;
esac
if [ "${SLURM_JOB_RESERVATION:-}" = "large-20260824" ]; then
  echo "Refusing large reservation." >&2; exit 2
fi

module load nvhpc/26.5
module load nvhpc-openmpi/26.5

cd /home/nfs1/q0000230/qcsc-prefect/algorithms/sbd/native
./build_sbd_gpu_uhf_fixed.sh

echo "=== result ==="
file diag-gpu_uhf_fixed
