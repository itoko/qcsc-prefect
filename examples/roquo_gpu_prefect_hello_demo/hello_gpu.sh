#!/bin/bash
# ROQUO GB200 GPU hello demo: confirm a GPU is visible in the allocation, then run the demo flow.
#   sbatch --account=<code> --partition=roquo --gres=gpu:1 --time=00:10:00 hello_gpu.sh
#SBATCH --job-name=roquo-gpu-hello
#SBATCH --partition=roquo
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=%x.%j.out
set -euo pipefail

REPO="${QCSC_REPO:-$HOME/qcsc-prefect}"
module load cuda/13.2 nvhpc/26.5 hpcx/2.50 2>/dev/null || true

echo "=== GPU visible in this allocation? ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || {
  echo "no GPU visible — check --gres=gpu:1 and the roquo partition"; exit 1; }

cd "$REPO"
# Run the demo flow (submits a trivial in-allocation GPU command through the hpc-roquo block).
uv run python -c "import asyncio; from examples.roquo_gpu_prefect_hello_demo.flow import roquo_gpu_hello_flow; print(asyncio.run(roquo_gpu_hello_flow()))"
echo "EXIT_roquo_gpu_hello=$?"
