#!/bin/bash
#SBATCH --account=q0000219
#SBATCH --partition=roquo
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# Build multi-GPU SBD (rwakizaka fork non-cuda-aware-mpi + NCCL warm-up fix) via CMake.
# SAFE-ALLREDUCE A/B VARIANT of build_sbd_mpi.sh: adds -DSBD_THRUST_SAFE_MPI_ALLREDUCE
# (the upstream #ifdef-guarded safe allreduce path that rwakizaka's working multi-node
# builds define) and emits a DISTINCT binary name so the working diag-gpu-mpi is preserved
# for A/B comparison. Promote to diag-gpu-mpi only after the multi-node test ladder passes.
# NOTE: shares the fixed build/nvhpc-thrust binaryDir with the UHF build -- run SEQUENTIALLY,
# never concurrently (each script rm -rf's it).
set -uo pipefail
REPO="$HOME/qcsc-prefect"; SBD="$REPO/algorithms/sbd"; SRC="$SBD/native/sbd_mpi"
NV=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5
module load cuda/13.2 nvhpc/26.5 2>/dev/null || true
export PATH="$NV/compilers/bin:$NV/comm_libs/hpcx/bin:$PATH"
export LD_LIBRARY_PATH="$NV/compilers/lib:$NV/cuda/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"

echo "=== tool check ==="
which nvc++ mpic++ cmake 2>&1 || module load cmake 2>/dev/null || true
which cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found"; module avail 2>&1 | grep -i cmake; }
cmake --version 2>/dev/null | head -1

echo "=== locate NCCL (nvhpc comm_libs) ==="
NCCL_ROOT=$(dirname "$(find $NV/comm_libs -name libnccl.so 2>/dev/null | head -1)" 2>/dev/null)
NCCL_ROOT=${NCCL_ROOT%/lib}
echo "NCCL_ROOT=$NCCL_ROOT"
find $NV/comm_libs -name "libnccl.so*" 2>/dev/null | head -2
find $NV -name "nccl.h" 2>/dev/null | head -1

echo "=== cmake configure (nvhpc-thrust preset + multi-GPU + NCCL + SAFE_MPI_ALLREDUCE) ==="
cd "$SRC"
rm -rf build/nvhpc-thrust
NCCL_INC=$(dirname "$(find $NV -name nccl.h 2>/dev/null | head -1)")
CUBLAS_LIB=$(dirname "$(find $NV -name libcublas.so 2>/dev/null | head -1)")
echo "CUBLAS_LIB=$CUBLAS_LIB"
cmake --preset nvhpc-thrust \
  -DSBD_USE_NCCL=ON \
  -DSBD_USE_CUBLAS=ON \
  -DSBD_USE_RANK_DISTRIBUTION=ON \
  -DSBD_USE_BLOCK_RANK_DISTRIBUTION=ON \
  -DCMAKE_CXX_FLAGS="-DSBD_PREFECT -DSBD_NON_CUDA_AWARE_MPI -DSBD_THRUST_SAFE_MPI_ALLREDUCE -I${NCCL_INC}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${NCCL_ROOT}/lib -lnccl -L${CUBLAS_LIB} -lcublas -cudalib=cublas" \
  2>&1 | tail -25
echo "=== build tpb_diag ==="
cmake --build build/nvhpc-thrust --target tpb_diag -j 8 2>&1 | tail -25

BIN="$SRC/build/nvhpc-thrust/apps/chemistry_tpb_selected_basis_diagonalization/diag"
echo "=== result ==="
if [ -f "$BIN" ]; then
  cp "$BIN" "$SBD/native/diag-gpu-mpi-safeallreduce"
  echo "OK: copied to $SBD/native/diag-gpu-mpi-safeallreduce"
  ldd "$SBD/native/diag-gpu-mpi-safeallreduce" 2>/dev/null | grep -iE "nccl|mpi|cuda" | head
else
  echo "BUILD FAILED: $BIN not found"
fi
echo "EXIT_build=$?"
