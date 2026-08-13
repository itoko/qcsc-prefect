#!/bin/bash
#SBATCH --account=q0000219
#SBATCH --partition=roquo
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# HPCX/2.50-CONSISTENT VARIANT: builds AND will be run against the standalone hpcx/2.50
# module (DOCA-OFED-optimized OpenMPI5), NOT NVHPC's bundled hpcx (comm_libs/hpcx) and NOT
# the nvhpc-openmpi/26.5 module (ompi4) used by build_sbd_mpi_uhf_nvhpcmpi.sh. Rationale
# (2026-07-31/08-01 investigation): both the 16-node/64-rank and 25-node/100-rank runs on
# the nvhpc-openmpi/26.5 (ompi4) stack deadlock in NCCL's bootstrap ring all-gather
# (ncclCommInitRank -> bootstrapInit -> socketRingAllGather -> recv(), confirmed via gdb on
# BOTH jobs, all ranks alive, H2 [unset OMPI_MCA_mca_base_env_list] and H3 [NCCL_SOCKET_
# IFNAME=ib0 + NCCL_BOOTSTRAP_TIMEOUT=120] already applied). Project memory
# (qcsc-multinode-fix, 2026-07-23) shows this EXACT site-3 hang was previously cleared with
# H2+H3 together, but only validated up to 16 ranks (a=4,b=4) -- our current failures are at
# 64-100 ranks, untested territory. The ROQUO MPI manual's own tested multi-node recipe
# explicitly uses hpcx/2.50 (mpirun), not the nvhpc-bundled or nvhpc-openmpi stacks. This
# script builds mpic++ FROM hpcx/2.50 (not nvhpc's own wrapper) so build-time RUNPATH and
# run-time LD_LIBRARY_PATH are CONSISTENT (the root cause of the earlier, separate DT_RUNPATH
# substitution bug), with OMPI_CXX/OMPI_CC pointed at nvc++/nvc for the actual CUDA/Thrust
# device-code compilation.
set -uo pipefail
REPO="$HOME/qcsc-prefect"; SBD="$REPO/algorithms/sbd"; SRC="$SBD/native/sbd_mpi"
NV=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5

module purge
# Load nvhpc FIRST (for nvc++/cuda/cmake), then hpcx/2.50 SECOND so its mpic++/PATH/
# LD_LIBRARY_PATH take precedence over nvhpc's own bundled hpcx for the MPI wrapper.
module load gcc/14 nvhpc/26.5
module load hpcx/2.50
export OMPI_CXX=nvc++
export OMPI_CC=nvc

echo "=== tool check (mpic++ must resolve to the STANDALONE hpcx/2.50 tree, not nvhpc's bundled one) ==="
which nvc++ mpic++ cmake 2>&1 || module load cmake 2>/dev/null || true
which cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found"; module avail 2>&1 | grep -i cmake; }
cmake --version 2>/dev/null | head -1
readlink -f "$(which mpic++)"
echo "--- mpic++ -show (must invoke nvc++ as the underlying compiler) ---"
mpic++ -show 2>&1

echo "=== verify PR79 cherry-pick + CUDA-before-MPI_Init fix are present ==="
cd "$SRC"
git log --oneline -3
grep -n "h_size == 1" include/sbd/chemistry/basic/davidson_thrust.h || {
  echo "ERROR: PR79 fix not found in davidson_thrust.h -- did the cherry-pick land?"; exit 1;
}
grep -n "cudaSetDevice(0)" apps/chemistry_tpb_selected_basis_diagonalization/main.cc || {
  echo "ERROR: CUDA-before-MPI_Init fix not found in main.cc"; exit 1;
}

echo "=== locate NCCL (nvhpc comm_libs) ==="
NCCL_ROOT=$(dirname "$(find $NV/comm_libs -name libnccl.so 2>/dev/null | head -1)" 2>/dev/null)
NCCL_ROOT=${NCCL_ROOT%/lib}
echo "NCCL_ROOT=$NCCL_ROOT"
find $NV/comm_libs -name "libnccl.so*" 2>/dev/null | head -2
find $NV -name "nccl.h" 2>/dev/null | head -1

echo "=== cmake configure (nvhpc-thrust preset's settings, hpcx/2.50 mpic++, SBD_GPU_ARCH=cc100 for GB200/Blackwell) ==="
rm -rf build/hpcx250-thrust
NCCL_INC=$(dirname "$(find $NV -name nccl.h 2>/dev/null | head -1)")
CUBLAS_LIB=$(dirname "$(find $NV -name libcublas.so 2>/dev/null | head -1)")
echo "CUBLAS_LIB=$CUBLAS_LIB"
cmake -S . -B build/hpcx250-thrust \
  --toolchain cmake/toolchains/nvhpc.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DSBD_GPU_BACKEND=thrust \
  -DSBD_GPU_ARCH=cc100 \
  -DSBD_THRUST_SAFE_MPI_ALLREDUCE=ON \
  -DSBD_USE_NCCL=ON \
  -DSBD_USE_CUBLAS=ON \
  -DSBD_USE_RANK_DISTRIBUTION=ON \
  -DSBD_USE_BLOCK_RANK_DISTRIBUTION=ON \
  -DCMAKE_CXX_FLAGS="-D_UHF -DSBD_PREFECT -DSBD_NON_CUDA_AWARE_MPI -DSBD_THRUST_SAFE_MPI_ALLREDUCE -I${NCCL_INC}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${NCCL_ROOT}/lib -lnccl -L${CUBLAS_LIB} -lcublas -cudalib=cublas" \
  2>&1 | tail -35
echo "=== build tpb_diag ==="
cmake --build build/hpcx250-thrust --target tpb_diag -j 8 2>&1 | tail -30

BIN="$SRC/build/hpcx250-thrust/apps/chemistry_tpb_selected_basis_diagonalization/diag"
echo "=== result ==="
if [ -f "$BIN" ]; then
  cp "$BIN" "$SBD/native/diag-gpu_uhf-mpi-hpcx250"
  echo "OK: copied to $SBD/native/diag-gpu_uhf-mpi-hpcx250"
  echo "--- RUNPATH ---"
  readelf -d "$SBD/native/diag-gpu_uhf-mpi-hpcx250" | grep -i 'rpath\|runpath'
  echo "--- ldd (mpi/nccl/cuda libs) ---"
  ldd "$SBD/native/diag-gpu_uhf-mpi-hpcx250" 2>/dev/null | grep -iE "nccl|mpi|cuda|ucx|pmix"
else
  echo "BUILD FAILED: $BIN not found"
fi
echo "EXIT_build=$?"
