#!/bin/bash
#PBS -q debug-g
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=0:30:00
#PBS -W group_list=gr42
#PBS -N sbd-mpi-uhf-build
#PBS -j oe
# NOTE: #PBS directives are static text, NOT shell-expanded -- if your group isn't gr42, edit
# the group_list line above directly before submitting (qsub -v cannot override a #PBS line).
# Build multi-GPU SBD (rwakizaka fork non-cuda-aware-mpi + NCCL warm-up fix) via CMake, for
# Miyabi-G (GH200 Grace-Hopper, 1 H100/node -- ANY multi-rank run is cross-node here, so this
# -mpi binary is required for every multi-node run, not just an optional multi-GPU-per-node case
# the way it is on ROQUO). Adapted from build_sbd_mpi_uhf.sh (ROQUO) -- same CMake preset/flags,
# only the module names and NVHPC-root discovery differ.
#
# Prerequisite (one-time, not done by this script -- matches algorithms/sbd/native/README.md):
#   cd algorithms/sbd/native
#   git clone --branch non-cuda-aware-mpi https://github.com/rwakizaka/sbd.git sbd_mpi
#   # then apply the NCCL warm-up fix described in README.md (shrink the warm-up device_vector
#   # in include/sbd/chemistry/tpb/sbdiag.h from A(W.size(),0.0) to A(1,0.0))
#
# Submit with: qsub build_sbd_mpi_uhf_miyabi.sh
set -uo pipefail
REPO="${MIYABI_REPO:-$HOME/qcsc-prefect}"; SBD="$REPO/algorithms/sbd"; SRC="$SBD/native/sbd_mpi"

module load nvidia/25.9 cuda/12.6 cmake/3.31.1 2>/dev/null || true

echo "=== tool check ==="
which nvc++ mpic++ cmake 2>&1
cmake --version 2>/dev/null | head -1

# NVHPC_ROOT discovery: standard NVHPC SDK layout is <root>/compilers/bin/nvc++, so walking up
# two directories from `which nvc++` gives <root> regardless of the exact module version string
# (this is the one thing that differs from build_sbd_mpi_uhf.sh's hardcoded ROQUO path -- VERIFY
# this resolves to something sane on Miyabi-G before trusting the rest of this script; if `which
# nvc++` is empty, `module avail` under LNG/core for the exact loaded nvidia/cuda module names).
NVCXX=$(command -v nvc++ 2>/dev/null)
if [ -z "$NVCXX" ]; then
  echo "ERROR: nvc++ not found after 'module load nvidia/25.9 cuda/12.6' -- check module names with"
  echo "       'module avail' (LNG/core tree) and re-run with the correct versions loaded first."
  exit 1
fi
NV=$(cd "$(dirname "$NVCXX")/.." && pwd)
echo "NV(NVHPC root)=$NV"

echo "=== locate NCCL ==="
NCCL_ROOT=$(dirname "$(find "$NV" -name libnccl.so 2>/dev/null | head -1)" 2>/dev/null)
NCCL_ROOT=${NCCL_ROOT%/lib}
echo "NCCL_ROOT=$NCCL_ROOT"
find "$NV" -name "libnccl.so*" 2>/dev/null | head -2
find "$NV" -name "nccl.h" 2>/dev/null | head -1
if [ -z "$NCCL_ROOT" ]; then
  echo "ERROR: no libnccl.so found under \$NV=$NV -- Miyabi's nvidia/25.9 module may not bundle"
  echo "       NCCL the way ROQUO's nvhpc/26.5 does. Check 'module avail' for a separate nccl"
  echo "       module, or a system NCCL package (e.g. /usr/lib64/libnccl.so.2, as seen on ROQUO)."
  exit 1
fi

echo "=== cmake configure (nvhpc-thrust preset + multi-GPU + NCCL) ==="
cd "$SRC"
rm -rf build/nvhpc-thrust
NCCL_INC=$(dirname "$(find "$NV" -name nccl.h 2>/dev/null | head -1)")
CUBLAS_LIB=$(dirname "$(find "$NV" -name libcublas.so 2>/dev/null | head -1)")
echo "CUBLAS_LIB=$CUBLAS_LIB"
cmake --preset nvhpc-thrust \
  -DSBD_USE_NCCL=ON \
  -DSBD_USE_CUBLAS=ON \
  -DSBD_USE_RANK_DISTRIBUTION=ON \
  -DSBD_USE_BLOCK_RANK_DISTRIBUTION=ON \
  -DCMAKE_CXX_FLAGS="-D_UHF -DSBD_PREFECT -DSBD_NON_CUDA_AWARE_MPI -I${NCCL_INC}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${NCCL_ROOT}/lib -lnccl -L${CUBLAS_LIB} -lcublas -cudalib=cublas" \
  2>&1 | tail -25
echo "=== build tpb_diag ==="
cmake --build build/nvhpc-thrust --target tpb_diag -j 8 2>&1 | tail -25

BIN="$SRC/build/nvhpc-thrust/apps/chemistry_tpb_selected_basis_diagonalization/diag"
echo "=== result ==="
if [ -f "$BIN" ]; then
  cp "$BIN" "$SBD/native/diag-gpu_uhf-mpi-miyabi"
  echo "OK: copied to $SBD/native/diag-gpu_uhf-mpi-miyabi"
  ldd "$SBD/native/diag-gpu_uhf-mpi-miyabi" 2>/dev/null | grep -iE "nccl|mpi|cuda" | head
  ldd "$SBD/native/diag-gpu_uhf-mpi-miyabi" 2>/dev/null | grep "not found" && \
    echo "WARNING: unresolved shared libraries above -- LD_LIBRARY_PATH will need the matching entry at runtime"
else
  echo "BUILD FAILED: $BIN not found"
fi
echo "EXIT_build=$?"
