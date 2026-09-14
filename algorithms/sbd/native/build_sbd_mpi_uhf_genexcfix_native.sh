#!/bin/bash
#SBATCH --job-name=build-genexcfix
#SBATCH --account=qc-prj-other02
#SBATCH --reservation=large-20260824
#SBATCH --partition=roquo
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#
# NATIVE build of the GenerateExcitation-fixed UHF GPU binary (option C, 2026-08-28).
#
# WHY THIS EXISTS: the container-isolated build (build_sbd_mpi_uhf_container.sh) was the
# previously-validated home of the GenerateExcitation cr/an-hoist fix, but ROQUO's storage
# migration deleted the 45GB nvhpc-26.5-devel.sif image, and the resulting binary is NOT
# runnable outside the container -- it needs GLIBC_2.38 / GLIBCXX_3.4.32 which the host
# does not provide:
#   diag-gpu_uhf-mpi-container: /lib64/libc.so.6: version `GLIBC_2.38' not found
# Rather than re-pull 45GB, this builds the SAME fixed source natively, using the proven
# build_sbd_mpi_uhf_hpcx250.sh recipe (which produced working multi-node binaries July 31).
#
# TRADE-OFF being accepted knowingly: the container existed specifically to compile away from
# ROQUO's shared module filesystem, which is where the rank=0/1 MPI_Init_thread bug was
# suspected to originate. That isolation is given up here. Mitigation: the container build
# already PROVED the bug is unrelated to the shared module filesystem (it reproduced inside
# the container too), so native compilation is not expected to reintroduce it.
#
# MIGRATION FIX: NV is taken from the module (NVHPC_ROOT), NOT hardcoded to
# /opt/nvidia/hpc_sdk/Linux_aarch64/26.5 -- that literal path no longer resolves on the login
# node post-migration, and ROQUO support explicitly warned that stale absolute paths now fail
# only AFTER a job starts running.
set -uo pipefail
REPO="$HOME/qcsc-prefect"; SBD="$REPO/algorithms/sbd"; SRC="$SBD/native/sbd_mpi"

module purge
# nvhpc FIRST (nvc++/cuda/cmake), hpcx/2.50 SECOND so its mpic++ wins over nvhpc's bundled hpcx.
module load gcc/14 nvhpc/26.5
module load hpcx/2.50
export OMPI_CXX=nvc++
export OMPI_CC=nvc

# Resolve the NVHPC tree from the module rather than a hardcoded path (see MIGRATION FIX above).
NV="${NVHPC_ROOT:-}"
if [ -z "$NV" ] || [ ! -d "$NV" ]; then
  NV=$(dirname "$(dirname "$(readlink -f "$(which nvc++)")")")
fi
echo "NV=$NV"
[ -d "$NV" ] || { echo "FATAL: cannot resolve NVHPC root"; exit 1; }

echo "=== tool check (mpic++ must be the STANDALONE hpcx/2.50, invoking nvc++) ==="
which nvc++ mpic++ cmake 2>&1
cmake --version 2>/dev/null | head -1
readlink -f "$(which mpic++)"
mpic++ -show 2>&1

cd "$SRC"
echo "=== source state ==="
git log --oneline -3 2>&1 | head -3

echo "=== gate: all three required fixes must be present ==="
# PR79 lived on its own `pr79-fix` branch (bfbb075) and was NOT part of
# `fix/genexcitation-cr-an-hoist`; this gate caught that on the first attempt (job 120214).
# It was cherry-picked onto the fix branch (-> 76e4eaf). PR79 matters specifically for our
# configuration: it fixes a deadlock in GetTotalD_Thrust when h_comm has a SINGLE rank, and we
# always run h_comm_size = ranks/(adet*bdet*task) = 1. Do not build without it.
grep -q "h_size == 1" include/sbd/chemistry/basic/davidson_thrust.h || {
  echo "FATAL: PR79 fix missing in davidson_thrust.h -- cherry-pick bfbb075 from branch pr79-fix"; exit 1; }
echo "  PR79 (davidson_thrust h_size==1): present"

# NOTE: the older build script grepped for the literal "cudaSetDevice(0)", but this tree calls
# cudaSetDevice(myDevice) -- same fix (CUDA context established before MPI_Init), different
# argument. Grep the call, not one hardcoded argument, so a correct tree isn't rejected.
grep -q "cudaSetDevice(" apps/chemistry_tpb_selected_basis_diagonalization/main.cc || {
  echo "FATAL: CUDA-before-MPI_Init fix missing in main.cc"; exit 1; }
echo "  CUDA-before-MPI_Init (main.cc): present"

# THE fix this build exists for: cr/an hoisted out of the ia-loop in GenerateExcitation.
# Without it, NVHPC's managed-memory pool allocator lock livelocks under concurrent OMP
# threads (ROQUO Issue #73 root cause, found 2026-08-03). The container build gated on this
# too -- keep the gate so a silently-unfixed binary can never reach a 121-node run.
grep -q "cr.reserve(2)" include/sbd/chemistry/tpb/helper.h || {
  echo "FATAL: GenerateExcitation cr/an hoist fix missing in helper.h -- this build's whole purpose"; exit 1; }
echo "  GenerateExcitation cr/an hoist (helper.h): present"

echo "=== locate NCCL / cuBLAS under \$NV ==="
NCCL_ROOT=$(dirname "$(find "$NV/comm_libs" -name libnccl.so 2>/dev/null | head -1)" 2>/dev/null)
NCCL_ROOT=${NCCL_ROOT%/lib}
NCCL_INC=$(dirname "$(find "$NV" -name nccl.h 2>/dev/null | head -1)")
CUBLAS_LIB=$(dirname "$(find "$NV" -name libcublas.so 2>/dev/null | head -1)")
echo "NCCL_ROOT=$NCCL_ROOT"
echo "NCCL_INC=$NCCL_INC"
echo "CUBLAS_LIB=$CUBLAS_LIB"
[ -n "$NCCL_INC" ] && [ -n "$CUBLAS_LIB" ] || { echo "FATAL: NCCL/cuBLAS not found under $NV"; exit 1; }

echo "=== cmake configure (cc100 = GB200/Blackwell, matching the proven hpcx250 recipe) ==="
rm -rf build/genexcfix-native
cmake -S . -B build/genexcfix-native \
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
  2>&1 | tail -25

echo "=== build tpb_diag ==="
cmake --build build/genexcfix-native --target tpb_diag -j 8 2>&1 | tail -25

BIN="$SRC/build/genexcfix-native/apps/chemistry_tpb_selected_basis_diagonalization/diag"
OUTBIN="$SBD/native/diag-gpu_uhf-mpi-genexcfix-native"
echo "=== result ==="
if [ -f "$BIN" ]; then
  cp "$BIN" "$OUTBIN"
  echo "OK: $OUTBIN"
  echo "--- RUNPATH ---"
  readelf -d "$OUTBIN" | grep -i 'rpath\|runpath'
  echo "--- missing libs (MUST be empty -- this is what killed the container binary) ---"
  ldd "$OUTBIN" 2>&1 | grep -i "not found" || echo "  (none -- all libs resolve natively)"
  echo "--- key libs ---"
  ldd "$OUTBIN" 2>/dev/null | grep -iE "nccl|mpi|cuda|ucx|pmix" | head -10
  echo "BUILD_OK=1"
else
  echo "BUILD FAILED: $BIN not found"
  echo "BUILD_OK=0"
fi
