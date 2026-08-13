#!/bin/bash
#SBATCH --account=q0000219
#SBATCH --partition=roquo
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#
# CONTAINER-ISOLATED BUILD. Compiles entirely inside nvhpc-26.5-devel.sif (own bundled
# nvc++ 26.5 + hpcx-2.50 + CUDA 13.2, Ubuntu 24.04/glibc 2.39), completely bypassing
# ROQUO's shared module filesystem (/work/hps0/roquo/shared/modulefiles). This is the
# decisive test for whether ANY environment factor tied to the shared module tree (loader
# state, Lmod caching, filesystem metadata, something not yet identified) is responsible
# for the rank=0/1-on-MPI_Init_thread failure that every freshly-compiled binary has hit
# since 2026-08-02 -- the pre-Aug-2 module-content test (build_oldmodule_test.sh, job
# 31760->31762) already ruled out the one KNOWN module diff (UCX_MAX_RNDV_RAILS), so this
# goes further: zero shared-filesystem module involvement at all during compilation.
set -uo pipefail
REPO="$HOME/qcsc-prefect"; SBD="$REPO/algorithms/sbd"; SRC="$SBD/native/sbd_mpi"
SIF="$SBD/native/containers/nvhpc-26.5-devel.sif"
NV_C=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5

echo "=== container sanity ==="
apptainer exec "$SIF" bash -c 'uname -m; nvc++ --version | head -1; cmake --version | head -1'

echo "=== verify PR79 cherry-pick + CUDA-before-MPI_Init fix + GenerateExcitation fix are present ==="
cd "$SRC"
git log --oneline -3
grep -n "h_size == 1" include/sbd/chemistry/basic/davidson_thrust.h || {
  echo "ERROR: PR79 fix not found in davidson_thrust.h"; exit 1;
}
grep -n "cudaSetDevice(0)" apps/chemistry_tpb_selected_basis_diagonalization/main.cc || {
  echo "ERROR: CUDA-before-MPI_Init fix not found in main.cc"; exit 1;
}
grep -n "cr.reserve(2)" include/sbd/chemistry/tpb/helper.h || {
  echo "ERROR: GenerateExcitation cr/an hoist fix not found in helper.h"; exit 1;
}

echo "=== cmake configure + build, entirely inside the container ==="
rm -rf build/container-thrust
mkdir -p "$SBD/native/containers/tmp"
# TMPDIR override: SLURM sets TMPDIR=/scratch/<jobid>/tmp on the host, but /scratch is NOT
# bind-mounted into the container by default -- nvc++'s device-link step (acclnk) SIGSEGVs
# when it can't create its temp file there. Point it at a path under $HOME instead, which
# Apptainer bind-mounts automatically.
export APPTAINER_TMPDIR="$SBD/native/containers/tmp"
apptainer exec --pwd "$SRC" --env "NV_C=$NV_C,TMPDIR=$SBD/native/containers/tmp" "$SIF" bash -c '
  set -e
  export OMPI_CXX=nvc++
  export OMPI_CC=nvc
  export PATH="${NV_C}/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/bin:$PATH"
  command -v mpic++
  mpic++ -show
  NCCL_ROOT=$(dirname "$(find "${NV_C}/comm_libs" -name libnccl.so 2>/dev/null | head -1)")
  NCCL_ROOT=${NCCL_ROOT%/lib}
  NCCL_INC=$(dirname "$(find "${NV_C}" -name nccl.h 2>/dev/null | head -1)")
  CUBLAS_LIB=$(dirname "$(find "${NV_C}" -name libcublas.so 2>/dev/null | head -1)")
  echo NCCL_ROOT=$NCCL_ROOT
  echo CUBLAS_LIB=$CUBLAS_LIB
  echo TMPDIR=$TMPDIR
  cmake -S . -B build/container-thrust \
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
  cmake --build build/container-thrust --target tpb_diag -j 8 2>&1 | tail -30
'

BIN="$SRC/build/container-thrust/apps/chemistry_tpb_selected_basis_diagonalization/diag"
echo "=== result ==="
if [ -f "$BIN" ]; then
  cp "$BIN" "$SBD/native/diag-gpu_uhf-mpi-container"
  echo "OK: copied to $SBD/native/diag-gpu_uhf-mpi-container"
  echo "--- RUNPATH ---"
  readelf -d "$SBD/native/diag-gpu_uhf-mpi-container" | grep -i 'rpath\|runpath'
  echo "--- ldd (mpi/nccl/cuda libs) ---"
  ldd "$SBD/native/diag-gpu_uhf-mpi-container" 2>/dev/null | grep -iE "nccl|mpi|cuda|ucx|pmix"
else
  echo "BUILD FAILED: $BIN not found"
fi
