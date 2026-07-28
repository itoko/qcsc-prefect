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
#
# This script then applies two idempotent source patches itself (both confirmed necessary on
# Miyabi-G; see native/README.md for the general description -- these are the exact fixes):
#   1. NCCL warm-up: sbdiag.h allocates a full-width device_vector for the warm-up all-reduce
#      (4 occurrences: h/b/t/a communicators) -- shrunk to width 1.
#   2. main.cc's SBD_PREFECT carryover.bin block references an undeclared `cobits` (a stale
#      variable name from before this fork split carryover into co_adet/co_bdet) -- rewritten to
#      write co_adet -> carryover.bin and, when running UHF (--bdetfile set), co_bdet ->
#      carryover_b.bin, using the same byte-packing logic the original block already had.
#
# Submit with: qsub build_sbd_mpi_uhf_miyabi.sh
set -uo pipefail
# NOTE: /home has a tiny quota on Miyabi (~50GB) -- clone/build under /work instead. There is no
# universal default here (it depends on your group), so MIYABI_REPO must be set explicitly.
REPO="${MIYABI_REPO:?set MIYABI_REPO to your qcsc-prefect checkout, e.g. /work/<group>/<user>/qcsc-prefect}"
SBD="$REPO/algorithms/sbd"; SRC="$SBD/native/sbd_mpi"

module load nvidia/25.9 cuda/12.6 cmake/3.31.1 2>/dev/null || true

echo "=== tool check ==="
which nvc++ mpic++ cmake 2>&1
cmake --version 2>/dev/null | head -1

# NVHPC_ROOT discovery: standard NVHPC SDK layout is <root>/compilers/bin/nvc++, so walking up
# TWO directories from `which nvc++` (bin -> compilers -> root) gives <root> regardless of the
# exact module version string.
NVCXX=$(command -v nvc++ 2>/dev/null)
if [ -z "$NVCXX" ]; then
  echo "ERROR: nvc++ not found after 'module load nvidia/25.9 cuda/12.6' -- check module names with"
  echo "       'module avail' (LNG/core tree) and re-run with the correct versions loaded first."
  exit 1
fi
NV=$(cd "$(dirname "$NVCXX")/../.." && pwd)
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

echo "=== apply source patches (idempotent) ==="
SBDIAG_H="$SRC/include/sbd/chemistry/tpb/sbdiag.h"
MAIN_CC="$SRC/apps/chemistry_tpb_selected_basis_diagonalization/main.cc"
if [ ! -f "$SBDIAG_H" ] || [ ! -f "$MAIN_CC" ]; then
  echo "ERROR: expected fork sources not found under $SRC -- did the git clone (see header"
  echo "       comment) actually complete?"
  exit 1
fi
# Patch 1: NCCL warm-up (4 occurrences: h/b/t/a communicators). Re-running is a harmless no-op
# once the pattern is gone.
sed -i 's/thrust::device_vector<double> A(W.size(), 0.0);/thrust::device_vector<double> A(1, 0.0);/g' \
  "$SBDIAG_H"
if grep -q 'A(W.size(), 0.0)' "$SBDIAG_H"; then
  echo "ERROR: NCCL warm-up pattern still present after sed -- sbdiag.h may not match what this"
  echo "       script expects; inspect it manually before continuing."
  exit 1
fi
echo "NCCL warm-up patch OK (0 occurrences of the full-width pattern remain)"

# Patch 2: undefined `cobits` in the SBD_PREFECT carryover.bin block -> co_adet/co_bdet.
python3 - "$MAIN_CC" << 'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
if "cobits" not in src:
    print("cobits patch: already applied (or not needed) -- skipping")
    sys.exit(0)
old = """    std::cout << "Number of carryover determinants: " << cobits.size() << std::endl;
    std::ofstream ofs_co_bin("carryover.bin", std::ios::binary);
    const size_t bytes_per_config = (L + 7) / 8;
    std::vector<uint8_t> bytes(bytes_per_config);
    for (size_t i = 0; i < cobits.size(); ++i) {
      std::fill(bytes.begin(), bytes.end(), 0);
      for (size_t j = 0; j < L; ++j) {
        size_t rev_idx = L - 1 - j;                 // sbd::makestring order
        size_t pw = rev_idx % sbd_data.bit_length;  // position in word
        size_t bw = rev_idx / sbd_data.bit_length;  // index of word
        bool bit = (cobits[i][bw] >> pw) & 1ULL;
        size_t pb = 7 - (j % 8);                    // big-endian bit order
        size_t bb = j / 8;                          // index of byte
        bytes[bb] |= static_cast<uint8_t>(bit << pb);
      }
      ofs_co_bin.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    ofs_co_bin.close();"""
new = """    auto write_carryover = [&](const std::string & filename,
                                const std::vector<std::vector<size_t>> & co_dets) {
      const size_t bytes_per_config = (L + 7) / 8;
      std::ofstream ofs(filename, std::ios::binary);
      std::vector<uint8_t> bytes(bytes_per_config);
      for (size_t i = 0; i < co_dets.size(); ++i) {
        std::fill(bytes.begin(), bytes.end(), 0);
        for (size_t j = 0; j < L; ++j) {
          size_t rev_idx = L - 1 - j;                 // sbd::makestring order
          size_t pw = rev_idx % sbd_data.bit_length;  // position in word
          size_t bw = rev_idx / sbd_data.bit_length;  // index of word
          bool bit = (co_dets[i][bw] >> pw) & 1ULL;
          size_t pb = 7 - (j % 8);                    // big-endian bit order
          size_t bb = j / 8;                          // index of byte
          bytes[bb] |= static_cast<uint8_t>(bit << pb);
        }
        ofs.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      }
      ofs.close();
    };
    std::cout << "Number of carryover determinants: " << co_adet.size() << std::endl;
    write_carryover("carryover.bin", co_adet);
    if( !bdetfile.empty() ) {
      std::cout << "Number of beta carryover determinants: " << co_bdet.size() << std::endl;
      write_carryover("carryover_b.bin", co_bdet);
    }"""
if old not in src:
    print("ERROR: cobits patch anchor not found -- main.cc may differ from what this script "
          "expects; inspect it manually before continuing.", file=sys.stderr)
    sys.exit(1)
n = src.count(old)
if n != 1:
    print(f"ERROR: expected exactly one match for the cobits block, found {n}", file=sys.stderr)
    sys.exit(1)
open(path, "w").write(src.replace(old, new))
print("cobits patch OK")
PYEOF
if [ "$?" != "0" ]; then
  echo "ERROR: cobits patch failed -- see output above."
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
  if ldd "$SBD/native/diag-gpu_uhf-mpi-miyabi" 2>/dev/null | grep -q "not found"; then
    echo "WARNING: unresolved shared libraries -- LD_LIBRARY_PATH will need the matching entry at runtime:"
    ldd "$SBD/native/diag-gpu_uhf-mpi-miyabi" 2>/dev/null | grep "not found"
  fi
  echo "EXIT_build=0"
else
  echo "BUILD FAILED: $BIN not found"
  echo "EXIT_build=1"
fi
