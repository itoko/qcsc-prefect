#!/usr/bin/env bash
# Build a FIXED unrestricted GPU solver binary (diag-gpu_uhf_fixed) from the patched sbd/ tree.
#
# This mirrors `build_sbd_gpu.sh UHF=1` (same compiler, flags, main.cc) but writes to a
# SEPARATE output name so it NEVER overwrites the production binaries:
#   - diag-gpu_uhf       (Aug27, production single-GPU)   <- untouched
#   - diag-gpu_uhf-mpi   (Sep15, production MPI)          <- untouched
#
# Before compiling it (idempotently) re-applies the opposite-spin 2-RDM sign fix to sbd/,
# so the resulting binary contains the corrected RDM kernel regardless of clone state.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SBD_DIR="${SBD_DIR:-${SCRIPT_DIR}/sbd}"
CCCOM="${CCCOM:-mpic++}"
CCFLAGS="${CCFLAGS:--std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST} -D_UHF"
SYSLIB="${SYSLIB:--lblas -llapack}"
OUTBIN="diag-gpu_uhf_fixed"

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH. Load nvhpc/26.5 + nvhpc-openmpi/26.5 first." >&2
    exit 1
fi

[ -d "$SBD_DIR" ] || { echo "SBD_DIR not found: $SBD_DIR (run build_sbd_gpu.sh once to clone it)"; exit 1; }

# Re-apply the RDM sign fix (idempotent). sbd/ is a re-clonable upstream checkout.
"$SCRIPT_DIR/patches/apply_rdm_patches.sh" "$SBD_DIR" "$SCRIPT_DIR/patches/sbd-rdm-opposite-spin-sign.patch"

# Clean only this target's artifacts.
rm -f "$SCRIPT_DIR"/main_fixed.o "$SCRIPT_DIR/$OUTBIN"

echo "$CCCOM $CCFLAGS -c main.cc -o main_fixed.o -I$SBD_DIR/include"
$CCCOM $CCFLAGS -c main.cc -o main_fixed.o -I"$SBD_DIR/include"
echo "$CCCOM $CCFLAGS $SYSLIB -o $OUTBIN main_fixed.o"
$CCCOM $CCFLAGS $SYSLIB -o "$OUTBIN" main_fixed.o
rm -f "$SCRIPT_DIR"/main_fixed.o

echo "Build completed: $SCRIPT_DIR/$OUTBIN"
