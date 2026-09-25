#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
SBD_DIR="${SBD_DIR:-${SCRIPT_DIR}/sbd}"
CCCOM="${CCCOM:-mpic++}"
CCFLAGS="${CCFLAGS:--std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST -D_UHF}"
SYSLIB="${SYSLIB:--lblas -llapack}"
OUTBIN="diag-gpu_uhf_diag"
[ -d "$SBD_DIR" ] || { echo "missing sbd/ at $SBD_DIR" >&2; exit 1; }
# Re-apply the opposite-spin 2-RDM sign fix (sbd/ is a re-clonable upstream checkout).
"$SCRIPT_DIR/patches/apply_rdm_patches.sh" "$SBD_DIR" "$SCRIPT_DIR/patches/sbd-rdm-opposite-spin-sign.patch"
# Clean ONLY this diagnostic target; production *.o and diag-gpu_uhf* are untouched.
rm -f "$SCRIPT_DIR/main_rdmcheck.o" "$SCRIPT_DIR/$OUTBIN"
echo "$CCCOM $CCFLAGS -c main_rdmcheck.cc -o main_rdmcheck.o -I$SBD_DIR/include"
$CCCOM $CCFLAGS -c main_rdmcheck.cc -o main_rdmcheck.o -I"$SBD_DIR/include"
echo "$CCCOM $CCFLAGS $SYSLIB -o $OUTBIN main_rdmcheck.o"
$CCCOM $CCFLAGS $SYSLIB -o "$OUTBIN" main_rdmcheck.o
echo "Build completed: $SCRIPT_DIR/$OUTBIN"
