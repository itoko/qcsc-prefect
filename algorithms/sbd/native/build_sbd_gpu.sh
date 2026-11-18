#!/usr/bin/env bash
set -euo pipefail

# Always operate in this script directory to avoid accidental cleanup in cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
SBD_DIR="${SBD_DIR:-${SCRIPT_DIR}/sbd}"
CCCOM="${CCCOM:-mpic++}"
CCFLAGS="${CCFLAGS:--std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST}"
SYSLIB="${SYSLIB:--lblas -llapack}"

# UHF=1 builds the unrestricted GPU binary diag-gpu_uhf (adds -D_UHF on top of -DSBD_THRUST,
# which selects the _UHF path in integrals_thrust.h); otherwise the restricted diag-gpu.
if [ "${UHF:-0}" = "1" ]; then
    CCFLAGS="$CCFLAGS -D_UHF"
    OUTBIN="diag-gpu_uhf"
else
    OUTBIN="diag-gpu"
fi

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH. Load the Miyabi-G compiler/MPI environment first." >&2
    exit 1
fi

if [ ! -d "$SBD_DIR" ]; then
    echo "Cloning SBD repo..."
    git clone "$REPO_URL" "$SBD_DIR"
else
    echo "SBD repo already exists: $SBD_DIR"
fi

# Apply the opposite-spin (ab/ba) 2-RDM sign fix (idempotent). The kernel is shared by the RHF and
# UHF builds, and the RHF orbital optimization sums the ab/ba blocks into the spin-summed rdm2.txt.
"$SCRIPT_DIR/patches/apply_rdm_patches.sh" "$SBD_DIR" "$SCRIPT_DIR/patches/sbd-rdm-opposite-spin-sign.patch"

# Clean previous build artifacts for this target only.
rm -f "$SCRIPT_DIR"/*.o "$SCRIPT_DIR/$OUTBIN"

# Compile and link
echo "$CCCOM $CCFLAGS -c main.cc -o main.o -I$SBD_DIR/include"
$CCCOM $CCFLAGS -c main.cc -o main.o -I"$SBD_DIR/include"
echo "$CCCOM $CCFLAGS $SYSLIB -o $OUTBIN main.o"
$CCCOM $CCFLAGS $SYSLIB -o "$OUTBIN" main.o

echo "Build completed: $SCRIPT_DIR/$OUTBIN"
