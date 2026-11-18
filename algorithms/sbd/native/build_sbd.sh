#!/usr/bin/env bash
set -euo pipefail

# Always operate in this script directory to avoid accidental cleanup in cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load modules (Miyabi defaults).
module load intel/2023.2.0
module load impi/2021.10.0

REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
SBD_DIR="${SBD_DIR:-${SCRIPT_DIR}/sbd}"
CXXFLAGS=("-std=c++17" "-axSAPPHIRERAPIDS,CORE-AVX512" "-qopenmp" "-O3")

# UHF=1 builds the unrestricted (open-shell) binary diag_uhf with -D_UHF; otherwise the
# restricted `diag` binary. Run twice to produce both.
if [ "${UHF:-0}" = "1" ]; then
    CXXFLAGS+=("-D_UHF")
    OUTBIN="diag_uhf"
else
    OUTBIN="diag"
fi

if command -v mpiicpc >/dev/null 2>&1; then
    CCCOM="mpiicpc"
elif command -v mpiicpx >/dev/null 2>&1; then
    CCCOM="mpiicpx"
else
    echo "Neither mpiicpc nor mpiicpx was found. Load Intel oneAPI modules first." >&2
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
"$CCCOM" "${CXXFLAGS[@]}" -c "$SCRIPT_DIR/main.cc" -o "$SCRIPT_DIR/main.o" -I"$SBD_DIR/include"
"$CCCOM" "${CXXFLAGS[@]}" -o "$SCRIPT_DIR/$OUTBIN" "$SCRIPT_DIR/main.o" -qmkl=parallel

echo "Build completed: $SCRIPT_DIR/$OUTBIN"
