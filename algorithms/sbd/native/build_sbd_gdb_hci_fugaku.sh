#!/usr/bin/env bash
set -euo pipefail

# Builds the sbd repo's bundled "chemistry_gdb_selected_basis_diagonalization" sample app on
# Fugaku (A64FX, Fujitsu compiler suite) -- same upstream sample as build_sbd_gdb_hci_roquo.sh,
# just targeting Fugaku's mpiFCCpx toolchain instead of ROQUO's mpicxx/hpcx. This exposes
# sbd::gdb::diag()'s --carryover_type 2/3 heat-bath expansion (real HCI-style candidate
# screening via --heatbath_cutoff/--heatbath_truncation/--heatbath_batch_size), not just a
# Davidson diagonalizer -- see docs/reference/sbd_gdb_heatbath_and_selection_gap.md.
#
# CPU-only (MPI+OpenMP) -- upstream ships no GPU/Thrust build path for this app's
# Makefile/CMakeLists, unlike this repo's own tpb-namespace GPU binaries.
#
# mpiFCCpx is on PATH by default on Fugaku's compute/login environment for this account (no
# module load needed -- confirmed via `which mpiFCCpx` -> /opt/FJSVxtclanga/tcsds-1.2.43/bin/).
# Uses the app's own Makefile + Configuration file (unlike build_sbd_gdb_hci_roquo.sh, which
# compiles main.cc directly) -- the Configuration file already ships commented-out Fugaku
# example flags matching this repo's own build_sbd_fugaku.sh toolchain exactly.
#
# Usage:
#   ./build_sbd_gdb_hci_fugaku.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SBD_DIR="${SBD_DIR:-sbd}"
REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
APP_DIR="$SBD_DIR/apps/chemistry_gdb_selected_basis_diagonalization"

CCCOM="${CCCOM:-mpiFCCpx}"
CCFLAGS="${CCFLAGS:--Nclang -std=c++17 -stdlib=libc++ -Kfast,openmp -Xpreprocessor -fopenmp}"
SYSLIB="${SYSLIB:--SSL2}"

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH." >&2
    exit 1
fi

if [ ! -d "$SBD_DIR" ]; then
    echo "Cloning SBD repo..."
    git clone "$REPO_URL" "$SBD_DIR"
else
    echo "SBD repo already exists."
fi

if [ ! -d "$APP_DIR" ]; then
    echo "Expected $APP_DIR not found in the cloned sbd repo -- check REPO_URL/SBD_DIR." >&2
    exit 1
fi

cd "$APP_DIR"

# Write Configuration with Fugaku's proven flags (the shipped file's own Fugaku example block,
# uncommented) rather than editing in place -- keeps this script idempotent/re-runnable.
cat > Configuration <<CONF_EOF
SBD_PATH=../..
CCCOM=$CCCOM
CCFLAGS= $CCFLAGS
SYSLIB= $SYSLIB
CONF_EOF

echo "Wrote Configuration:"
cat Configuration

make clean
make

if [ ! -f diag ]; then
    echo "Build did not produce ./diag -- check make output above." >&2
    exit 1
fi

mv diag gdb_diag
echo "Build completed: $APP_DIR/gdb_diag"
echo "(Renamed to avoid clashing with this repo's own diag/diag_uhf -- see run_gdb_hci_recover.py.)"
