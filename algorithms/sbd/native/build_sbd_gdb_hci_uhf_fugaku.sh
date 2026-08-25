#!/usr/bin/env bash
set -euo pipefail

# UHF (independent alpha/beta spatial orbitals) variant of build_sbd_gdb_hci_fugaku.sh. Same
# upstream "chemistry_gdb_selected_basis_diagonalization" sample app, compiled with -D_UHF so
# oneInt/twoInt (include/sbd/chemistry/basic/integrals.h) switch to their spin-resolved storage
# layout -- see docs/reference/sbd_gdb_heatbath_and_selection_gap.md and the
# examples/fe4s4_hci_from_bsuhf_reference/ scripts that produce the matching FCIDUMP/detfile
# inputs this binary needs.
#
# _UHF is a compile-time macro (not a runtime flag), so this produces a SEPARATE binary
# (gdb_diag_uhf) from the plain gdb_diag build_sbd_gdb_hci_fugaku.sh produces -- one process can
# only ever be one variant. Run the plain build first if you want both binaries side by side in
# the same $APP_DIR (make clean only removes main.o, so this ordering is safe); or point SBD_DIR
# at separate clones to avoid any risk of stepping on a half-renamed intermediate `diag`.
#
# Usage:
#   ./build_sbd_gdb_hci_uhf_fugaku.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SBD_DIR="${SBD_DIR:-sbd}"
REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
APP_DIR="$SBD_DIR/apps/chemistry_gdb_selected_basis_diagonalization"

CCCOM="${CCCOM:-mpiFCCpx}"
CCFLAGS="${CCFLAGS:--Nclang -std=c++17 -stdlib=libc++ -Kfast,openmp -Xpreprocessor -fopenmp -D_UHF}"
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

mv diag gdb_diag_uhf
echo "Build completed: $APP_DIR/gdb_diag_uhf"
echo "(Renamed to avoid clashing with the plain gdb_diag / this repo's own diag/diag_uhf.)"
