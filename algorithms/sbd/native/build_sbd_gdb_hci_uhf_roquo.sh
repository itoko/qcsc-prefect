#!/usr/bin/env bash
set -euo pipefail

# UHF (independent alpha/beta spatial orbitals) variant of build_sbd_gdb_hci_roquo.sh. Same
# upstream "chemistry_gdb_selected_basis_diagonalization" sample app, compiled with -D_UHF so
# oneInt/twoInt (include/sbd/chemistry/basic/integrals.h) switch to their spin-resolved storage
# layout -- see docs/reference/sbd_gdb_heatbath_and_selection_gap.md and the
# examples/fe4s4_hci_from_bsuhf_reference/ scripts that produce the matching FCIDUMP/detfile
# inputs this binary needs.
#
# _UHF is a compile-time macro (not a runtime flag), so this produces a SEPARATE binary
# (gdb_diag_uhf) from the plain gdb_diag build_sbd_gdb_hci_roquo.sh produces.
#
# On ROQUO this needs to run INSIDE a compute-node allocation (hpcx/2.50 is unavailable on the
# login node -- "please run on a compute node"), e.g.:
#   srun --jobid=<id> --pty bash   # inside an existing allocation, or
#   salloc -N 1 -n 4 -t 00:30:00 -p roquo
# then:
#   module load hpcx/2.50 gcc/14
#   ./build_sbd_gdb_hci_uhf_roquo.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SBD_DIR="${SBD_DIR:-sbd}"
REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
APP_DIR="$SBD_DIR/apps/chemistry_gdb_selected_basis_diagonalization"

CCCOM="${CCCOM:-mpicxx}"
CCFLAGS="${CCFLAGS:--std=c++17 -fopenmp -O3 -D_UHF}"
SYSLIB="${SYSLIB:--llapack -lblas -fopenmp}"

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH. Load an MPI module first (e.g. \`module load" >&2
    echo "hpcx/2.50 gcc/14\` on ROQUO) -- this must run inside a compute-node allocation." >&2
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
rm -f ./*.o ./diag

echo "$CCCOM $CCFLAGS -c main.cc -o main.o -I../../include"
$CCCOM $CCFLAGS -c main.cc -o main.o -I../../include
echo "$CCCOM $CCFLAGS $SYSLIB -o diag main.o"
$CCCOM $CCFLAGS $SYSLIB -o diag main.o

echo "Build completed: $APP_DIR/diag"
mv diag gdb_diag_uhf
echo "Renamed to $APP_DIR/gdb_diag_uhf (to coexist with the plain gdb_diag build)."
