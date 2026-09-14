#!/usr/bin/env bash
set -euo pipefail

# Builds the sbd repo's bundled "chemistry_gdb_selected_basis_diagonalization" sample app
# (upstream sample code, added in sbd v1.2.0 -- NOT part of the tpb-namespace SQD binaries this
# repo's other build_sbd_*.sh scripts produce). Unlike diag/diag_uhf/diag-gpu*, this exposes
# sbd::gdb::diag()'s --carryover_type 2/3 heat-bath expansion (real HCI-style candidate
# screening via --heatbath_cutoff/--heatbath_truncation/--heatbath_batch_size), not just a
# Davidson diagonalizer. CPU-only (MPI+OpenMP) -- upstream ships no GPU/Thrust build path for
# this app's Makefile/CMakeLists.
#
# On ROQUO this needs to run INSIDE a compute-node allocation (hpcx/2.50 is unavailable on the
# login node -- "please run on a compute node"), e.g.:
#   srun --jobid=<id> --pty bash   # inside an existing allocation, or
#   salloc -N 1 -n 4 -t 00:30:00 -p roquo
# then:
#   module load hpcx/2.50 gcc/14
#   ./build_sbd_gdb_hci_roquo.sh
#
# One round of sbd::gdb::diag() is one Davidson-solve + one heat-bath-expansion step -- same
# single-round shape as this repo's own diag/diag_uhf. A multi-round HCI outer loop needs
# repeated invocations feeding each round's --carryoverfile back in as the next round's
# --detfiles, mirroring how run_recover.py already drives multi-step SQD recovery (see
# run_gdb_hci_recover.py in this same directory).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SBD_DIR="${SBD_DIR:-sbd}"
REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
APP_DIR="$SBD_DIR/apps/chemistry_gdb_selected_basis_diagonalization"

CCCOM="${CCCOM:-mpicxx}"
CCFLAGS="${CCFLAGS:--std=c++17 -fopenmp -O3}"
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
echo "(Renamed to avoid clashing with this repo's own diag/diag_uhf -- see run_gdb_hci_recover.py.)"
mv diag gdb_diag
echo "Renamed to $APP_DIR/gdb_diag"
