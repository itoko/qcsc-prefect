#!/bin/bash
#SBATCH --job-name=rebuild-venv
#SBATCH --account=qc-prj-other02
#SBATCH --reservation=large-20260824
#SBATCH --partition=roquo
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#
# Rebuild the sbd venv after ROQUO's storage migration (2026-08-28).
#
# WHY: the migration moved $HOME from /home/q0000219 -> /home/nfs1/q0000219, and a venv bakes
# absolute paths in three places, ALL of which broke:
#   1. 46 of 54 console scripts in .venv/bin/ have a hardcoded shebang:
#        #!/home/q0000219/qcsc-prefect/algorithms/sbd/.venv/bin/python
#      -> "bad interpreter: No such file or directory". This is why `prefect` appeared to be
#      "not found" (execvp reports ENOENT for a bad INTERPRETER, which reads as a missing exe).
#   2. .venv/bin/python symlinked -> /usr/bin/python3.12 (fine on compute, ABSENT on login).
#   3. editable-install .pth / _finder.py files point into the old /home/q0000219 tree, so
#      qcsc_prefect_* and sbd all fail to import.
# None of these are fixable with env vars, hence a real rebuild.
#
# MUST RUN ON A COMPUTE NODE: /usr/bin/python3.12 exists only there post-migration (login has 3.9).
#
# uv.lock survived the migration, so `uv sync` reproduces the EXACT prior dependency set --
# this is a faithful rebuild, not a re-resolve that could drift versions under us hours before
# a production run.
set -uo pipefail
REPO="$HOME/qcsc-prefect"; SBD="$REPO/algorithms/sbd"
UV="$HOME/.local/bin_arm/uv"    # the aarch64 build; ~/.local/bin/uv is x86_64

echo "=== preflight ==="
echo "HOME=$HOME"
"$UV" --version || { echo "FATAL: aarch64 uv not runnable"; exit 1; }
ls /usr/bin/python3.12 || { echo "FATAL: python3.12 missing -- are we on a compute node?"; exit 1; }
ls "$SBD/uv.lock" || { echo "FATAL: uv.lock missing -- cannot do a faithful rebuild"; exit 1; }

# Keep the old venv aside rather than deleting it: if the rebuild fails we still have the
# (broken-shebang but otherwise complete) tree to inspect, and nothing else is lost.
if [ -d "$SBD/.venv" ]; then
  BK="$SBD/.venv.broken_migration_$(date +%Y%m%d_%H%M%S)"
  echo "=== moving old venv aside -> $BK ==="
  mv "$SBD/.venv" "$BK" || { echo "FATAL: could not move old venv"; exit 1; }
fi

cd "$SBD"
export UV_CACHE_DIR="/tmp/uvcache_$USER"
export TMPDIR="/tmp/uvtmp_$USER"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

echo "=== uv sync (frozen: honor uv.lock exactly, no re-resolution) ==="
"$UV" sync --frozen --python /usr/bin/python3.12 2>&1 | tail -30
RC=$?
echo "uv_sync_rc=$RC"

echo "=== verify: the three things that were broken ==="
FAIL=0

echo "--- 1. shebangs point at the NEW home ---"
BAD=$(grep -l '/home/q0000219' "$SBD/.venv/bin/"* 2>/dev/null | wc -l)
echo "scripts still referencing the dead path: $BAD (must be 0)"
[ "$BAD" -eq 0 ] || FAIL=1

echo "--- 2. bare prefect CLI actually runs ---"
if "$SBD/.venv/bin/prefect" --version 2>&1 | head -2; then :; else echo "  prefect FAILED"; FAIL=1; fi

echo "--- 3. imports resolve from a NEUTRAL cwd (not \$SBD, which masks 'sbd' via ./sbd) ---"
( cd /tmp && "$SBD/.venv/bin/python" -c \
  "import qcsc_prefect_blocks, qcsc_prefect_adapters, qcsc_prefect_core, qcsc_prefect_executor, sbd; print('  all five imports OK')" ) || FAIL=1

echo "--- 4. create_blocks.py loads (the step that actually died 4x tonight) ---"
( cd "$SBD" && "$SBD/.venv/bin/python" "$SBD/create_blocks.py" --help >/dev/null 2>&1 && echo "  create_blocks OK" ) || { echo "  create_blocks FAILED"; FAIL=1; }

if [ "$FAIL" -eq 0 ]; then
  echo "VENV_REBUILD_OK=1"
else
  echo "VENV_REBUILD_OK=0"
fi
