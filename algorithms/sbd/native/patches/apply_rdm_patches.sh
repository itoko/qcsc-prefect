#!/usr/bin/env bash
# Idempotently apply the opposite-spin 2-RDM sign-fix patches to a vendored sbd clone.
# The sbd/ and sbd_mpi/ trees are git clones of upstream (gitignored by qcsc-prefect and
# re-cloned by the build scripts), so the fix lives here as a patch and is re-applied after
# every clone. Safe to run repeatedly: skips a patch that is already applied.
#
# Usage:
#   apply_rdm_patches.sh <SBD_DIR> <patch-file>
# Example (GPU tree):
#   apply_rdm_patches.sh "$SCRIPT_DIR/sbd" "$SCRIPT_DIR/patches/sbd-rdm-opposite-spin-sign.patch"
set -euo pipefail

SBD_DIR="${1:?usage: apply_rdm_patches.sh <SBD_DIR> <patch-file>}"
PATCH="${2:?usage: apply_rdm_patches.sh <SBD_DIR> <patch-file>}"

[ -d "$SBD_DIR" ] || { echo "[patch] SBD_DIR not found: $SBD_DIR" >&2; exit 1; }
[ -f "$PATCH" ]   || { echo "[patch] patch not found: $PATCH" >&2; exit 1; }

# Resolve to absolute paths: `git -C "$SBD_DIR"` changes git's cwd, so a relative
# patch path would otherwise be looked up inside SBD_DIR and not found.
SBD_DIR="$(cd "$SBD_DIR" && pwd)"
PATCH="$(cd "$(dirname "$PATCH")" && pwd)/$(basename "$PATCH")"

if git -C "$SBD_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "[patch] already applied: $(basename "$PATCH") -> $SBD_DIR"
elif git -C "$SBD_DIR" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SBD_DIR" apply "$PATCH"
  echo "[patch] applied: $(basename "$PATCH") -> $SBD_DIR"
else
  echo "[patch] ERROR: $(basename "$PATCH") does not apply cleanly to $SBD_DIR" >&2
  echo "[patch] upstream HEAD may have moved; regenerate the patch (see patches/README.md)." >&2
  exit 1
fi
