#!/usr/bin/env python3
"""Create the ROQUO GB200 GPU Prefect blocks for the hello demo.

Thin wrapper over the shared ``algorithms/sbd/create_blocks.py`` generator with
ROQUO defaults (Slurm target, GPU solver mode, one GPU). Produces:
  - ``cmd-roquo-gpu-hello``        (CommandBlock)
  - ``exec-sbd-slurm-gpu``         (ExecutionProfileBlock)
  - ``hpc-roquo``                  (HPCProfileBlock instance)

Environment:
  ROQUO_ACCOUNT   Slurm --account (required)
  ROQUO_PARTITION Slurm partition (default: roquo)
  ROQUO_GRES      Slurm --gres    (default: gpu:1)
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("QCSC_REPO", Path(__file__).resolve().parents[2]))
CREATE_BLOCKS = REPO / "algorithms" / "sbd" / "create_blocks.py"


def main() -> int:
    account = os.environ.get("ROQUO_ACCOUNT")
    if not account:
        sys.exit("Set ROQUO_ACCOUNT to your ROQUO Slurm account code.")
    partition = os.environ.get("ROQUO_PARTITION", "roquo")
    gres = os.environ.get("ROQUO_GRES", "gpu:1")

    cmd = [
        sys.executable, str(CREATE_BLOCKS),
        "--hpc-target", "slurm",
        "--solver-mode", "gpu",
        "--slurm-account", account,
        "--slurm-partition", partition,
        "--slurm-gres", gres,
        "--launcher", "srun",
        "--mpi-options", "--gpu-bind=closest",
        "--modules", "cuda/13.2", "nvhpc/26.5", "hpcx/2.50",
    ]
    print("Creating ROQUO GPU blocks via:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
