"""ROQUO (GB200 GPU) Prefect hello demo for qcsc-prefect.

Confirms the Slurm+GPU block plumbing on RIKEN R-CCS ROQUO: creates the
`hpc-roquo` / `exec-*-slurm-gpu` blocks via the shared `create_blocks.py`
generator and runs a trivial in-allocation GPU command through them.
"""
