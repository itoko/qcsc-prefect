"""Generic RHF/UHF/BS-UHF multi-step SQD recovery template for qcsc-prefect.

Molecule-agnostic wrapper around `create_blocks.py` + `sbd.main.riken_sqd_de`: point it at any
FCIDUMP (and, for broken-symmetry UHF, an AF-groups JSON spec) and a persisted sample pool, and it
runs a deep, checkpointed configuration-recovery sweep on Fugaku, Slurm (e.g. ROQUO), or locally.
See README.md for usage and docs/tutorials/run_uhf_bsuhf_any_molecule.md for the full tutorial.
"""
