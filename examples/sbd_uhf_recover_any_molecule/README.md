# Generic RHF/UHF/BS-UHF SQD Recovery

`run_recover.py` is a molecule-agnostic template for the "sample once, then re-diagonalize
offline for many recovery steps" pattern used throughout this repo's Fe2S2/Fe4S4 work — with no
Fe-specific defaults. Point it at any FCIDUMP (and, for broken-symmetry UHF, your own AF-groups
JSON) and a persisted sample pool, and it runs a checkpointed, multi-step SQD recovery on Fugaku,
Slurm (e.g. ROQUO), or locally.

See [`docs/tutorials/run_uhf_bsuhf_any_molecule.md`](../../docs/tutorials/run_uhf_bsuhf_any_molecule.md)
for the full walkthrough (getting an FCIDUMP, building the AF-groups spec, building the solver
binaries) and [`docs/reference/hpc_resource_sizing.md`](../../docs/reference/hpc_resource_sizing.md)
for how to choose `--nodes/--ranks-per-node/--omp-threads/--adet/--bdet`.

[`walkthrough.ipynb`](walkthrough.ipynb) demonstrates the same pipeline end-to-end on a small
molecule with no HPC required, before you scale up with this script.

## Prerequisites

```bash
cd /path/to/qcsc-prefect
uv sync
cd algorithms/sbd && uv sync          # installs the `sbd` package into the active venv
```

Then build the native solver for your platform/method — see
[`algorithms/sbd/native/README.md`](../../algorithms/sbd/native/README.md). You need the `_uhf`
binary variant for both plain UHF and BS-UHF FCIDUMPs.

You also need a persisted sample pool (a `raw_samples_*.npz`) from a prior `quantum_source=
"real-device"` (or `"random"`, for a dry run) sampling run — this script only re-diagonalizes,
it does not sample.

## Quick correctness check (no HPC, no real pool)

```bash
python run_recover.py \
  --fcidump /path/to/your.fcidump --method uhf \
  --pool /path/to/any_pool.npz --quantum-source random \
  --sqd-dim 2000 --recovery-steps 1 --n-batches 1 \
  --run-dir /tmp/sbd_recover_test \
  --hpc-target local --solver-mode cpu \
  --nodes 1 --ranks-per-node 1 --omp-threads 1 --adet 1 --bdet 1 \
  --sbd-executable /path/to/native/diag --sbd-executable-uhf /path/to/native/diag_uhf
```

`--quantum-source random` ignores `--pool`'s contents (deterministic random bitstrings instead) —
useful to confirm the FCIDUMP, AF-groups spec, and block/flow plumbing all work before you touch a
real pool or a multi-node allocation. `--pool` is still required by the CLI even though its
contents aren't read in this mode.

## Fugaku example (deep recovery, checkpointed)

```bash
python run_recover.py \
  --fcidump /path/to/your.fcidump --method uhf \
  --af-groups '{"metal_a":[2,3,4,5,6],"metal_b":[7,8,9,10,11],"up":["metal_a"],"down":["metal_b"]}' \
  --pool file:///path/to/raw_samples.npz \
  --sqd-dim 1000000000 --recovery-steps 50 --n-batches 1 \
  --ckpt-dir /path/to/ckpt --run-dir /path/to/run \
  --hpc-target fugaku --group ra000000 --queue large \
  --nodes 2304 --ranks-per-node 1 --omp-threads 48 --adet 48 --bdet 48 \
  --sbd-executable /path/to/native/diag --sbd-executable-uhf /path/to/native/diag_uhf
```

Re-running the exact same command (same `--ckpt-dir`) after a wall-time kill resumes automatically
from the last completed step — look for `[ckpt] RESUME from ... completed N step(s)` in the log.

## ROQUO example (in-allocation GPU, single node / 4 GPUs)

```bash
python run_recover.py \
  --fcidump /path/to/your.fcidump --method uhf --af-groups @af_groups.json \
  --pool file:///path/to/raw_samples.npz \
  --sqd-dim 300000000 --recovery-steps 10 --n-batches 4 \
  --ckpt-dir /path/to/ckpt --run-dir /path/to/run \
  --hpc-target local --solver-mode gpu \
  --nodes 1 --ranks-per-node 4 --omp-threads 1 --adet 1 --bdet 4 \
  --launcher srun --mpi-options --gpu-bind=closest \
  --sbd-executable /path/to/native/diag-gpu --sbd-executable-uhf /path/to/native/diag-gpu_uhf
```

Wrap this in an `sbatch --gres=gpu:4 --ntasks-per-node=4 ...` submission — `--hpc-target local`
runs every Davidson solve in-allocation (no nested `sbatch`), which is the fast path on ROQUO.

## Flag reference

| Group | Flag | Notes |
| --- | --- | --- |
| molecule | `--fcidump` | required |
| | `--method {rhf,uhf}` | default `uhf` |
| | `--af-groups` | raw JSON, `@path/to/file.json`, or the `fe4s4` convenience keyword. Omit for plain UHF/RHF |
| | `--af-pol`, `--af-free-s` | fractional-polarization / free-bridging-fragment knobs, see the tutorial |
| SQD | `--pool` | one or more sample-pool paths/URIs, required |
| | `--sqd-dim`, `--recovery-steps`, `--n-batches` | subspace size, recovery passes, K-batches |
| | `--quantum-source {saved,random}` | `saved` is the production path |
| checkpoint | `--ckpt-dir` | omit to disable checkpoint/resume |
| HPC | `--hpc-target {fugaku,slurm,local}` | `slurm`/`local` cover ROQUO (`local` = in-allocation, fast) |
| | `--nodes`, `--ranks-per-node`, `--omp-threads`, `--cores-per-node` | validated against each other before anything is submitted |
| | `--adet`, `--bdet`, `--task-comm-size` | must multiply to `nodes × ranks-per-node` |
| | `--launcher`, `--mpi-options` | e.g. `mpiexec` (Fugaku), `srun`/`mpirun` (ROQUO) |
| | `--queue`, `--group`, `--fugaku-gfscache`, `--fugaku-mpi-options-for-pjm` | Fugaku-only |
| | `--slurm-account`, `--slurm-partition`, `--slurm-gres` | Slurm-only (ROQUO) |
| solver | `--sbd-executable`, `--sbd-executable-uhf` | paths to the built binaries |
| | `--block`, `--iteration`, `--carryover-ratio`, `--carryover-type`, `--solver-timeout-seconds` | Davidson solver knobs, defaults match the production Fe4S4 launchers |
| misc | `--run-dir` | required; holds `work/`, `prefect_home/`, `result.json` |

Run `python run_recover.py --help` for the exact, always-current flag list.
