# Create Your QCSC Workflow with Prefect for ROQUO (GB200 GPU)

This hands-on tutorial guides you through running a QCSC (quantum-classical
self-consistent) SBD workflow on **ROQUO**, the RIKEN R-CCS GB200 (Grace-Hopper
successor) system with NVIDIA Blackwell GPUs on Arm Neoverse-V2 hosts. It mirrors
the [Fugaku](create_qcsc_workflow_for_fugaku.md),
[Miyabi](create_qcsc_workflow_for_miyabi.md), and
[local Slurm](create_qcsc_workflow_for_local_slurm.md) tutorials, but targets the
**GPU** SBD solver (`diag-gpu` / `diag-gpu_uhf`) inside a Slurm allocation.

Our objective is to run the SBD selected-CI eigensolver on a GPU, driven by the
same `qcsc-prefect` Prefect blocks used on the CPU systems, and — for the
iron-sulfur workloads — to run a full **recovery / density-estimation closed
loop** (optionally with orbital optimization) against a persisted sample pool.

Key principles in this tutorial:

- Users do not write new Python flow code — the SBD sweep entrypoints are reused
- The GPU solver runs **in-allocation** (`--hpc-target local`), so there are no
  nested `sbatch` submissions per Davidson solve (this is the ROQUO fast path)
- ROQUO is Arm (aarch64); the GPU solver is built with the NVIDIA HPC SDK
  (`nvc++` / CUDA), and MPI comes from the bundled HPC-X
- One GPU per solver rank is the default; multi-GPU uses NCCL over `srun`

---

## Prefect Core Concepts

You will see these terms (same model as the other tutorials):

- **Flow**: the end-to-end workflow entrypoint
  - hello demo: `examples/roquo_gpu_prefect_hello_demo/flow.py`
  - SBD sweep: `algorithms/sbd/sweep/run_fe4s4_gpu_roquo.py`
- **Task**: individual units executed inside a flow (quantum-sampling task,
  the `hpc-*` solver task)
- **Block**: reusable server-side configuration stored in Prefect
  - `CommandBlock`, `ExecutionProfileBlock`, `HPCProfileBlock`
- **Variable**: server-side runtime parameters (e.g. the Slurm option string)

The SBD sweep uses `algorithms/sbd/create_blocks.py` to generate the blocks and
the Slurm submission asset from environment variables, so you rarely create
blocks by hand.

## What you need

- An account on ROQUO with a valid project/account code (`--account=<code>`)
- Access to the `roquo` Slurm partition and at least one GPU (`--gres=gpu:1`)
- A checkout of `qcsc-prefect` in your home directory
- A reachable Prefect backend (Prefect Cloud, or the ephemeral local server the
  sweep starts for you)
- The Arm GPU SBD binary built (see **Step 3**)

## Prerequisites

ROQUO login nodes do **not** carry the compute-node toolchain. Build and run on
a **compute node** (via `sbatch` / `srun`); the login node is only for editing,
`git`, and job submission.

```bash
# On a ROQUO login node
module avail                # confirm: cuda/13.x, nvhpc/26.5, hpcx/2.50, gcc/14
sinfo -h -o "%P %a %D"      # confirm the 'roquo' partition and node count
```

## Existing files used in this tutorial

| Purpose | Path |
| --- | --- |
| Hello demo flow | `examples/roquo_gpu_prefect_hello_demo/flow.py` |
| Hello demo block creator | `examples/roquo_gpu_prefect_hello_demo/create_blocks.py` |
| Hello demo GPU probe | `examples/roquo_gpu_prefect_hello_demo/hello_gpu.sh` |
| SBD GPU sweep driver (Python) | `algorithms/sbd/sweep/run_fe4s4_gpu_roquo.py` † |
| SBD GPU sweep wrapper (sbatch) | `algorithms/sbd/sweep/run_fe4s4_gpu_roquo.sh` † |
| Block generator | `algorithms/sbd/create_blocks.py` |

† `algorithms/sbd/sweep/` is experiment scratch (git-ignored, not shipped). The
GPU sweep driver/wrapper are project-specific launchers built on the shipped
`create_blocks.py` + `run_job_from_blocks`; Steps 5–5.1 give the exact `sbatch`
invocation, environment variables, and the `create_blocks.py` call they wrap, so
you can reproduce the flow in your own `sweep/` directory.

---

## Run the QCSC GPU Workflow on ROQUO

### Step 1. Log in and reach a Prefect backend

Either export your Prefect Cloud credentials, or let the SBD sweep launch an
**ephemeral local Prefect server** for you (it does this automatically and tags a
per-job `PREFECT_HOME` so concurrent jobs don't collide):

```bash
# Optional — only if using Prefect Cloud
uv run prefect cloud login
```

### Step 2. Install dependencies

The project targets Python 3.12 via `uv`. On a ROQUO login node:

```bash
cd ~/qcsc-prefect
uv sync
cd algorithms/sbd
uv sync                      # builds the sbd venv used by the GPU wrapper
```

### Step 3. Build the GPU SBD solver (aarch64 + NVIDIA HPC SDK)

The GPU eigensolver (`diag-gpu` for RHF, `diag-gpu_uhf` for UHF/broken-symmetry)
is built with `nvc++` + CUDA from the NVIDIA HPC SDK. Build it on a compute node:

```bash
srun --account=<code> --partition=roquo --gres=gpu:1 --time=01:00:00 --pty bash
module load cuda/13.2 nvhpc/26.5
cd ~/qcsc-prefect/algorithms/sbd/native/sbd_mpi
# CMake with the GPU/thrust backend enabled (RHF binary: diag-gpu)
cmake -S . -B build-gpu -DSBD_THRUST=ON -DCMAKE_CXX_COMPILER=nvc++
cmake --build build-gpu -j --target diag-gpu
# UHF binary (broken-symmetry FCIDUMP): rebuild with -D_UHF -> diag-gpu_uhf
cmake -S . -B build-gpu-uhf -DSBD_THRUST=ON -D_UHF=ON -DCMAKE_CXX_COMPILER=nvc++
cmake --build build-gpu-uhf -j --target diag-gpu_uhf
```

> **Why two binaries?** The RHF and UHF solvers differ in the FCIDUMP layout they
> expect. Running the RHF `diag-gpu` on a UHF (broken-symmetry) FCIDUMP crashes;
> build `diag-gpu_uhf` with `-D_UHF` for the unrestricted path.

### Step 4. Run the GPU hello demo (smoke test)

This confirms the block plumbing and that a GPU is visible inside the allocation
before you run the full solver:

```bash
cd ~/qcsc-prefect
export ROQUO_ACCOUNT=<code>
uv run python examples/roquo_gpu_prefect_hello_demo/create_blocks.py
sbatch --account=<code> --partition=roquo --gres=gpu:1 --time=00:10:00 \
       examples/roquo_gpu_prefect_hello_demo/hello_gpu.sh
```

`hello_gpu.sh` prints `nvidia-smi` and runs the flow, which submits a trivial
in-allocation GPU command through the `hpc-roquo` block. A successful run reports
`exit_status: 0` and the GPU name (e.g. `GB200`).

### Step 5. Run the SBD GPU recovery sweep (iron-sulfur)

The SBD sweep driver reads a persisted **sample pool** (an `.npz` of raw quantum
samples) and runs the recovery / density-estimation loop on the GPU solver. The
wrapper sets the ROQUO environment (HPC-X MPI, NVIDIA runtime, in-allocation
solver) for you:

```bash
cd ~/qcsc-prefect/algorithms/sbd/sweep
# RHF recovery, 1 GPU, from a persisted pool
METHOD=rhf \
FE4S4_POOL=/path/to/raw_samples.npz \
FE4S4_SQD_DIM=300000000 \
FE4S4_RECSTEPS=5 FE4S4_NBATCH=5 \
sbatch --account=<code> run_fe4s4_gpu_roquo.sh
```

Watch it:

```bash
squeue -u $USER
tail -f fe4s4-gpu.<jobid>.out      # look for the per-step energies + EXIT_fe4s4_gpu=0
```

### Step 5.1. What the wrapper does

`run_fe4s4_gpu_roquo.sh` → `run_fe4s4_gpu_roquo.py` → `create_blocks.py`:

- Loads `cuda/13.2 nvhpc/26.5 hpcx/2.50` and puts `nvc++` + the CUDA runtime on
  `PATH`/`LD_LIBRARY_PATH` so `diag-gpu` finds its libs
- Sets `SBD_LAUNCHER=srun` and `SBD_MPI_OPTIONS=--gpu-bind=closest` so each rank
  binds to its nearest GPU
- Runs the solver **in-allocation** (`--hpc-target local`): every Davidson solve
  is a direct `srun` subprocess in the orchestrator's own allocation — **no
  nested `sbatch`**, no per-solve queue wait (this is the key ROQUO speed-up)
- Starts a per-job ephemeral Prefect server (tagged by `SLURM_JOB_ID`) so
  concurrent jobs never share a database

### Step 6. Environment variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `METHOD` | `rhf` (→ `diag-gpu`) or `uhf` (→ `diag-gpu_uhf`) | `rhf` |
| `FE4S4_POOL` | persisted `raw_samples` npz path(s) — **required** | — |
| `FE4S4_SQD_DIM` | selected-CI subspace dimension | `3e8` |
| `FE4S4_RECSTEPS` | recovery passes | `5` |
| `FE4S4_NBATCH` | K-batches per recovery step | `5` |
| `ROQUO_OMPTHREADS` | OpenMP threads for host-side work | `140` |
| `SEED_CISD` / `SEED_CISD_FRAC` | inject CISD seed determinants | `0` / `1.0` |
| `DO_RDM` / `ITERATIONS` / `NUM_WALKERS` | orbital-optimization loop knobs | `0` / `1` / `1` |

### Step 7. Multi-GPU (optional)

The single-GPU path is the robust default (the solver can OOM when several ranks
share one GPU). For a genuine multi-GPU run, request several GPUs and let the
solver use NCCL across them within one allocation:

```bash
sbatch --account=<code> --partition=roquo \
       --nodes=1 --ntasks-per-node=4 --gres=gpu:4 --time=03:00:00 \
       run_fe4s4_gpu_roquo.sh
```

The in-allocation `srun --gpu-bind=closest` assigns one GPU per rank. Cross-node
multi-GPU (NCCL over the fabric) works for the sampling/aggregation stages;
cross-node Davidson can deadlock, so keep the eigensolver within a single node.

---

## Troubleshooting

### `module: command not found` or `nvc++` missing on the login node
Build and run on a **compute node**. The login node lacks `cuda`/`nvhpc`; the
`module load` lines only resolve inside an `srun`/`sbatch` allocation.

### `sbatch: invalid partition specified`
The GPU partition is `roquo` (check `sinfo`). Some clusters name it differently;
pass `--partition=<name>` and update the wrapper's `#SBATCH --partition` line.

### Solver runs out of GPU memory
Reduce `FE4S4_SQD_DIM`, or drop to `--gres=gpu:1` with a single rank. Multiple
ranks on one GPU multiply the subspace memory footprint.

### Nested `sbatch` jobs appear during recovery (slow)
That is the Slurm target (`--hpc-target slurm`), which submits each Davidson
solve as its own job. The ROQUO wrapper uses `--hpc-target local` to avoid this —
confirm `squeue` shows only your one orchestrator job.

### Open MPI / PMIx errors under `srun`
Load `hpcx/2.50` in the wrapper and confirm `srun --mpi=list` includes `pmix`.
Use `srun --mpi=pmix` for cross-rank launches (the wrapper does this).

### `diag-gpu` crashes immediately on a UHF FCIDUMP
Build and use `diag-gpu_uhf` (`-D_UHF`) for unrestricted / broken-symmetry
FCIDUMPs; the RHF binary assumes the restricted integral layout.
