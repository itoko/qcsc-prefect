# Sizing an SBD Job: Nodes, MPI Ranks, Threads, ADET/BDET

This page answers "how many nodes / MPI processes / threads should I request?" for an SBD
Davidson-solver job (RHF or UHF) on **Fugaku** or **ROQUO**. It is system-agnostic math first,
then a per-system table. It applies to any molecule — the constraints come from the solver's
communicator layout, not the chemistry.

See [Run UHF/BS-UHF for Any Molecule](../tutorials/run_uhf_bsuhf_any_molecule.md) for how these
numbers plug into `create_blocks.py` / the example recovery script.

## The two independent constraints

### 1. Per-node oversubscription: `ranks_per_node × omp_threads ≤ physical cores/node`

Every MPI rank on a node gets `omp_threads` OpenMP threads for its share of the local BLAS/LAPACK
work. Exceeding the physical core count oversubscribes and slows every rank down — this is a hard
correctness-of-performance rule, not a soft guideline.

| System | Physical cores/node | Typical split |
| --- | --- | --- |
| Fugaku (A64FX, 4 CMGs/node) | 48 | proven: 1 rank × 48 threads; CMG-optimized (untested at scale): 4 ranks × 12 threads (1 per CMG) |
| ROQUO (GB200, 2× Grace, aarch64) | 144 | typically 1 rank/node for the GPU solver (host threads do I/O + subsample prep), `ROQUO_OMPTHREADS=140` leaves headroom for the OS |

Fugaku's `create_blocks.py --num-nodes/--mpiprocs/--ompthreads` (or the CMG-variant script's
`FE4S4_RANKS_PER_NODE`/`FE4S4_OMP_THREADS`) enforces this: `ranks_per_node * omp_threads > 48`
raises immediately rather than silently oversubscribing.

### 2. Communicator grid: `adet_comm_size × bdet_comm_size × task_comm_size == total MPI ranks`

The Davidson solver splits its alpha-determinant and beta-determinant axes across
`--adet-comm-size` / `--bdet-comm-size` ranks respectively (`--task-comm-size` is almost always
`1`). Their product **must equal** `total MPI ranks = nodes × ranks_per_node` — `create_blocks.py`
(and the CMG launcher) checks this and exits if it doesn't multiply out.

The subspace those ranks jointly diagonalize is the outer product of the per-spin determinant
lists:

```
dets_per_spin  ≈ sqrt(sqd_dim)          # SQD builds an equal-size alpha and beta list
net_dim        = (alpha dets) × (beta dets) ≈ sqd_dim
```

`adet_comm_size`/`bdet_comm_size` don't have to equal `sqrt(total_ranks)` exactly, but a roughly
square grid balances memory/compute evenly across ranks — this is what "ADET × BDET grid" means
below.

## Worked sizing table (Fugaku, `diag_uhf`)

These are real numbers from production Fe4S4 (54e, 36o / 72-qubit) runs — the grid choice
generalizes to any molecule at the same `sqd_dim`, since the constraint is purely combinatorial.

| `sqd_dim` | dets/spin (≈`sqrt`) | ADET × BDET grid | total ranks | nodes × ranks/node |
| --- | --- | --- | --- | --- |
| 1e8 | ~10,000 | 16 × 16 | 256 | 256 × 1 |
| 1e9 | ~31,623 | 48 × 48 | 2,304 | 2,304 × 1 |
| 4e9 | ~63,245 | 96 × 96 | 9,216 | 2,304 × 4 (CMG variant) |

The 4e9 row is the CMG-optimized 4-ranks/node layout (12 threads/rank); the same 9,216-rank total
could instead be 9,216 nodes × 1 rank (proven layout, but far more nodes for the same rank count —
usually not worth it unless you're rank-count-bound for another reason).

**Memory scales with `bdet_comm_size` per rank**, not with `sqd_dim` directly: each rank holds
roughly `(dets/spin) / bdet_comm_size` beta strings' worth of the wavefunction. If a run OOMs,
increase `bdet_comm_size` (more nodes/ranks) rather than shrinking `sqd_dim` first, unless you
have genuinely hit a node-count ceiling.

## ROQUO GPU sizing

The default is **1 GPU per rank** — the Davidson solver can OOM if several ranks share one GPU's
HBM. Request `--gres=gpu:N --ntasks-per-node=N` for `N` GPUs on one node; `main.cc` binds rank *i*
to GPU *i* automatically (`cudaSetDevice(rank % gpu_count)`), so **do not** pass `srun
--gpu-bind` — it's redundant with the binary's own binding and can conflict.

Cross-node multi-GPU (NCCL) is validated only within a single node's 4 GPUs to date — see
[`native/README.md`](https://github.com/qiskit-community/qcsc-prefect/blob/main/algorithms/sbd/native/README.md)'s
multi-GPU section for the exact `mpirun -np N --adet_comm_size/--bdet_comm_size` recipe (note:
`mpirun` from the loaded `hpcx` module, not `srun` — ROQUO's `srun` cannot bootstrap the HPC-X
OpenMPI these binaries link against).

Single-GPU is the robust default; reach for multi-GPU only when a subspace genuinely won't fit in
one GPU's memory (e.g. `sqd_dim ≥ 1e9`).

## Queue wait vs. a hung job

The two HPC targets have very different step-to-step latency profiles — knowing which one you're
on avoids mistaking a busy shared queue for a hang:

- **Fugaku**: each recovery step submits its own PJM job (via `pjsub`) to the compute partition and
  waits for it. On a busy shared resource group (e.g. `large`), the wait for node allocation can
  take anywhere from minutes to several hours — this is queue congestion, not a bug. Check
  `pjstat -H` for your job IDs; if one is sitting queued (not yet `RUN`), that's expected. The
  orchestrator process itself (on `mem2`, `squeue -u $USER`) stays alive the whole time it's
  waiting.
- **ROQUO**: `--hpc-target local` runs every Davidson solve **in-allocation** — no nested
  submission, no per-step queue wait. If a ROQUO step goes quiet for a long time, that's much more
  likely a genuinely slow solve (or a real hang) than a queue wait, since there's no scheduler in
  the loop between steps.

If you're unsure whether Fugaku is hung or queued, check the timestamp on the step's `job_*`
working directory under your solver's `work_dir` — a fresh directory with no output yet, paired
with the corresponding PJM job showing `QUE` in `pjstat`, means it's waiting on the scheduler, not
stuck in computation.
