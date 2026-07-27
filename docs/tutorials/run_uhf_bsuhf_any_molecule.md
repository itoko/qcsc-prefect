# Run UHF / Broken-Symmetry-UHF SQD for Any Molecule (Fugaku + ROQUO)

The SBD closed-loop pipeline in this repo — `riken_sqd_de`, `FlowParameters`, the broken-symmetry
UHF guess in `qcsc_workflow_utility.chem` — is molecule-agnostic. Everything you've seen in this
repo's Fe2S2/Fe4S4 material is one *instance* of that generic pipeline, not a special code path.
This tutorial shows how to point the same pipeline at **your own** open-shell system, and how to
scale that up to a genuine multi-node, multi-step recovery run on Fugaku or ROQUO.

It complements, rather than replaces, the existing tutorials:

- [Create Your QCSC Workflow for Fugaku](create_qcsc_workflow_for_fugaku.md) and
  [Run SBD Closed-Loop Workflow (Fugaku)](run_sbd_closed_loop_workflow_fugaku.md) — the Prefect
  Flow/Task/Block/Deployment concepts and the UI-driven single-flow-run path (RHF, N2). Read that
  first if you're new to this repo's Prefect layer; this page assumes it.
- [Create Your QCSC Workflow for ROQUO (GB200 GPU)](create_qcsc_workflow_for_roquo_gpu.md) — the
  GPU-specific execution model (`--hpc-target local`, one Davidson solve per rank's GPU).
- [Sizing an SBD Job](../reference/hpc_resource_sizing.md) — the node/rank/thread/ADET-BDET math
  this tutorial's Step 6 points to.
- [SQD Concepts Explained](../2026-07-08/SQD_CONCEPTS_EXPLAINED.md) and
  [How Determinants Are Built](../2026-07-08/HOW_DETERMINANTS_ARE_BUILT.md) — the recovery/
  occupancy/carryover mechanics this tutorial doesn't re-derive.

## A naming quirk, up front

Several of the generic knobs below are named `FE4S4_*` (`FE4S4_AF_GROUPS`, `FE4S4_CKPT_DIR`,
`FE4S4_AF_POL`, `FE4S4_AF_FREE_S`) because they were first added for the Fe4S4 study. **The
mechanism behind each is fully generic** — the code only ever consumes the JSON/path you give it,
with zero Fe-specific logic — only the env var *name* is Fe4S4-branded. You'll use these same
variable names for any molecule.

## Step 0 — Get an FCIDUMP for your molecule

This repo consumes an FCIDUMP (the standard PySCF/Molpro integral-dump format: an `&FCI
NORB=.. NELEC=.. MS2=..` header followed by one- and two-electron integrals) — it does not
generate one. Any of the following produce a valid FCIDUMP:

- PySCF: `pyscf.tools.fcidump.from_scf(mf)` after an RHF/UHF/CASSCF calculation.
- A DMRG/CASSCF active-space carve-out (e.g. via ORCA's `%casscf` + DMRG block, or NWChem's
  `fcidump` block with `freeze core N virtual M` for a mid-window active space) when your system is
  too large for a full-orbital FCIDUMP.

Whatever tool you use, confirm the header parses to the electron/orbital counts you expect before
moving on — a wrong `MS2` or orbital count silently produces a physically different active space.

## Step 1 — Decide RHF vs UHF vs BS-UHF

- **RHF**: closed-shell singlet, no broken symmetry needed. Use `create_blocks.py --method rhf`.
- **UHF (energy-ordered)**: open-shell (non-zero `MS2`), or you want the unrestricted solution but
  don't need a specific spin-localized guess. `--method uhf` with no extra env vars.
- **BS-UHF (broken-symmetry)**: a formally closed-shell system (`MS2=0`) with strongly-correlated,
  spatially-separated open-shell character (Fe-S clusters are the case this repo was built around,
  but the mechanism is general — any multi-center system with localizable, oppositely-polarizable
  fragments qualifies). Plain UHF started from the RHF density stays *at* the RHF solution here
  (RHF is a stationary point of the UHF equations) unless you seed it with a spin-localized guess.

For BS-UHF, build an `AF_GROUPS` JSON spec and set it as `FE4S4_AF_GROUPS` (see the naming note
above):

```json
{
  "metal_a":  [2, 3, 4, 5, 6],
  "bridge":   [7, 8, 9, 10],
  "metal_b":  [11, 12, 13, 14, 15],
  "up":   ["metal_a"],
  "down": ["metal_b"]
}
```

This is the same schema the Fe4S4 convenience shortcut (`FE4S4_AF_GROUPS=fe4s4`) expands to
internally — see `_parse_af_groups()` in
[`chem.py`](https://github.com/qiskit-community/qcsc-prefect/blob/main/algorithms/qcsc_workflow_utility/src/qcsc_workflow_utility/chem.py):
any orbital-index fragment names work, not just `fe1`/`fe2`/`s`. Rules:

- Every fragment not listed in `up`/`down`/`free` is treated as **closed** (doubly occupied).
- `up` fragments get `0.5 + 0.5·pol` alpha / `0.5 − 0.5·pol` beta occupancy on their orbitals;
  `down` fragments get the reverse. `pol` (env `FE4S4_AF_POL`, default `1.0`) controls how sharply
  localized the guess is — `1.0` is a full 1/0 split, lower values couple the two sides more.
  If your BS-UHF converges to a highly spin-contaminated state, try a lower `pol` first.
  `free` fragments (env `FE4S4_AF_FREE_S=1`, if you name one `"s"`) are seeded half-filled/
  unpolarized instead of closed — useful for a bridging ligand that should mediate exchange rather
  than sit doubly-occupied.
- Orbital indices must cover every orbital in your FCIDUMP exactly once — mismatches will silently
  under- or over-count electrons in the guess.

## Step 2 — Build the SBD solver binaries

Covered in full in
[`algorithms/sbd/native/README.md`](https://github.com/qiskit-community/qcsc-prefect/blob/main/algorithms/sbd/native/README.md)
— not duplicated here. Short version: build both the restricted (`diag` / `diag-gpu`) and
unrestricted (`diag_uhf` / `diag-gpu_uhf`, `UHF=1`) binaries for your platform; you need the UHF
one for both plain-UHF and BS-UHF runs (the FCIDUMP layout differs from RHF's).

## Step 3 — A quick, no-HPC correctness check

Before spending node-hours, sanity-check your FCIDUMP and (if applicable) `AF_GROUPS` locally —
no Prefect server, no HPC, no IBM Quantum call:

```bash
cd algorithms/sbd
# RHF, any FCIDUMP, purely local:
python run_local.py --fcidump /path/to/your.fcidump --iters 2 --sqd_dim 10000

# UHF + orbital rotation, for a small test molecule already in run_local_uhf.py's registry
# (H3/OH/CH2/...) — add your own geometry to the MOLECULES dict there for a new small system:
python run_local_uhf.py --mol OH --iters 3 --sqd_dim 5000
```

`run_local_uhf.py` builds its molecule from a PySCF geometry string (its own small `MOLECULES`
dict), not from an arbitrary FCIDUMP — it's meant for fast local UHF/orbital-rotation validation
on a handful of atoms, not for driving your production system. For a BS-UHF FCIDUMP from Step 0/1,
the next real correctness check is a small single-flow run (Step 4) with `quantum_source="random"`
and a small `sqd_dim`, which exercises the real FCIDUMP + `AF_GROUPS` path end-to-end.

## Step 4 — A single-flow run (small scale, either HPC target)

This is the path from the existing [closed-loop tutorial](run_sbd_closed_loop_workflow_fugaku.md):
generate blocks with `create_blocks.py`, then run one `riken_sqd_de` flow (`n_recovery_steps=1` or
a small number) via the Prefect UI or CLI. The only things that change for UHF/BS-UHF are:

```bash
python algorithms/sbd/create_blocks.py \
  --config algorithms/sbd/sbd_blocks.toml \
  --hpc-target fugaku \
  --method uhf \
  --sbd-executable-uhf /path/to/diag_uhf
```

and, if using BS-UHF, exporting `FE4S4_AF_GROUPS` (Step 1) in the environment the flow runs in —
the flow reads it from the process environment, not from a block field.

**Fugaku vs. ROQUO delta at this step:**

| | Fugaku | ROQUO |
| --- | --- | --- |
| `--hpc-target` | `fugaku` | `local` (in-allocation, fast path) or `slurm` (nested `sbatch` per solve, slow path — avoid for deep runs) |
| launcher | `mpiexec` (PJM) | `srun` (GPU, in-allocation) |
| `--solver-mode` | `fugaku` (CPU) | `gpu` |
| executable | `diag_uhf` | `diag-gpu_uhf` |

## Step 5 — Deep multi-step recovery with checkpoint/resume

A "deep run" — dozens to hundreds of recovery steps at a large `sqd_dim`, sampling once and
re-diagonalizing offline (`quantum_source="saved"`) — is the **same** `FlowParameters` used above
with `n_recovery_steps`/`n_batches` set high, plus two things a single UI run doesn't need:

- **A persisted sample pool** — sample once (`quantum_source="real-device"` or `"random"`), then
  point every subsequent recovery run at the saved `.npz` pool(s) with `quantum_source="saved"` so
  you never re-sample the device.
- **A checkpoint directory** — set `FE4S4_CKPT_DIR` (again: generic mechanism, Fe4S4-branded name)
  to a real path. After every completed recovery step, `sqd.py` writes
  `ckpt_t<trial>_w<walker>.npz` with `next_step`/`best`; on the next run with the *same*
  `FE4S4_CKPT_DIR`, already-completed steps are skipped automatically — this is what makes a
  24h-wall-time-limited job resumable in minutes instead of hours.

[`examples/sbd_uhf_recover_any_molecule/run_recover.py`](https://github.com/qiskit-community/qcsc-prefect/blob/main/examples/sbd_uhf_recover_any_molecule/run_recover.py)
wraps all of this into one CLI, molecule-agnostic (no Fe4S4 defaults):

```bash
python examples/sbd_uhf_recover_any_molecule/run_recover.py \
  --fcidump /path/to/your.fcidump \
  --pool file:///path/to/raw_samples.npz \
  --af-groups '{"metal_a":[2,3,4,5,6],"bridge":[7,8,9,10],"metal_b":[11,12,13,14,15],"up":["metal_a"],"down":["metal_b"]}' \
  --method uhf --sqd-dim 4000000000 --recovery-steps 100 --n-batches 1 \
  --ckpt-dir /path/to/ckpt --hpc-target fugaku \
  --nodes 2304 --ranks-per-node 4 --omp-threads 12 --adet 96 --bdet 96 --queue large
```

For ROQUO, the same script with `--hpc-target local --nodes 1 --ranks-per-node <GPUs>`. See its
`README.md` for the full flag reference and a ready ROQUO example.

Re-running the **exact same command** (same `--ckpt-dir`) after a wall-time kill or any other
interruption resumes automatically — look for this log line to confirm it picked up where it left
off:

```
[ckpt] RESUME from <ckpt-dir>/ckpt_t00_w0.npz: completed <N> step(s), best_energy so far=<E>
```

## Step 6 — Size the job

See [Sizing an SBD Job](../reference/hpc_resource_sizing.md) for the node/rank/thread and
ADET/BDET math, a worked table, and ROQUO GPU sizing — this is what decides the `--nodes
--ranks-per-node --omp-threads --adet --bdet` values in Step 5.

## Step 7 — Monitor and diagnose

- Fugaku: `squeue -u $USER` for the orchestrator (on `mem2`), `pjstat` / `pjstat -H` for the
  per-step compute-partition PJM jobs. A step "not updating" is very often a **queue wait**, not a
  hang — see the last section of the [sizing reference](../reference/hpc_resource_sizing.md).
- ROQUO: `squeue -u $USER` — with `--hpc-target local` there's only the one orchestrator job; no
  per-step queue to check.
- Either system: `tail -f` the job's `.err`/`.out` and look for the per-step
  `[diag] recovery <n>/<total> ... dets: ... S=.. D=.. T=.. Q=.. ≥5=..` line — this is the
  excitation-rank breakdown of the current recovered subspace, and a healthy run should show `S`
  saturate near its theoretical max (`1 + norb_active` singles) within the first few steps while
  `D`/`T`/`Q` keep growing.

## Troubleshooting

**BS-UHF converges but `<S^2>` is very large / result looks wrong.** Your `AF_GROUPS` fragments
probably don't partition the orbital space correctly (overlap or gap), or `pol=1.0` is over-
localizing for this system — try a fractional `FE4S4_AF_POL` (e.g. `0.6`) or mark a bridging
fragment `free`.

**`NameError` or missing-module error deep in the checkpoint code path.** Make sure you're on a
version of `sqd.py` that imports `os` at module level — this was a real bug (checkpoint code used
`os.environ`/`os.path` without the import) fixed alongside this tutorial.

**Fugaku job killed mid-step, nothing lost?** Confirm the checkpoint directory has a
`ckpt_t*_w*.npz` newer than your last completed step's log line, then just resubmit the same
launcher — see Step 5's resume behavior.

**A GPU/CPU binary crashes immediately on your FCIDUMP.** Confirm you built and are pointing at
the `_uhf` binary variant for any UHF/BS-UHF FCIDUMP — the restricted binary assumes a different
integral layout and will not just be slower, it will produce wrong or crashing output.

**Multi-node Fugaku job hangs at startup (SIO-node overload symptoms).** Confirm your Fugaku batch
template invokes `llio_transfer` on the executable before `mpiexec` for large node counts — this
is wired into the shipped `batch.pjm.j2` template already; if you've customized it, check the fix
is still present.
