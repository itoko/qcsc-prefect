# Handover: RDM Diagnostics and Orbital Optimization

**Branch:** `ylt/feature/rdm-check-diag` · **13 commits ahead of `origin/main` · 20 files · +2584 / −65 lines** · author: Yuto Terashima

This document hands over the work on the `ylt/feature/rdm-check-diag` branch. The
branch name says "RDM check", but the branch actually holds **two related lines of
work**:

- **Part A — RDM correctness and diagnostics.** A bug fix and several diagnostic
  tools for the reduced density matrices (RDMs) that the native SBD solver returns.
- **Part B — Orbital optimization (OO).** A large set of OO features that ride on the
  same branch. They run between differential-evolution (DE) trials in the SBD flow.

The reader is expected to be comfortable reading code, so this document points at
files and functions instead of copying everything. But for **how to use** the new
features (build steps, environment variables, run commands) it is deliberately
step by step.

A guiding rule for the whole branch, like the earlier UHF work: **additive and
default-off.** Every OO path is a new branch guarded by an environment variable, and
all of them default to OFF. The RDM diagnostics live in separate binaries that never
overwrite the production solver. So existing runs behave exactly as before.

---

## TL;DR — what changed

**Part A (RDM):**

- Fixed a wrong Fermi sign in the **opposite-spin (ab/ba) 2-RDM** double-excitation
  elements in the native SBD kernel. Before the fix, the energy rebuilt from the RDMs
  (`E_recon`) did not match the solver's Davidson energy.
- Added a **pure-Python + PySCF numerical proof** that the patched kernel is correct
  (matches PySCF to ~1e-16; the original is wrong only in the ab/ba block).
- Added an **in-process "RDM-CHECK recon"** that contracts the FCIDUMP with the
  returned RDMs inside the solver and prints `E_RDM_recon` next to `E_davidson`.
- Fixed a separate **UHF integral-sector bug** in that recon (it contracted every spin
  block against alpha-alpha integrals, giving a false +19 Ha on spin-polarized UHF
  FCIDUMPs).

**Part B (OO):**

- OO can now run **between DE trials** and as an **initial step** (`OO_INIT`), with
  energy guards, GPU re-check (`OO_CHECK`), a CC seed refresh in the rotated basis
  (`OO_TO_LUCJ`), and a unified convergence knob (`OO_GRAD_TOL`).
- Added formal `FlowParameters` fields for OO (config/deployment side).
- Added excitation-population diagnostics in `sqd.py` (`[exc-pop]`, `[exc-cnt]`).

---

## Part A — RDM correctness and diagnostics

### A.1 The bug: opposite-spin 2-RDM sign

Commit `bb3e562`. The native `TwoDiffCorrelation` kernel computed the wrong Fermi
sign for **opposite-spin (ab/ba)** 2-RDM double-excitation elements. It used a
sorted-index pairing that crossed spins and ignored the intermediate determinant.
The same-spin blocks (aa/bb) were fine.

The visible symptom: the energy rebuilt from the RDMs did **not** equal the solver's
Davidson energy. The fix branches on `si != sj` and recomputes the opposite-spin sign
with spin-consistent pairing.

### A.2 The fix is patch-managed (important)

The native trees `algorithms/sbd/native/sbd/` and `.../sbd_mpi/` are **gitignored
upstream clones**. So the fix is **not committed inside them.** It lives as patch files
that the build scripts re-apply every time:

- `algorithms/sbd/native/patches/sbd-rdm-opposite-spin-sign.patch` — GPU
  (`basic/correlation_thrust.h` + `correlation.h`)
- `algorithms/sbd/native/patches/sbd_mpi-rdm-opposite-spin-sign.patch` — MPI
- `algorithms/sbd/native/patches/apply_rdm_patches.sh` — idempotent applier. It first
  tries `--reverse --check`, and skips if the patch is already applied. Safe to run
  again.

> **Watch out:** if you re-clone the native tree, the fix is gone until you run a build
> script (or `apply_rdm_patches.sh`) again. Never assume the clone is patched.

### A.3 PySCF numerical proof

Commit `32c7f12`. `algorithms/sbd/native/patches/test_rdm_vs_pyscf.py` is a
self-contained pure-Python + PySCF replica of the correlation kernel — no GPU, no
compiled binary. It builds all four 2-RDM blocks and the 1-RDM from a full-CI vector,
using **both** the original and the patched kernel logic, and compares them
element-wise to PySCF `make_rdm12s`.

Result: patched matches to ~1e-16 and reconstructs the Davidson energy; the original
gets only the ab/ba block wrong (max|Δ| ~1e-2, `E_recon` off by ~1–9 mHa).

Run it on a compute node (share a running job's allocation):

```bash
cd /home/nfs1/q0000230/qcsc-prefect/algorithms/sbd
srun --jobid=<running-jobid> --overlap -N1 -n1 \
  .venv/bin/python native/patches/test_rdm_vs_pyscf.py
```

There is also `native/patches/run_rdm_pyscf_test.sbatch` to run it as its own job.

### A.4 In-process RDM-CHECK recon

The namesake diagnostic. Commit `ac2d951` plus the new binary source
`algorithms/sbd/native/main_rdmcheck.cc`. After the solver returns the RDMs, this
code contracts the **same** FCIDUMP it just loaded with the **same** RDMs, in the
**same process**, and prints `E_RDM_recon` directly next to `E_davidson`. No FCIDUMP
round-trip and no Python contraction in between. This separates two failure modes:

- `E_RDM_recon == E_davidson` → native is self-consistent; any Python-side delta is a
  FCIDUMP round-trip / integral-convention mismatch.
- `E_RDM_recon != E_davidson` → the solver's energy-W and RDM-W differ (solver-side
  bug).

The recon itself had a bug (fixed in `ac2d951`): it contracted **every** spin block
against the alpha-alpha integral sector, which is correct only for RHF. On a
spin-expanded UHF FCIDUMP this gave a false +19 Ha delta (Fe4S4 ZDL). Now each block
uses its own spin indices. In `algorithms/sbd/native/main.cc` (around lines 226–246):

```cpp
// spin-orbital index = 2*orbital + spin (0=alpha, 1=beta)
//   1-RDM: block s (0=a,1=b)                 -> I1.Value(2io+s, 2jo+s)
//   2-RDM: block s+2t (0=aa,1=ba,2=ab,3=bb)  -> I2.Value(2io+s, 2ia+s, 2jo+t, 2ja+t)
onebody += I1.Value(2*io+s, 2*jo+s) * one_p_rdm[s][io+L*jo];
twobody += 0.5 * I2.Value(2*io+s, 2*ia+s, 2*jo+t, 2*ja+t)
                 * g2[io+L*jo+L*L*ia+L*L*L*ja];
```

The solver prints lines tagged `[RDM-CHECK]` (Zero/One/Two-Body energy, `E_RDM_recon`,
`E_davidson`, and their delta). GPU-verified: Fe4S4 delta 19.11 Ha → 2.4e-8;
NORB=20 case 2.52 → 1e-12. This is **diagnostic only** — the solver, the RDM output,
and integral loading are untouched.

> **Caveat:** `main.cc` and `main_rdmcheck.cc` are currently byte-identical (the recon
> fix was folded back into both). So the `_diag` binary now differs from the fixed
> production binary only by its build-target name, not by source. If you keep both,
> consider merging them or deleting `main_rdmcheck.cc` later.

---

## Part B — Orbital optimization (OO)

### B.1 When OO runs, and where the code is

OO runs between DE trials whenever the solver wrote RDMs (`do_rdm != 0`), and
optionally once as an initial step before the t2 seed. Entry points:

- `algorithms/sbd/sbd/main.py` — the DE/OO loop: reads the env flags, calls
  `optimize_orbitals`, applies the energy guards, and does the logging.
- `algorithms/qcsc_workflow_utility/src/qcsc_workflow_utility/orbital_opt.py` —
  `optimize_orbitals`, `rotate_electronic_properties`,
  `resolve_orbitals_self_consistent`.
- `algorithms/qcsc_workflow_utility/src/qcsc_workflow_utility/chem.py` —
  `_apply_initial_oo` (OO_INIT), `refresh_cc_seed` / `_refresh_cc_seed_uhf` /
  `_refresh_cc_seed_rhf` (CC refresh in the rotated basis), a BS-UHF MO cache, and a
  NumPy-2.x PySCF DIIS shim.

### B.2 Environment-variable knobs

All default OFF or to a legacy value, so existing runs are unchanged. Set them in the
job script (see Section 6).

| Variable | Default | Meaning |
|---|---|---|
| `OO_INIT` | 0 | One OO step between the initial BS-UHF/UCCSD and the t2 seed. An energy-gain guard discards the rotation on no gain or NaN. |
| `OO_REFIRE` | 0 | Run OO again on each DE trial (DE-loop OO). |
| `OO_CHECK` | 0 | Re-diagonalize the same subspace on GPU (Davidson) to confirm OO actually lowered the variational energy. Logs `E_before` / `E_after` / `E_sc`. |
| `OO_TO_LUCJ` | 0 | Refresh the UCCSD seed in the rotated basis after OO (stagnation-triggered variant). |
| `OO_GRAD_TOL` | 1e-5 | Single knob for inner L-BFGS-B `pgtol` and outer Brillouin-freeze convergence on all OO paths (unified in `eca0121`). |
| `OO_TRUST` | 0.5 | Trust radius for the OO step. |
| `OO_MAXITER` | 300 | Max OO iterations. |
| `OO_RESOLVE` | 0 | Self-consistently re-solve the RDMs after rotation. |
| `OO_RESOLVE_BACKEND` | davidson_gpu | Backend for the re-solve (`solve_fermion` or `davidson_gpu`). |
| `OO_SKIP_NONBEST` | 0 | Skip OO when the current trial did not beat the all-time best Davidson energy. |
| `OO_SAT_STAGNATION` | 2 | Trials of stagnation before a CC refresh triggers. |
| `OO_DE_TOL` | 1e-3 | DE-loop OO tolerance. |
| `SEED_CCSD_MAX_CYCLE` | 200 | Max CCSD cycles for the seed refresh. |
| `SBD_UHF_CACHE` | on | Cache converged BS-UHF MOs so every run starts from an identical reference (Fe4S4's ~400-cycle SCF is sensitive to BLAS ordering). |
| `SBD_UHF_CACHE_DIR` | `~/.cache/qcsc_uhf` | Where that cache lives. |
| `FE4S4_AF_GROUPS` | (unset) | Atom-localized antiferromagnetic guess (pre-existing). |

### B.3 Prefect `FlowParameters` fields

`algorithms/sbd/sbd/flow_params.py` adds formal OO fields (these are the
config-file / deployment knobs, distinct from the env vars above; each has a long
docstring in the file):

`oo_grad_tol`, `oo_trust_radius`, `oo_maxiter`, `oo_selfconsistency_tol`, `oo_de_tol`,
`oo_resolve_rdms`, `oo_resolve_maxdim`, `oo_resolve_backend`, `oo_refire_every_trial`.

Use these when you drive the flow through a Prefect deployment / config file. Use the
env vars when you drive it from a job script.

### B.4 Excitation-population diagnostics (`sqd.py`)

Unrelated to OO. `algorithms/sbd/sbd/sqd.py` logs, per recovery step:

- `[exc-pop]` — probability-weighted population of excitation levels
  (HF / {1–2} / {3} / {4} / {>4}).
- `[exc-cnt]` — raw configuration counts of the same levels.

Useful to see how the sampled subspace is distributed.

---

## 5. How to build the native binaries (roquo)

Two build scripts under `algorithms/sbd/native/`. Both re-apply the sign patch first,
and both write to a **separate output name**, so they never overwrite the production
binaries `diag-gpu_uhf` (single-GPU) or `diag-gpu_uhf-mpi` (MPI).

```bash
module load nvhpc/26.5
module load nvhpc-openmpi/26.5
cd /home/nfs1/q0000230/qcsc-prefect/algorithms/sbd/native

# Diagnostic binary (from main_rdmcheck.cc) -> diag-gpu_uhf_diag
./build_rdmcheck_roquo.sh

# Fixed production-equivalent binary (from main.cc, -D_UHF) -> diag-gpu_uhf_fixed
./build_sbd_gpu_uhf_fixed.sh
```

Both use compiler `mpic++` with
`-std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST -D_UHF`. They must run on a
node that has `nvhpc/26.5` loaded. There are `.sbatch` wrappers next to each script if
you want to build inside a batch job instead of on the login node.

---

## 6. How to run an end-to-end job (worked example)

The clearest template is a real job script:
`/home/nfs1/q0000230/work/fe4s4_val/run_fe4s4_oo_init_ibm_kobe_100k.sh`. It is a
Fe4S4 OO_INIT study that samples from IBM Quantum. Below is what each part does.

**SLURM header** — one GPU, 36 CPUs, 72 h, on the `roquo` partition, with a guard that
refuses `large` partitions/reservations:

```bash
#SBATCH --account=qc-prj-other02
#SBATCH --partition=roquo
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=36
#SBATCH --time=72:00:00
```

**Toolchain and Prefect** — load nvhpc, put `uv` on PATH, set threads, and run Prefect
in ephemeral (server-less) mode:

```bash
module load nvhpc/26.5 nvhpc-openmpi/26.5
export PATH="$HOME/opt/uv-aarch64/bin:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-36}"
export PREFECT_HOME="/tmp/prefect_${SLURM_JOB_ID:-local}"
export PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=true
export SBD_TASK_RUNNER=concurrent
```

**Physics / sampling** — Fe4S4 broken-symmetry guess, subspace size, recovery steps,
and IBM Quantum backend:

```bash
export FE4S4_AF_GROUPS=fe4s4  FE4S4_AF_POL=1.0
export SQD_DIM=200000000  N_RECOVERY=240  ITERS=1  NUM_WALKERS=1  SEED=42
export QSOURCE=real-device  IBM_BACKEND=ibm_kobe  SHOTS=5000000
```

**OO knobs** — the values this study used (see the table in Section B.2):

```bash
export OO_INIT=1
export OO_GRAD_TOL=1e-5  OO_TRUST=0.5  OO_MAXITER=300
export OO_RESOLVE=0  OO_RESOLVE_BACKEND=davidson_gpu
export OO_REFIRE=1  OO_CHECK=1  OO_SKIP_NONBEST=0  OO_TO_LUCJ=0
export OO_SAT_STAGNATION=2  OO_DE_TOL=1e-3
```

**Three run steps** the script then does:

1. Build the solver block and set the shot count:

   ```bash
   cd /home/nfs1/q0000230/qcsc-prefect/algorithms/sbd
   uv run --no-sync python create_blocks.py \
     --config sbd_blocks.u-gpu.strict.toml \
     --shots "$SHOTS" --carryover-type 1 --carryover-ratio 0.5
   ```

2. Register the IBM Quantum runner (needed for `QSOURCE=real-device`), saved as the
   Prefect block `ibm-runner` via `prefect_qiskit.QuantumRuntime(...).save("ibm-runner")`.

3. Launch the flow:

   ```bash
   cd /home/nfs1/q0000230/qcsc-prefect
   uv run --no-sync --project algorithms/sbd python \
     algorithms/sbd/run_oo_study_fe4s4_zdl.py
   ```

**Submit and read logs:**

```bash
sbatch /home/nfs1/q0000230/work/fe4s4_val/run_fe4s4_oo_init_ibm_kobe_100k.sh
# output/error both go to fe4s4_ooinit_ibm100k.<jobid>.log
```

**Reusing the script.** Every knob is written as `${VAR:-default}`, so you can override
any of them from the command line without editing the file:

```bash
OO_INIT=0 SHOTS=100000 IBM_BACKEND=ibm_fez \
  sbatch /home/nfs1/q0000230/work/fe4s4_val/run_fe4s4_oo_init_ibm_kobe_100k.sh
```

To reproduce with the fixed/diagnostic solver, point the flow at `diag-gpu_uhf_fixed`
(or `diag-gpu_uhf_diag` for the `[RDM-CHECK]` output) instead of the production binary,
and pass `--rdm N` (N != 0) so the solver writes `rdm1_a/b.txt` and
`rdm2_aa/ab/bb.txt` (native "prqs" storage) that OO consumes.

---

## 7. Provenance and upstream plan

The native fix is meant to go upstream. The write-ups currently live only under
`algorithms/sbd/native/patches/` (not yet in `docs/`):

- `patches/README.md` — what the patches are and how to apply them.
- `patches/UPSTREAM_PR_PLAN.md` — the plan to send the sign fix upstream.
- `patches/RDM_PYSCF_PROOF.md` — the PySCF proof write-up.

Remember: the native `sbd/` and `sbd_mpi/` trees are re-clonable upstream checkouts,
so the source of truth for the fix is the patch files, not the clone.

---

## 8. File map

Native / patches:

- `algorithms/sbd/native/main.cc` — production solver (now carries the recon fix).
- `algorithms/sbd/native/main_rdmcheck.cc` — diagnostic binary source (currently
  identical to `main.cc`).
- `algorithms/sbd/native/build_rdmcheck_roquo.sh` (+ `.sbatch`)
- `algorithms/sbd/native/build_sbd_gpu_uhf_fixed.sh` (+ `.sbatch`)
- `algorithms/sbd/native/patches/{README.md, UPSTREAM_PR_PLAN.md, RDM_PYSCF_PROOF.md,
  test_rdm_vs_pyscf.py, run_rdm_pyscf_test.sbatch, apply_rdm_patches.sh,
  sbd-rdm-opposite-spin-sign.patch, sbd_mpi-rdm-opposite-spin-sign.patch}`

Python:

- `algorithms/sbd/sbd/{main.py, sqd.py, flow_params.py, solver_job.py}`
- `algorithms/qcsc_workflow_utility/src/qcsc_workflow_utility/{chem.py, orbital_opt.py}`

Example job script:

- `/home/nfs1/q0000230/work/fe4s4_val/run_fe4s4_oo_init_ibm_kobe_100k.sh`

---

## 9. Open items and notes

- **`main.cc` ≡ `main_rdmcheck.cc`.** They are byte-identical after the fix was folded
  in. Decide whether to keep the diagnostic as a separate binary or merge/remove it.
- **Uncommitted scratch.** The working tree has untracked experiment files
  (`run_oo_study_*.py`, `sbd_blocks.*.toml`, SLURM `.err/.out` logs). Decide which of
  these should be committed (some, like `run_oo_study_fe4s4_zdl.py`, are referenced by
  the job script and probably should be).
- **Upstream PR.** The sign fix still needs to be submitted upstream per
  `patches/UPSTREAM_PR_PLAN.md`; until then the patch-apply step is mandatory on every
  fresh clone.
