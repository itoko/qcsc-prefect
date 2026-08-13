# Fe2S2 RHF-vs-UHF multi-iteration recovery study

Clean, reproducible harness for comparing **RHF vs UHF** SQD on the Fe2S2 40q active space
(30 electrons, 20 orbitals, MS2=0) across configuration-recovery iterations, with near-exact
reference lines (DMRG, UCCSD, CCSD(T), HCI) for the presentation.

## Design

- **RHF vs UHF is chosen by the solver block** (`create_blocks.py --method`), not a flow flag.
- **Sample once per method, then reuse.** The device is sampled a single time per method; the
  merged shot pool is persisted and every recovery-depth experiment re-diagonalizes it offline
  (`quantum_source="saved"`). The expensive device shots are never re-taken.
- **One parent, one folder per run.** Everything lives under `runs/`; identity is in the folder
  name (`fe2s2_uhf`, `fe2s2_rhf`).
- **Credentials come from the environment only.** Nothing secret is committed — the launchers
  source a gitignored `sweep/.env.local` (template: `env.local.example`).

```
runs/
  fe2s2_uhf/
    samples/        persisted merged pool + pool_manifest.json   (sample ONCE)
    recover/        per-recovery-step telemetry JSON             (offline re-diag)
    post/           this method's copy of the plots + energies.csv
    prefect_home/   isolated Prefect DB + storage for this run   (gitignored)
    work_sample/, work_recover/                                  (solver work dirs)
  fe2s2_rhf/  ...
  fe2s2_post/       combined RHF-vs-UHF plots + energies.csv
  refs.json         DMRG / UCCSD / CCSD(T) / HCI reference energies
```

## Constraints

- **Only one Fugaku job at a time.** Never `sbatch` while another job (yours or a running
  large-scale run) is queued/active. Check with `pjstat` first.
- Runs execute on the pre/post `mem2` partition: `sbatch --partition=mem2 --time=<min> <script>`.

## Workflow

```bash
# 0. one-time: credentials
cp env.local.example .env.local && $EDITOR .env.local      # fill IBM_API_KEY / IBM_CRN / IBM_BACKEND

# 1. sample ONCE per method (real device). Idempotent: re-running skips if a pool exists.
METHOD=uhf sbatch --partition=mem2 --time=360 run_fe2s2_sample.sh
METHOD=rhf sbatch --partition=mem2 --time=360 run_fe2s2_sample.sh
#    cheap dry-run with no IBM call:
#    FE2S2_QSRC=random METHOD=uhf sbatch --partition=mem2 --time=60 run_fe2s2_sample.sh

# 2. reference energies (reference venv: pyscf + block2). Reuse existing where available.
sbatch --partition=mem2 --time=360 run_fe2s2_refs.sh

# 3. offline recovery sweep from the saved pool (NO device call). One run gives the whole curve.
METHOD=uhf MAX_RECOVERY=10 sbatch --partition=mem2 --time=360 run_fe2s2_recover.sh
METHOD=rhf MAX_RECOVERY=10 sbatch --partition=mem2 --time=360 run_fe2s2_recover.sh

# 4. plots + energies.csv (local, no scheduler)
python fe2s2_plot.py
```

## Files

| file | role |
|------|------|
| `fe2s2_common.py`   | shared paths, `create_blocks` arg builder, saved-pool discovery |
| `fe2s2_sample.py`   | sample once per method, persist pool (idempotent) |
| `fe2s2_recover.py`  | offline recovery sweep, dumps per-step `recovery_trace` JSON |
| `build_fe2s2_refs.py` | UHF/UCCSD/CCSD(T)/HCI/DMRG → `runs/refs.json` |
| `fe2s2_plot.py`     | presentation plots (energy vs iter + panels) + `energies.csv` |
| `run_fe2s2_*.sh`    | thin sbatch launchers |
| `env.local.example` | credential template (copy to gitignored `.env.local`) |

## Plots

`fe2s2_plot.py` writes to `runs/fe2s2_post/`:

- `fe2s2_energy_vs_iter.{png,svg}` — E_SQD per recovery iteration, RHF & UHF, with flat DMRG /
  CCSD(T) / UCCSD / HCI reference lines.
- `fe2s2_panels.{png,svg}` — 2×2: energy, error vs DMRG (mHa), net subspace dimension, spin
  density (2·Sz).
- `energies.csv` — method, iter, energy, ΔE-vs-DMRG, net_dim, n_post, 2Sz, useful-det fraction.

Colors follow the repo's dataviz conventions (CVD-safe: UHF blue, RHF orange; one y-axis per
panel; references gray/dashed with direct labels).

The per-iteration data comes from the `recovery_trace` field added to `sbd/sqd.py` (the recovery
loop records energy, subspace dim, spin density, and useful-det fraction at every pass, not just
the best one).
