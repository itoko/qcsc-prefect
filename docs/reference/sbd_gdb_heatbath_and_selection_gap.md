# The `sbd` library's `gdb` heat-bath CI code, and why SQD selection doesn't use it

This note documents two related findings from an investigation into whether this repo's SQD
recovery pipeline uses Slater-Condon (Hamiltonian-coupling) information when selecting which
sampled bitstrings survive into the diagonalized subspace, and whether the `sbd` native library
already contains usable heat-bath/selected-CI machinery that could fill that gap.

**Verdict, up front:** selection in `sqd.py` is coupling-blind today (particle-number sector +
sampled frequency + occupancy consistency only). A real, working Slater-Condon-based heat-bath CI
implementation already exists in the `sbd` library's `gdb` (General Determinant Basis) namespace
-- but it is entirely unwired from the production solver path (`sbd::tpb`) that this repo's
`main.cc` and every build script actually compile and run. It is upstream sample code, not
something this project wrote and forgot about, but it is directly reusable.

## 1. What "Slater-Condon rules" mean here, precisely

Slater-Condon rules govern which PAIRS of determinants have a nonzero Hamiltonian matrix element:
$\langle D_I|H|D_J\rangle$ is rigorously zero whenever $D_I$ and $D_J$ differ by more than two
occupied orbitals from EACH OTHER (the electronic Hamiltonian is a two-body operator). This is a
statement about pairwise coupling structure, not about a single determinant's "distance" from some
fixed reference like Hartree-Fock (HF) or the BS-UHF reference. A determinant that is a
triple-excitation from HF can still couple strongly to another determinant that is a
double-excitation from HF, because their *mutual* difference can be $\le 2$ even though each is
individually far from HF. Filtering "triples-from-HF" by rank alone is therefore not the same
question as filtering by Slater-Condon coupling, and is not the correct generalization of a
heat-bath-style importance criterion.

## 2. What this repo's SQD pipeline actually checks today

Four selection points exist in `algorithms/sbd/sbd/sqd.py`, none of them coupling-based:

- **`recover_configurations`** (calls `qiskit_addon_sqd.configuration_recovery.recover_configurations`)
  flips bits toward the current average orbital occupancy. Criterion: occupancy consistency.
- **`postselect_bitstrings`** (calls `post_select_by_hamming_weight`) keeps only bitstrings whose
  Hamming weight matches the correct $(N_\alpha, N_\beta)$ particle-number sector. Rigorous, but
  purely a sector filter, not an energetic one.
- **`subsample_open_shell` / `subsample_close_shell` / `_subsample_one_spin`** draw the determinants
  that actually go into the Davidson solve, weighted by **sampled probability**
  (`ci_probs_unique`, accumulated across shots). Carryover determinants from the previous solve's
  own high-$|c|$ output, and HF, are forced in first; everything else is drawn by frequency. No
  Hamiltonian matrix element, no $|H_{ij}|$ or $|H_{ij}c_j|$ score, appears anywhere in this draw.
- **`_merge_with_seed`** optionally injects classically-enumerated CISD singles/doubles
  (`_cisd_strings`) ahead of the sampled pool -- again classified by excitation rank from HF, not
  by coupling to anything already selected.

`_recombine_same_spin_candidates` unions disjoint excitation-delta bitmasks of two sampled
determinants to *synthesize* new higher-rank candidates, but does so "with zero Hamiltonian-matrix-
element evaluation" (its own docstring) -- ranked purely by the product of the two parents' sampled
weights, $w_i \cdot w_j$. It is coupling-blind by design, same as everything else.

The `[diag] recovery N/150 alpha dets: S=... D=... T=... higher(>2)=... max_exc=9` log lines and
the composition plots (`docs/2026-07-26/figures/composition_*.py`) come from
`_compute_excitation_counts`/`_excitation_summary`, which compute
`popcount(D_i XOR HF) // 2` -- excitation rank of each determinant **relative to the single fixed
HF/BS-UHF reference**, individually. This is a real and useful diagnostic (a subspace dominated by
rank->2 determinants literally cannot lower the energy below HF, since $H$ doesn't couple HF to
anything past doubles), but it is a single-determinant distance measure, not a Slater-Condon
pairwise-coupling calculation. Don't conflate the two.

## 3. A working heat-bath CI implementation already exists -- in the unused `gdb` namespace

The `sbd` library (cloned from `https://github.com/r-ccs-cms/sbd.git` by this repo's own
`algorithms/sbd/native/build_sbd_*.sh` scripts) ships genuine Slater-Condon-based heat-bath
screening, but in a code path our build scripts never touch.

**Where it lives** (inside the cloned tree, e.g. on ROQUO at
`~/qcsc-prefect/algorithms/sbd/native/sbd/include/sbd/chemistry/gdb/`):

- `expansion.h` -- `local_heatbath_expansion_lookup` computes real Slater-Condon matrix elements
  via `OneExcite(...)`/`TwoExcite(...)` (also defined in `qcham.h`) for single/double excitations
  generated from each currently-selected determinant, and screens candidates by
  `|hij * w[idet]| > cutoff` -- textbook heat-bath CI. `HeatbathExpansion` is the driver that calls
  this repeatedly to grow the determinant set.
- `sbdiag.h` -- the `sbd::gdb::diag()` orchestrator: build integrals -> init/load wavefunction ->
  Davidson -> RDM/occupation eval -> carryover selection, where `carryover_type == 2 or 3` invokes
  `HeatbathExpansion` (confirmed by reading the source: lines print `"start heatbath expansion"` /
  `"end heatbath expansion"` around the call). `--heatbath_cutoff`, `--heatbath_truncation`, and
  `--heatbath_batch_size` are real, working CLI flags parsed by `generate_sbd_data` in this same
  file and threaded through to the expansion call.
- `davidson_thrust.h` / `davidson.h` in this same folder are just the GPU/CPU Davidson
  diagonalizers `sbd::gdb::diag()` calls after each expansion round -- no selection logic of their
  own. (The name `gdb` stands for "General Determinant Basis," per the file's own doc comment --
  unrelated to the GNU Debugger.)

**Where it is invoked from:** a complete, standalone, buildable sample app at
`apps/chemistry_gdb_selected_basis_diagonalization/` (its own `main.cc`, `Makefile`, `CMakeLists.txt`,
`Configuration`, `README.md`, and a worked `run.sh` example against a bundled `fcidump_Fe4S4.txt` --
note: that bundled FCIDUMP is a *different* file from our own `fe4s4_zhendongli.txt`, confirmed by
differing sha256 checksums; same molecule, different active-space source). Per the app's `main.cc`,
one call to `sbd::gdb::diag()` performs **one Davidson solve followed by one heat-bath-expansion
round** -- the same single-round shape as our own `diag`/`diag_uhf` binaries. A genuine multi-round
HCI outer loop requires repeated invocations, feeding each round's `--carryoverfile` output back in
as the next round's `--detfiles` -- exactly the pattern `run_recover.py` already implements for
multi-step SQD recovery, just pointed at this binary instead.

**Where it is NOT invoked from -- confirmed, not assumed:** `algorithms/sbd/native/main.cc` (what
`build_sbd_fugaku.sh`/`build_sbd_gpu.sh`/etc. actually compile) calls `sbd::tpb::diag(...)`
exclusively. `sbd::tpb`'s own `sbdiag.h` takes the determinant lists as fixed input and goes
straight to integral construction and the Davidson solve -- no expansion or coupling-filter step of
any kind. A direct grep of every header under `include/sbd/chemistry/tpb/` for `heatbath` returns
zero matches (verified on ROQUO). The `gdb` namespace's heat-bath machinery is therefore
**vestigial with respect to the pipeline this repo actually runs** -- upstream sample code sitting
unused alongside the production `tpb` code, not a fix that was tried and abandoned inside `sqd.py`.

## 4. Prior related attempts, and why they stopped short

- `docs/2026-08-04/README.md` ("Idea 1 -- Multi-reference S+D with Heat-Bath/Epstein-Nesbet
  screening") independently proposes exactly this class of fix and explicitly shelves it: per-
  candidate Hamiltonian matrix-element evaluation is the classical selected-CI cost that SQD's
  frequency-driven design is built to avoid.
- `sqd.py` has an opt-in `hci_boost` path (`_hci_boost_strings`/`_excitation_amplitude`) that
  computes real driving-integral magnitudes (`|h1[p,q]|` for singles,
  `|g_psqt - g_ptqs|` for doubles) and ranks single/double-connected children of high-weight
  carryover determinants by an importance score. This is genuine Slater-Condon-flavored selection,
  implemented in pure Python. It improved a small-scale test (+21.7 mHa at $d=10^7$) but stalled at
  production scale ($d=10^9$: pure-Python candidate generation ran 31+ minutes with 0% GPU
  utilization) -- on record as "do NOT productionize" pending a GPU/C++ implementation.

Both prior attempts independently reached the same conclusion this note reaches from the `sbd`
source: the fix needs to happen in compiled code close to the solver, not in Python. The `gdb`
namespace's `expansion.h`/`sbdiag.h` is exactly that code, already written, already using the
correct Slater-Condon evaluators -- it has just never been connected to anything this repo runs.

## 5. What a correct fix looks like (and what it does NOT look like)

A physically correct heat-bath/importance filter keeps a candidate determinant -- regardless of
its excitation rank from HF -- if it is strongly coupled ($|H_{ij} \cdot c_j|$ large) to a
determinant already in the subspace, and discards a weakly-coupled candidate regardless of rank.
It is not a rank-based filter ("keep singles/doubles, drop triples+"); rank-from-HF and
Slater-Condon pairwise coupling are different quantities, and only the latter is the rigorous
selection criterion. Any implementation should screen by coupling magnitude, not by excitation
class.

## 6. Status / next steps

- **Not yet built or tested in this project.** `apps/chemistry_gdb_selected_basis_diagonalization`
  has never been compiled here; no `gdb_diag`-style binary exists anywhere on ROQUO or locally.
  Confirmed CPU-only (MPI + OpenMP) -- the upstream Makefile/CMakeLists ship no GPU/Thrust build
  target for this app, unlike the `tpb`-namespace binaries this repo already builds for GPU.
- A build script (`algorithms/sbd/native/build_sbd_gdb_hci_roquo.sh`) exists in this repo to build
  it, but building requires a compute-node allocation on ROQUO (`hpcx/2.50` is unavailable on the
  login node) -- blocked as of this writing on available billing-point quota (both live SQD chains
  hold the current allocation).
- Planned first step once quota is available: build `gdb_diag`, run it against our own
  `fe4s4_zhendongli.txt` FCIDUMP and BS-UHF reference (not the bundled sample FCIDUMP) with
  `--carryover_type 2` or `3` and real `--heatbath_cutoff`/`--heatbath_truncation` values, and
  compare the resulting energy/subspace composition against the SQD trajectory at a matched
  subspace dimension, as a smoke test before writing any outer-loop driver or Prefect wiring.
