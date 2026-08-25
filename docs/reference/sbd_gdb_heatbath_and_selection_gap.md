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

**Built and smoke-tested on Fugaku, against real BS-UHF integrals and a real sampled
determinant subset -- both confirmed working.** ROQUO was not used for the build/run (a
large-node-count maintenance reservation left only 9 free nodes at the time); Fugaku's
`mpiFCCpx` toolchain built both the plain (`gdb_diag`) and `-D_UHF` (`gdb_diag_uhf`) variants of
`apps/chemistry_gdb_selected_basis_diagonalization` on the first attempt via
`algorithms/sbd/native/build_sbd_gdb_hci_{,uhf_}fugaku.sh` (ROQUO scripts exist as siblings for
whenever node quota frees up).

Also fetched and built against upstream PR
[r-ccs-cms/sbd#87](https://github.com/r-ccs-cms/sbd/pull/87) ("Fix and accelerate GDB heatbath
expansion", open at time of writing) -- a real correctness fix for `HeatbathExpansion` under
replicated MPI communicator layouts, plus a ~32x-faster integral-driven expansion path
(`carryover_type=3`). The smoke test below ran against this PR's code, not plain upstream `main`.

**The real single-basis-FCIDUMP gap turned out narrower than the "gdb never sees UHF integrals"
framing above might suggest.** `oneInt`/`twoInt`'s `-D_UHF` storage layout and every heatbath/
Davidson call site in `expansion.h`/`qcham.h` are already spin-backing-store-transparent -- the
missing piece was purely a build flag (`gdb`'s `Makefile`/`Configuration` never passed `-D_UHF`)
plus a file-format bridge, not new solver logic. Two new scripts close that gap:

- `examples/fe4s4_hci_from_bsuhf_reference/merge_bsuhf_to_uhf_fcidump.py` -- merges the
  `prepare_bsuhf_fcidump.py`-produced `.alpha.fcidump`/`.beta.fcidump`/`.mixed.npz` triple into
  the single interleaved-spin-orbital FCIDUMP `-D_UHF`'s `SetupIntegrals` reads, reusing
  `solver_job.py`'s already-tested `_write_uhf_fcidump` writer as a pure format bridge. Verified
  against the real Fe4S4 BS-UHF reference: round-trips losslessly and reconstructs
  `E=-327.08091697 Ha` to 8 decimal places, exactly matching the known BS-UHF energy.
- `examples/fe4s4_hci_from_bsuhf_reference/pool_to_gdb_detfile.py` -- converts a real subset of
  the production 5M-shot sample pool into `gdb`'s plain-ASCII interleaved-bit `--detfiles`
  format (one merged determinant per sampled shot, not a Cartesian product of independently
  -drawn alpha/beta values). Caught a real bug during verification: the raw saved pool is
  genuinely unfiltered hardware data (most shots do not have the correct particle number -- only
  245 of ~2M unique bitstrings in `fe4s4_uhf_5M_zhendongli.npz` survive Hamming-weight
  post-selection), so post-selection had to be added before the top-N-by-probability sampling
  step.

**Live-binary result** (`gdb_diag_uhf`, real `fe4s4_bsuhf.uhf.fcidump`, 200 real post-selected
determinants from the production pool, `--carryover_type 0` then `2`): both runs completed
cleanly. Baseline Davidson energy `-323.4286516396394 Ha` (physically sensible for a 200
-determinant subset of the full ~173,205x173,205 SQD subspace -- well above the full
`-327.08 Ha` ground state, not NaN/inf/wildly wrong). Heat-bath run reproduced the identical
energy and printed real `"start heatbath expansion"`/`"end heatbath expansion"` timing (3.8ms),
confirming the expansion machinery runs correctly against genuine UHF integrals end-to-end. The
carryover set came back empty at this tiny scale/cutoff combination -- expected, not a bug (same
behavior seen on the bundled toy example).

**Known landmine, explicitly deferred:** `main.cc`'s post-solve RDM-derived diagnostic energy
printout (`"one-body energy"`/`"two-body energy"` lines) hardcodes an RHF-style combination of
the alpha/beta RDM blocks and would be wrong for a genuine UHF run -- only trust the
Davidson-solved `"sbd: Energy = ..."` line, which comes from the already-UHF-aware Hamiltonian
machinery and is unaffected. A real fix belongs upstream in `r-ccs-cms/sbd`.

**Not yet done:** running against the full production-scale sample pool or a larger determinant
subset (this pass deliberately used 200 shots as a first live-binary check); a multi-round HCI
outer-loop driver (`run_gdb_hci_recover.py`, referenced in the build scripts' comments but not
yet written) that would feed each round's `--carryoverfile` output back in as the next round's
`--detfiles`, mirroring `run_recover.py`'s existing multi-step SQD pattern; and any comparison of
the resulting energy/subspace composition against the SQD trajectory at a matched subspace
dimension.
