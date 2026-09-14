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

**Not yet done (at time of writing above):** running against the full production-scale sample
pool or a larger determinant subset (this pass deliberately used 200 shots as a first
live-binary check); a multi-round HCI outer-loop driver; and any comparison of the resulting
energy/subspace composition against the SQD trajectory at a matched subspace dimension.

## 7. Real multi-round HCI convergence study (starting from the BS-UHF HF determinant)

Following collaborator guidance (external review of the HCI approach): don't seed from SQD
sample-pool determinants for the standard HCI reference; start from the **single BS-UHF
Hartree-Fock determinant** and let heat-bath expansion build the variational space outward from
there, sweeping the heat-bath cutoff and checking energy convergence rather than fixing a
determinant-count target up front. `--heatbath_truncation` (a `|c|^2`-weight threshold, not a
determinant-count cap -- see the code comment in `carryover.h`'s `WeightTruncation`) left at
`0.0` throughout, i.e. no pre-truncation before expansion, matching the collaborator's own stated
practice. Davidson `--tolerance 1.0e-5`, `--iteration 50` (generous headroom, not fixed low).

**Single-determinant sanity check** (`gdb_diag_uhf`, `--heatbath_cutoff 1.0e-3`, seed = the
72-bit BS-UHF HF determinant string): reproduced `E = -327.0809169678712 Ha`, matching the
independently-computed BS-UHF reference energy (`-327.08091697 Ha`, `<S^2>=8.877`) to 10
decimal places -- confirms the whole HF-seed -> merged-FCIDUMP -> `gdb_diag_uhf` pipeline is
correct before committing to a real multi-round run.

### Stage 1: fixed cutoff = 1e-4, single node

**Job**: PJM 50884271, `#PJM -L "node=1" -L "rscgrp=small" -L "elapse=07:00:00" --mpi
"max-proc-per-node=16"` -- 16 MPI ranks total, `--b_comm_size 2 --t_comm_size 1` (so
`h_comm_size = 16/(2*1) = 8`). Each round re-invokes `gdb_diag_uhf` fresh (no
`--loadname`/`--savename` in this stage) against the previous round's deduplicated
`--carryovername` output as the next round's `--detfiles`; script:
`algorithms/sbd/native/../../../run_hci_multiround.sh` (deployed at
`/vol0206/data/ra010014/u14924/u14924_space/gdb_hci/run_hci_multiround.sh` on Fugaku).

| Round | n_det in | n_det out | Energy (Ha) | ΔE vs prev round (mHa) | Wall (s) | Davidson (s) | Heatbath (s) |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 53,169 | -327.0809169678712 | -- | 30.1 | 0.005 | 0.19 |
| 1 | 53,169 | 143,704 | -327.1605311509047 | -79.63 | 365.8 | 109.3 | 216.8 |
| 2 | 143,704 | 239,932 | -327.1990585145335 | -38.53 | 1054.0 | 404.3 | 588.6 |
| 3 | 239,932 | 257,143 | -327.2049636125996 | -5.91 | 1868.0 | 800.8 | 981.0 |

Basis growth is already slowing sharply by round 3 (+67% round 1->2, only +7.2% round 2->3),
consistent with approaching saturation at this cutoff -- most Slater-Condon-coupled candidates
above the `1e-4` threshold have already been found. Reference points: DMRG (SU2, D=4000,
block2) = **-327.239 Ha**; best SQD checkpoint (Miyabi-G, d=2e10, step 85) = **-327.234400 Ha**.
Round 3 sits **34.0 mHa above DMRG** and **29.6 mHa above best SQD**, down from 79.6 mHa (round
1) above the BS-UHF starting point -- HCI is recovering correlation energy monotonically and is
closing on both references, at a real per-round compute cost that roughly doubles round over
round as the basis grows (heatbath-expansion time dominates total wall time throughout, not
Davidson).

### Stage 2: tighter cutoff = 1e-5

Once round 5 visibly saturated at `1e-4` (259,139 -> 259,429 determinants, +0.11%; final
energy **-327.205577 Ha**, see the table above extended through round 5 in the timing log),
Stage 2 continued at `--heatbath_cutoff 1.0e-5`, seeded from `1e-4`'s final detfile -- cheaper
than restarting from the bare HF determinant, since `1e-5` mostly adds newly-eligible
smaller-coupling determinants on top of what `1e-4` already found.

**Round 7** (still on Stage 1's validated 1-node/16-rank/`b_comm_size=8` config, `rscgrp=small`):
ran successfully. Heatbath expansion at the tighter cutoff grew the basis from **259,429 to
7,719,923 determinants** (a ~30x jump, as expected from a 10x-tighter cutoff) in
`wall=3971.2s` (`davidson=2653.1s`, `heatbath=1157.8s`). Davidson's energy for this round
(diagonalizing the *old*, still-259k, basis before the new expansion is folded in) reproduced
`-327.2055877378335 Ha`, matching round 5's converged 1e-4 value as expected.

**Round 8 -- real OOM, config scale-out does NOT fix it:** diagonalizing across the new
7.72M-determinant basis genuinely does not fit in the memory configurations tried so far:

| Attempt | Nodes | Ranks | `b_comm_size` | Memory | Result |
|---|---|---|---|---|---|
| 1 | 1 (Fugaku) | 16 | 8 | Fugaku per-node default | **OOM**, exit 137 (SIGKILL) at 71s |
| 2 | 1 (ROQUO `qr08n01`) | 36 | 36 | 400GB (SLURM "1/4 size" tier) | **Degraded**: 1 real `oom_kill` event (task 27), only 6/36 ranks still alive/printing after; job left in a zombie `RUNNING` state rather than terminating |
| 3 | 1 (ROQUO `qr08n01`) | 144 | 144 | 1.6TB (full node, `--mem=1600000` -- SLURM's real usable ceiling, ~100GB below `scontrol`'s reported `RealMemory=1715000`) | **Worse degradation**: 2 `oom_kill` events, **43/144 ranks** OOM-killed, same zombie-`RUNNING` pattern |

**This rules out "just needs more total memory."** Going from 36 to 144 ranks on the *same*
physical node (same 1.6TB ceiling either way) made the failure more severe (1 rank OOM'd at 36
ranks vs. 43 ranks OOM'd at 144), which is the opposite of what should happen if `b_comm_size`
were genuinely sharding the 7.72M-determinant basis proportionally across ranks -- more shards
should mean *less* memory per rank, not more failures. The real implication: something in the
current `gdb_diag_uhf`/PR #87 heatbath-expansion or Davidson-setup path is replicating
state whose size scales with the *whole* basis (or with rank count itself) on every rank,
rather than genuinely partitioning it by `b_comm`, at this basis size. This is a real
`gdb_diag_uhf`-level scaling gap, not a resource-availability problem -- fixing it would need
source-level investigation into what `--b_comm_size` actually shards during `make diagonal
term`/heatbath at this scale, out of scope for this pass.

**Also confirmed real, separately:** in every OOM case, the SLURM/PJM job stayed reported as
`RUNNING`/`R` well after the OOM kill(s) rather than terminating -- a real robustness gap
(missing collective-failure propagation) worth flagging upstream, since it means job status
alone is not a reliable signal that a run is healthy once any rank has died.

**Multi-node scale-out on Fugaku's `large` queue was also explored but never tested against
round 8 directly**, because `rscgrp=large`'s real system-wide contention (confirmed via
`pjstat -A`: other users' queued jobs at this account's node-priority tier include 256-node,
4096-node, and even a single ~36,864-node request competing for the same pool) produced
multi-day queue estimates (16 nodes/`small`: ~1 week; 2 nodes/`small`: ~9 days; 385
nodes/`large`, its real node floor confirmed via a live PJM rejection at `node=4`: ~3.5 days)
regardless of node count requested -- `small` was backlogged independent of size, and `large`'s
floor of 385 nodes was still a multi-day wait. Checked all 3 real allocation groups this
account belongs to (`ra010014`, `hp240496`, `trial`) via `pjacl -g <group>`: identical static
priority/fairshare policy (127/0/100) across all three, and the queued job's own `pjstat -v`
priority showed the plain default `127` with an empty `REASON` field -- confirming the wait is
genuine system-wide contention on `large`, not a group-specific deprioritization fixable by
switching groups. The 385-node job was cancelled once the round-8 OOM was understood to be a
config/code issue rather than something more nodes would resolve.

**Checkpoint/restart infrastructure built for this stage** (real, tested, but not what
ultimately blocked round 8 -- the OOM happens before a checkpoint would even be relevant):
every round also passes `--savename wavefunction_1e5_round<N>.dat`, persisting the actual
Davidson wavefunction (`sbd::SaveWavefunction`/`LoadWavefunction` in `caop/basic/restart.h`)
rather than only the determinant list. A resubmitted job passes the previous round's saved
wavefunction back in via `--loadname`, warm-starting Davidson instead of re-solving from a cold
guess. Verified against the actual `LoadWavefunction` source that this is safe across a
growing/reordered basis: it does a per-determinant `lower_bound` lookup against the *current*
basis, carries over the coefficient for every determinant present in both the old and new
basis, leaves newly-added determinants at a zero initial guess, and re-normalizes -- correct for
exactly the round-to-round scenario here (basis strictly grows and is freshly sorted each
round). One real constraint, and a real bug this session caught before it could bite: the save
file is sharded by `mpi_rank_b` (`statefilename(name, mpi_rank_b)`), so a reload is only valid
if `--b_comm_size` matches between the save and load runs -- the driver script's state file now
records `b_comm_size` alongside the wavefile path and drops a stale `--loadname` (cold-starting
Davidson instead) whenever it doesn't match the current run's value, rather than silently
loading mismatched shards.

A small persistent `hci_1e5_state.txt` (`last_completed_round,detfile,wavefile,b_comm_size`)
drives resume logic: on start, if the state file exists, resume from `last_completed_round+1`
using its recorded detfile/wavefile; if a round's `gdb_diag_uhf` call exits non-zero, the state
file is *not* advanced, so a resubmit retries that same round rather than skipping past a
failure. If the elapse budget runs low (within 40 min of the 8h limit) before finishing the next
round, the script self-resubmits (`pjsub "$SELF_SCRIPT"`) and exits cleanly. Script:
`run_hci_multiround_1e5.sh`, staged at
`/vol0206/data/ra010014/u14924/u14924_space/gdb_hci/run_hci_multiround_1e5.sh`.

**Status: paused here.** The Stage 1 (`1e-4`) result (`-327.205577 Ha`, 33.4 mHa above DMRG,
28.8 mHa above best SQD) stands as the real, reproducible HCI-vs-DMRG/SQD data point from this
investigation. Pushing to `1e-5` at this basis size needs either a source-level fix to
`gdb_diag_uhf`'s memory scaling, or a fundamentally different multi-node distribution strategy
than tried here -- not further config/node-count tuning.
