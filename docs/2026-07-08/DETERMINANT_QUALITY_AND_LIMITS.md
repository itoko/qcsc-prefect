# Determinant subspace: sizes, singles+doubles content, and the sample-quality ceiling

## Measured determinant profiles (Fe4S4 72q, per spin)

From the `[diag] ... dets:` log lines. `n` = determinants in that spin's list; `HF` = Hartree-Fock;
`singles+doubles` = determinants ≤2 excitations from HF; `higher(>2)` = triples and beyond;
`max_exc` = the most-excited determinant's excitation level.

| Run | step | n (per spin) | HF | singles+doubles | higher(>2) | S+D fraction |
|---|---|---|---|---|---|---|
| Baseline 9M (raw sample) | 1 | 3000 | 1 | **198 (α) / 160 (β)** | 2801 / 2839 | ~6 % |
| Baseline 9M (raw sample) | 6 | 3000 | 1 | 591 | 2408 | ~20 % |
| Baseline 9M (raw sample) | 12 | 3000 | 1 | **810** | 2189 | ~27 % |
| Extended 3e7 (raw sample) | 1 | 5477 | 1 | 344 / 256 | 5132 / 5220 | ~6 % |
| **Full CISD injection** | 1 | 12922 | 1 | **12879** | 42 | **~99.7 %** |

## What this shows

**1. The raw quantum sample contains very few genuine singles/doubles.** At step 1, only ~198 of
3000 determinants (≈6 %) are S+D; the other ~2800 are high-excitation configurations (`max_exc=9`)
— i.e. noise. The singles/doubles are the determinants that carry almost all the correlation
energy, and the noisy device buries them among high-excitation junk.

**2. Recovery does help — but slowly and only up to a point.** Over 12 recovery steps the S+D count
climbs 198 → 591 → 810 (6 % → 27 %). Configuration recovery re-weights toward the physically
important configurations, so each self-consistent pass recovers a few more real S+D. This is exactly
why the energy keeps descending with iterations. **But it plateaus:** you can only re-discover S+D
that are *reachable* from what the sample actually contained; recovery cannot invent a determinant
the noisy distribution never produced with any weight. The raw run flattens around −326.79.

**3. CISD injection removes the bottleneck by construction.** It forces all 12,879 S+D into the
subspace (99.7 % of the list), independent of what the device sampled. That is why one injected step
(−326.805) beats 45 raw recovery steps (−326.787): it hands recovery the full S+D floor instead of
making it slowly claw a fraction of them out of noise.

## The key physical point (your intuition, stated precisely)

**There is a hard accuracy ceiling set by the *level* of excitations available, not by how many
determinants you pile in.**

- A subspace spanned by {HF + all singles + all doubles} has a fixed lowest eigenvalue — the **CISD
  energy** for that reference. For this Fe4S4 reference that is ≈ −326.805 Ha. No amount of
  additional determinants of the *same or lower* excitation level, more recovery iterations, or more
  K-batches can go below it — they are all still inside (or a subset of) the CISD space.
- To go lower you must add determinants of a **higher excitation class** (triples, quadruples, …)
  that carry real weight in the true wavefunction.
- Here is the trap for a noisy sample: the device *does* emit lots of "higher(>2)" determinants —
  but they are dominated by **measurement noise**, not by the true wavefunction's triples. We tested
  this directly: adding ~7000 sampled higher-excitation determinants on top of full CISD moved the
  energy by **< 1 mHa**, and replacing half the doubles with sampled higher-excitations made it
  **281 mHa worse**. So the sample's higher excitations are noise, and recovery on noise cannot
  manufacture the correlation that only *genuine* triples would provide.

**Therefore:** for a fixed-quality (noisy) sample, SQD + recovery + injection converges to roughly
the CISD ceiling and stops. Pushing past it does **not** come from more excitations pulled out of
the same noisy sample (more doubles, more recovery steps, bigger subspace) — it requires either
(a) genuinely higher-quality bitstrings (lower noise / more shots so real triples emerge above the
noise floor), (b) deterministic higher-order excitations added classically (CISDT+, expensive), or
(c) a better reference via orbital/circuit optimization so more correlation folds into the S+D of
the rotated basis.

## What to ask someone claiming "better data" / a lower energy

Ask for the numbers that separate *real improvement* from *noise / basis artifacts*:

1. **What is the subspace excitation profile?** For their best point: n determinants, and the
   breakdown HF / singles+doubles / higher(>2), with `max_exc`. A lower energy with the *same* S+D
   ceiling and only more high-excitation configs is suspicious (likely noise or over-counting).
2. **Is it below the CISD energy of the same reference — and by a mechanism that adds genuine
   triples?** If they claim below-CISD, what determinant class provides it? Show the triples' weight.
3. **Sample-preservation / shot-retention rate** (kept ÷ raw shots after post-selection). Low
   retention with a low energy suggests the answer is being carried by a few determinants — check
   sensitivity.
4. **K-batch spread (min vs mean vs std) at the best point.** If the batches disagree by many mHa,
   the "best" is a lucky draw, not a converged variational estimate. Ours had spread ≈ 0 (converged).
5. **Same reference and active space?** Confirm norb, nelec, MS2, the *exact* FCIDUMP/integrals, and
   frozen-core choices. A different active space or basis makes energies incomparable.
6. **Variational sanity:** SQD energy must be ≥ the exact ground state of the *chosen* subspace and
   ≥ FCI of the active space. Is their number above a trusted near-exact reference (DMRG/FCI) for the
   same space? If it's below DMRG for the same active space, something is wrong.
7. **Reproducibility across seeds/samples**, and convergence vs subspace dimension (does it keep
   dropping as dim grows, or has it plateaued?). A converged plateau at the CISD level is the honest
   signature; a still-dropping curve means the subspace is undersized, not that the method is better.
8. **Reference anchors reported alongside:** UHF, UCCSD, CCSD(T), and ideally DMRG/FCI for the same
   space — so the claimed number is placed on an absolute scale, not just "lower than before."

The single most diagnostic question: **"Show me the excitation-level breakdown of your best
subspace and where the energy below CISD is coming from."** Real gains come with genuine
higher-excitation determinants carrying real weight; noise-driven "gains" don't survive that check.
