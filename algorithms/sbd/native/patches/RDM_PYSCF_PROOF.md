# Numerical proof: opposite-spin 2-RDM sign fix vs. PySCF

This report documents `test_rdm_vs_pyscf.py`, the self-contained numerical proof that qcsc-prefect
commit `bb3e562` ("Fix opposite-spin 2-RDM double-excitation sign in native SBD") is correct.

## What is being proven

The native SBD kernel `TwoDiffCorrelation` used a sorted-index pairing
`(I=min(i,j) -> A=min(a,b)), (J -> B)` with two parities evaluated on the (unchanged) bra
determinant. That is correct for **same-spin** double excitations (aa/bb blocks) but wrong for
**opposite-spin** ones (ab/ba blocks): it pairs creation/annihilation operators *across* spins and
ignores the intermediate determinant, producing a wrong Fermi sign on some ab/ba 2-RDM elements.

The patch replaces the opposite-spin branch with spin-consistent pairing
(alpha-creation <-> alpha-annihilation, beta-creation <-> beta-annihilation) plus the intermediate
determinant, expressed purely on the bra determinant.

## Design of the test (no GPU, no compiled binary, zero production impact)

`test_rdm_vs_pyscf.py` is a **pure-Python + PySCF replica**. It does not build or link SBD, does not
touch the shipped RDM path, and runs anywhere `numpy`+`pyscf` are importable. It:

1. Builds a genuine **full-CI** state with PySCF (`fci.direct_uhf`) from UHF integrals and takes the
   reference spin-resolved RDMs from `make_rdm12s` (default `reorder=True`, i.e. the *pure*
   normal-ordered 2-RDM — the same 2-body object SBD accumulates). We truncate at the 2-RDM;
   PySCF's higher-order RDMs are not used.
2. Re-implements the SBD correlation kernel **line-for-line** in Python
   (`parity` / `ZeroDiffCorrelation` / `OneDiffCorrelation` / `TwoDiffCorrelation`, ported from
   `determinants.h`, `correlation.h`, `correlation_thrust.h`). The **only** thing that changes
   between the "original" and "patched" runs is the opposite-spin branch of `TwoDiffCorrelation`,
   gated by a `patched` flag. The CI vector fed to both is identical, so any difference between
   them is due solely to the sign fix.
3. Builds all four 2-RDM blocks + the 1-RDM from that CI vector with each kernel variant and
   compares element-wise, before vs. after the patch.

### The two bridging conventions (the only glue outside the kernel)

These are properties of the PySCF <-> SBD interface, not of the kernel under test:

- **CI ordering sign.** PySCF orders spin-orbitals as (all alpha ascending)(all beta ascending);
  SBD interleaves them as `2*spatial + spin`. Re-ordering one operator string into the other costs
  a per-determinant Fermi sign `(-1)^{#(p in alpha_occ, q in beta_occ : p > q)}`, folded into each
  CI coefficient in `_build_reference`. It cancels in occupation-diagonal quantities (traces,
  diagonals), which is why those are correct even without it, but it is essential for the
  off-diagonal RDM elements.
- **Block -> PySCF layout.** SBD stores block `b` at flat index `io + L*jo + L^2*ia + L^3*ja` with
  `block[io,jo,ia,ja] = <c^dag_{io,s} c^dag_{jo,t} c_{ja,t} c_{ia,s}>`. PySCF `make_rdm12s` uses
  chemist ordering `dm2[p,q,r,s] = <p^dag q r^dag s>`. For the **same-spin** blocks this maps by
  `transpose(0,2,1,3)`; for the **opposite-spin** blocks the alpha pair and beta pair sit in
  different chemist slots, so the map is `transpose(1,3,0,2)`. Both permutations were pinned by an
  independent brute-force operator build (`<a^dag_{p,a} a_{q,a} a^dag_{r,b} a_{s,b}>`), which agrees
  with PySCF `make_rdm12s` to 2e-16 and with the SBD kernel to machine zero.

> On PySCF RDM order: `make_rdm12s` implements RDMs of order 2 (and PySCF has higher orders
> elsewhere). We deliberately use only the 2-RDM and its default reordered convention, matching the
> SBD truncation at the 2-body level.

## Assertions (the "patch is proven" checklist)

For each case, with the identical CI vector:

| Quantity | patched | original |
|---|---|---|
| 1-RDM `dm1a`,`dm1b` vs PySCF | agree (~1e-16) | identical to patched (unaffected) |
| same-spin `aa`,`bb` vs PySCF | agree (~1e-16) | identical to patched (unaffected) |
| opposite-spin `ab` vs PySCF | **agree (~1e-16)** | **disagree, max\|delta\| > 1e-6** |
| `ba` == transpose(dm2ab,(2,3,0,1)) | holds | — |
| `E_recon` vs `E_davidson` | equal (< 1e-8) | deviates (> 1e-6) |
| patched `ab` != original `ab` | patch has a real effect | — |

## Cases and measured results (roquo, aarch64 compute node)

Run: `srun --jobid=<running> --overlap -N1 -n1 .venv/bin/python native/patches/test_rdm_vs_pyscf.py`

```
=== H3_asym_doublet: norb=3 nelec=(2, 1) ndet=9 ===
  E_davidson              = -1.588486422874
  E_recon (patched)       = -1.588486422874   dE=+1.55e-15
  E_recon (original)      = -1.579260320735   dE=+9.23e-03   <- bug
  max|ab_patched  - pyscf|= 0.00e+00
  max|ab_original - pyscf|= 4.98e-02          <- bug
  max|dm1a - pyscf|       = 0.00e+00
  max|aa   - pyscf|       = 2.34e-17
  [PASS]

=== OH_doublet: norb=6 nelec=(5, 4) ndet=90 ===
  E_davidson              = -74.387184744061
  E_recon (patched)       = -74.387184744061   dE=-1.42e-14
  E_recon (original)      = -74.385838174944   dE=+1.35e-03   <- bug
  max|ab_patched  - pyscf|= 3.33e-16
  max|ab_original - pyscf|= 3.83e-02           <- bug
  max|dm1a - pyscf|       = 1.74e-18
  max|aa   - pyscf|       = 2.22e-16
  [PASS]

2/2 cases passed.
```

The patched replica reproduces PySCF's 1-RDM and all four 2-RDM blocks to machine precision, and
its RDM-reconstructed energy equals the Davidson energy. The original code reproduces the exact
same 1-RDM and same-spin blocks but gets the opposite-spin block wrong by ~1e-2, and its
reconstructed energy is off by ~1-9 mHa — the same `E_recon != E_davidson` signature observed on
the GPU binary in job062419 (NORB=20: +0.937 mHa -> -6.25e-13 Ha after the fix). The bug is a pure
opposite-spin phenomenon: same-spin blocks and the 1-RDM are untouched.

Note: use asymmetric geometries. A symmetric chain (e.g. equally spaced H4) still shows the wrong
RDM elements but can make the sign error **cancel in the scalar energy**, hiding the `E_recon`
signature; the RDM element-wise check is the robust, geometry-independent proof.

## How to run

```bash
cd ~/qcsc-prefect/algorithms/sbd

# (a) Piggyback on an existing allocation — needs no billing points/account:
srun --jobid=<your running jobid> --overlap -N1 -n1 \
     .venv/bin/python native/patches/test_rdm_vs_pyscf.py

# (b) Standalone batch job (needs billing points on qc-prj-other02):
sbatch native/patches/run_rdm_pyscf_test.sbatch

# (c) Anywhere numpy+pyscf are importable:
python native/patches/test_rdm_vs_pyscf.py           # script runner, prints the table above
pytest -v native/patches/test_rdm_vs_pyscf.py        # if pytest is installed
```

The test module works with or without `pytest` (a tiny shim provides the `mark` decorators when
pytest is absent, and the `__main__` block runs the same assertions and prints the diagnostics).

## Provenance

- Commit under proof: `bb3e562`.
- GPU verification: job062419, NORB=20, `diag-gpu_uhf_diag`,
  `Delta(E_recon - E_davidson): +0.937 mHa -> -6.25e-13 Ha` (jobs 164204 -> 164217, 2026-09-26).
- Related note: `note/24_sqd_rdm_signbug_fix_resolved.md`.
- Source lines mirrored (verified 2026-09):
  `determinants.h:231` (parity), `correlation.h` (Zero/One/TwoDiffCorrelation + Correlation
  dispatch), `correlation_thrust.h:108` (device TwoDiffCorrelation).
