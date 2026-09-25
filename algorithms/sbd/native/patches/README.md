# sbd RDM opposite-spin sign-fix patches

The native SBD source trees (`algorithms/sbd/native/sbd/`, `.../sbd_mpi/`) are **git clones of
upstream** and are `.gitignore`d by qcsc-prefect (the build scripts `git clone` them on demand).
We do **not** commit into those clones. Instead the fix lives here as a patch and the build
scripts re-apply it after cloning. Permanent upstream adoption is tracked separately via PRs to
the upstream repos.

## What the patch fixes

Opposite-spin (ab/ba) 2-RDM double-excitation **sign bug** in `TwoDiffCorrelation`. The sorted-index
pairing `(I=min(i,j), A=min(a,b))` with two parities on the unchanged bra determinant is correct for
same-spin but wrong for opposite-spin (it pairs across spins and ignores the intermediate
determinant). The fix uses spin-consistent pairing (alpha-cr<->alpha-an, beta-cr<->beta-an) with the
intermediate determinant, expressed purely on the bra det:

```
s  = parity(det, aCr..aAn)
if bCr in (aCr,aAn): s *= -1
if bAn in (aCr,aAn): s *= -1
s *= parity(det, bCr..bAn)
```

Verified: Python replica `|Δr2ab|_max = 6.9e-18`; GPU binary `diag-gpu_uhf_diag` on job062419
(NORB=20) `Delta(E_recon - E_davidson): +0.937 mHa -> -6.25e-13 Ha` (job 164204 -> 164217, 2026-09-26).
See `note/24_sqd_rdm_signbug_fix_resolved.md`.

## Patches and provenance

| patch | target clone | upstream | base HEAD | files |
|---|---|---|---|---|
| `sbd-rdm-opposite-spin-sign.patch` | `sbd/` (GPU) | `github.com/r-ccs-cms/sbd.git` `main` | `21f73b4` | `basic/correlation_thrust.h` (GPU 本命), `basic/correlation.h` (別経路の保険) |
| `sbd_mpi-rdm-opposite-spin-sign.patch` | `sbd_mpi/` (MPI) | `github.com/rwakizaka/sbd.git` `non-cuda-aware-mpi` | `034743e` | `basic/correlation_thrust.h` |

`basic/correlation_thrust.h::CorrelationKernels::TwoDiffCorrelation` is the kernel the GPU
`tpb::diag` RDM path actually uses (via `tpb/correlation_thrust.h::CorrelationAlphaBeta`).
`basic/correlation.h::TwoDiffCorrelation` is a separate path unused by GPU tpb — patched for safety.

## Applying (done automatically by the build scripts)

```bash
patches/apply_rdm_patches.sh sbd     patches/sbd-rdm-opposite-spin-sign.patch
patches/apply_rdm_patches.sh sbd_mpi patches/sbd_mpi-rdm-opposite-spin-sign.patch
```

Idempotent: re-running skips an already-applied patch. If upstream HEAD moves and the patch no
longer applies, regenerate it:

```bash
# after re-editing the fix in the clone:
cd sbd && git diff -- include/sbd/chemistry/basic/correlation_thrust.h \
                       include/sbd/chemistry/basic/correlation.h \
     > ../patches/sbd-rdm-opposite-spin-sign.patch
```

## Permanent upstream adoption (parallel track)

Propose the same fix upstream via fork + PR (push access not required):
- `github.com/r-ccs-cms/sbd` (GPU 本命)
- `github.com/rwakizaka/sbd` (MPI fork), if that fork is the long-term MPI source.
