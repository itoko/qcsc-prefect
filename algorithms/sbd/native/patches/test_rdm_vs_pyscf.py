"""PySCF-equivalent proof of the opposite-spin 2-RDM sign fix in native SBD.

This is the numerical proof behind qcsc-prefect commit ``bb3e562`` ("Fix opposite-spin 2-RDM
double-excitation sign in native SBD"). It is a **pure-Python + PySCF replica** -- it needs no GPU
and no compiled SBD binary, so it runs anywhere ``numpy`` and ``pyscf`` are importable and has zero
effect on the production build or the shipped RDM path.

What it does
------------
1. Builds a genuine full-CI state with PySCF (``fci.direct_uhf``) and takes its reference
   spin-resolved RDMs from ``make_rdm12s``:
       (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = cisolver.make_rdm12s(ci, norb, nelec)
   Note ``make_rdm12s`` defaults to ``reorder=True``, i.e. it returns the *pure* (normal-ordered)
   2-RDM Gamma[p,q,r,s] with the same-spin delta term already removed -- the same 2-body object
   the SBD kernel accumulates. We truncate at the 2-RDM; PySCF's higher-order RDMs are not used.
2. Faithfully re-implements the native SBD correlation kernel that builds the RDMs from the CI
   vector -- ``parity`` / ``ZeroDiffCorrelation`` / ``OneDiffCorrelation`` / ``TwoDiffCorrelation``
   -- ported line-for-line from
       sbd/include/sbd/chemistry/basic/determinants.h   (parity)
       sbd/include/sbd/chemistry/basic/correlation.h     (Zero/One/TwoDiffCorrelation)
       sbd/include/sbd/chemistry/basic/correlation_thrust.h (device TwoDiffCorrelation, same logic)
   The ONLY thing that changes between "original" and "patched" is the opposite-spin branch of
   ``TwoDiffCorrelation`` (``si != sj``), gated by the ``patched`` flag. Everything else -- and in
   particular the CI vector fed in -- is identical between the two runs, so any difference between
   them is attributable solely to the sign fix.
3. Compares element-wise, before vs after the patch, the "patch is proven" checklist:
       * patched  ab/ba block  ==  PySCF dm2ab            (agree, ~1e-16)
       * original ab/ba block  !=  PySCF dm2ab            (bug reproduced, max|Δ| ~1e-2..1e-3)
       * aa, bb blocks          unchanged by the patch AND == PySCF
       * 1-RDM (dm1a, dm1b)     unchanged by the patch AND == PySCF
       * E_recon(patched)       == E_davidson             (RDM energy self-consistent)
       * E_recon(original)      != E_davidson             (the observed E_recon-E_dav gap, ~mHa)

The primary, self-contained proof is the **patched-vs-original** contrast (it needs only the shared
CI vector and the kernel, no external reference). The element-wise agreement with PySCF is an
independent cross-check that pins down the *correct* answer.

SBD conventions mirrored (verified in source, 2026-09)
------------------------------------------------------
* spin-orbital index = 2*spatial + spin,  spin 0 = alpha, 1 = beta.
* Excitation extraction (correlation.h::Correlation): for bra DetI, ket DetJ,
  c = bits set in DetI & ~DetJ (ascending), d = DetJ & ~DetI; dispatch on nc=len(c).
  The determinant handed to every routine is the *bra* DetI. Every ordered pair (I,J) with
  <= 2 differing spin-orbitals is visited once (both (I,J) and (J,I) occur), so the Hermitian
  conjugate contributions are included automatically.
* parity(det,start,end): (-1)^(# occupied bits strictly between start and end), i.e. exclusive
  of both endpoints (determinants.h:231 counts [start,end) then flips if `start` is occupied).

PySCF <-> SBD bridging conventions (the only glue outside the kernel)
--------------------------------------------------------------------
* CI ordering sign: PySCF orders spin-orbitals as (all alpha ascending) then (all beta ascending),
  while SBD interleaves them as 2*spatial+spin. Re-ordering one operator string into the other
  costs a per-determinant Fermi sign (-1)^{#(p in alpha_occ, q in beta_occ : p>q)}. We fold that
  sign into the CI coefficient in ``_build_reference`` so the replica CI matches PySCF's phase.
  (It cancels in occupation-diagonal quantities, which is why traces/diagonals are sign-agnostic,
  but it is essential for the off-diagonal RDM elements.)
* Block -> PySCF layout: SBD stores block b at flat io + L*jo + L^2*ia + L^3*ja with
  block[io,jo,ia,ja] = <c^dag_{io,s} c^dag_{jo,t} c_{ja,t} c_{ia,s}>. PySCF ``make_rdm12s`` uses
  chemist ordering dm2[p,q,r,s] = <p^dag q r^dag s>. For the same-spin blocks (aa/bb) this maps by
  transpose (0,2,1,3); for the opposite-spin blocks (ab/ba) the alpha and beta index pairs sit in
  different chemist slots and the map is transpose (1,3,0,2). See ``_same_to_pyscf`` /
  ``_opp_to_pyscf``.
"""
from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:  # allow running as a plain script where pytest is absent
    class _PytestShim:
        class mark:
            @staticmethod
            def skipif(cond, reason=""):
                return lambda f: f

            @staticmethod
            def parametrize(argnames, argvalues, ids=None):
                return lambda f: f

    pytest = _PytestShim()  # type: ignore

try:
    from pyscf import gto, scf, ao2mo, fci
    from pyscf.fci import cistring
    _HAVE_PYSCF = True
except Exception:  # pragma: no cover - environment without pyscf
    _HAVE_PYSCF = False


# =============================================================================
# Faithful Python port of the native SBD correlation kernel
# =============================================================================
def _popcount(x: int) -> int:
    return bin(x).count("1")


def _set_bits(x: int) -> list[int]:
    """Ascending list of set bit positions (matches the bit-scan order in Correlation)."""
    out = []
    while x:
        low = x & -x
        out.append(low.bit_length() - 1)
        x ^= low
    return out


def _parity(det: int, start: int, end: int) -> float:
    """Mirror determinants.h::parity: (-1)^(occupied bits strictly between start and end).

    Counts occupied bits in [start, end) then flips the sign once if `start` itself is occupied,
    which is exactly "strictly between start and end" (exclusive both ends). Requires start<=end.
    """
    if start > end:
        raise ValueError("start > end")
    mask = ((1 << end) - 1) ^ ((1 << start) - 1)   # bits [start, end)
    cnt = _popcount(det & mask)
    s = -1.0 if (cnt & 1) else 1.0
    if (det >> start) & 1:
        s *= -1.0
    return s


class _RDMKernel:
    """Accumulates onebody[2] and twobody[4] exactly like the SBD kernel.

    twobody block order (s + 2*t): 0=aa, 1=ba, 2=ab, 3=bb.
    twobody flat index: io + L*jo + L^2*ia + L^3*ja  = <c^dag_{io,s} c^dag_{jo,t} c_{ja,t} c_{ia,s}>.
    onebody flat index: oi + L*oa                    = <c^dag_{oi,s} c_{oa,s}>.
    """

    def __init__(self, norb: int, patched: bool):
        self.L = norb
        self.patched = patched
        self.onebody = [np.zeros(norb * norb) for _ in range(2)]
        self.twobody = [np.zeros(norb ** 4) for _ in range(4)]

    def _t_add(self, blk: int, io: int, jo: int, ia: int, ja: int, val: float) -> None:
        L = self.L
        self.twobody[blk][io + L * jo + L * L * ia + L * L * L * ja] += val

    # --- ZeroDiffCorrelation (correlation.h) : diagonal, I == J -----------------------------
    def zero_diff(self, DetI: int, WeightI: float) -> None:
        L = self.L
        w2 = WeightI * WeightI  # SquaredNorm (real)
        closed = [b for b in range(2 * L) if (DetI >> b) & 1]
        for a in range(len(closed)):
            oi, si = closed[a] // 2, closed[a] % 2
            self.onebody[si][oi + L * oi] += w2
            for bb in range(a + 1, len(closed)):
                oj, sj = closed[bb] // 2, closed[bb] % 2
                self._t_add(si + 2 * sj, oi, oj, oi, oj, w2)
                self._t_add(sj + 2 * si, oj, oi, oj, oi, w2)
                if si == sj:
                    self._t_add(si + 2 * sj, oi, oj, oj, oi, -w2)
                    self._t_add(sj + 2 * si, oj, oi, oi, oj, -w2)

    # --- OneDiffCorrelation (correlation.h) : single excitation i -> a ----------------------
    def one_diff(self, DetI: int, WeightI: float, WeightJ: float, i: int, a: int) -> None:
        L = self.L
        sgn = _parity(DetI, min(i, a), max(i, a))
        oi, si = i // 2, i % 2
        oa, sa = a // 2, a % 2
        coef = WeightI * WeightJ * sgn
        self.onebody[si][oi + L * oa] += coef
        for soj in [b for b in range(2 * L) if (DetI >> b) & 1]:
            oj, sj = soj // 2, soj % 2
            self._t_add(si + 2 * sj, oa, oj, oi, oj, coef)
            self._t_add(sj + 2 * si, oj, oa, oj, oi, coef)
            if si == sj:
                self._t_add(si + 2 * sj, oa, oj, oj, oi, -coef)
                self._t_add(sj + 2 * si, oj, oa, oi, oj, -coef)

    # --- TwoDiffCorrelation (correlation.h / correlation_thrust.h) : double excitation ------
    def two_diff(self, DetI: int, WeightI: float, WeightJ: float,
                 i: int, j: int, a: int, b: int) -> None:
        L = self.L
        I, J = min(i, j), max(i, j)
        A, B = min(a, b), max(a, b)
        sgn = _parity(DetI, min(I, A), max(I, A))
        sgn *= _parity(DetI, min(J, B), max(J, B))
        if A > J or B < I:
            sgn *= -1.0
        oi, si = I // 2, I % 2
        oa, sa = A // 2, A % 2
        oj, sj = J // 2, J % 2
        ob, sb = B // 2, B % 2
        coef0 = WeightI * WeightJ  # Conjugate(WeightI)*WeightJ, real

        if si != sj:
            # Opposite-spin double (ab/ba blocks).
            if not self.patched:
                # ORIGINAL (buggy): sorted-index scatter with the two same-spin guards, using sgn.
                if si == sa:
                    self._t_add(si + 2 * sj, oa, ob, oi, oj, coef0 * sgn)
                    self._t_add(sj + 2 * si, ob, oa, oj, oi, coef0 * sgn)
                if si == sb:
                    self._t_add(si + 2 * sj, oa, ob, oj, oi, -coef0 * sgn)
                    self._t_add(sj + 2 * si, ob, oa, oi, oj, -coef0 * sgn)
            else:
                # PATCHED: spin-consistent pairing + intermediate determinant, expressed on DetI.
                aCr = i if (i % 2 == 0) else j
                bCr = j if (i % 2 == 0) else i
                aAn = a if (a % 2 == 0) else b
                bAn = b if (a % 2 == 0) else a
                p1, q1 = min(aCr, aAn), max(aCr, aAn)
                p2, q2 = min(bCr, bAn), max(bCr, bAn)
                s = _parity(DetI, p1, q1)
                if p1 < bCr < q1:
                    s *= -1.0
                if p1 < bAn < q1:
                    s *= -1.0
                s *= _parity(DetI, p2, q2)
                oaC, oaA = aCr // 2, aAn // 2
                obC, obA = bCr // 2, bAn // 2
                self._t_add(2, oaC, obC, oaA, obA, coef0 * s)
                self._t_add(1, obC, oaC, obA, oaA, coef0 * s)
        else:
            # Same-spin double (aa/bb): original scatter is correct (patch does not touch it).
            if si == sa:
                self._t_add(si + 2 * sj, oa, ob, oi, oj, coef0 * sgn)
                self._t_add(sj + 2 * si, ob, oa, oj, oi, coef0 * sgn)
            if si == sb:
                self._t_add(si + 2 * sj, oa, ob, oj, oi, -coef0 * sgn)
                self._t_add(sj + 2 * si, ob, oa, oi, oj, -coef0 * sgn)

    # --- driver: iterate every ordered determinant pair, dispatch by #differences -----------
    def build(self, dets: list[int], weights: np.ndarray) -> None:
        M = len(dets)
        for I in range(M):
            DI = dets[I]
            WI = float(weights[I])
            for Jd in range(M):
                DJ = dets[Jd]
                c = DI & ~DJ
                nc = _popcount(c)
                if nc == 0:
                    if DI == DJ:
                        self.zero_diff(DI, WI)
                elif nc == 1:
                    d = DJ & ~DI
                    self.one_diff(DI, WI, float(weights[Jd]),
                                  _set_bits(c)[0], _set_bits(d)[0])
                elif nc == 2:
                    d = DJ & ~DI
                    cb = _set_bits(c)
                    db = _set_bits(d)
                    self.two_diff(DI, WI, float(weights[Jd]),
                                  cb[0], cb[1], db[0], db[1])
                # nc > 2 -> no 2-RDM contribution

    def blocks(self) -> dict:
        L = self.L
        return {
            "aa": self.twobody[0].reshape(L, L, L, L),
            "ba": self.twobody[1].reshape(L, L, L, L),
            "ab": self.twobody[2].reshape(L, L, L, L),
            "bb": self.twobody[3].reshape(L, L, L, L),
            "1a": self.onebody[0].reshape(L, L),
            "1b": self.onebody[1].reshape(L, L),
        }


def _same_to_pyscf(block: np.ndarray) -> np.ndarray:
    """Same-spin SBD block (aa/bb) -> PySCF make_rdm12s chemist layout.

    SBD block[io,jo,ia,ja] = <c^dag_{io} c^dag_{jo} c_{ja} c_{ia}> (all one spin). PySCF (reordered)
    dm2[p,q,r,s] = <p^dag r^dag s q>, so p=io, r=jo, q=ia, s=ja -> transpose (0,2,1,3)."""
    return np.transpose(block, (0, 2, 1, 3))


def _opp_to_pyscf(block: np.ndarray) -> np.ndarray:
    """Opposite-spin SBD block (ab or ba) -> PySCF chemist dm2ab layout.

    SBD ab-block[io,jo,ia,ja] = <c^dag_{io,alpha} c^dag_{jo,beta} c_{ja,beta} c_{ia,alpha}>. PySCF
    dm2ab[p,q,r,s] = <a^dag_{p,alpha} a_{q,alpha} a^dag_{r,beta} a_{s,beta}> puts the two alpha
    indices in slots (p,q) and the two beta indices in slots (r,s), so io->p, ia->q, jo->r, ja->s,
    i.e. transpose (1,3,0,2). (Verified element-wise against an independent brute-force build of
    <a^dag_{p,alpha} a_{q,alpha} a^dag_{r,beta} a_{s,beta}> and against PySCF make_rdm12s.)"""
    return np.transpose(block, (1, 3, 0, 2))


# =============================================================================
# Reference full-CI state from PySCF (UHF integrals)
# =============================================================================
def _reorder_sign(det: int, norb: int) -> float:
    """Fermi sign relating PySCF's (alpha-then-beta) spin-orbital order to SBD's interleaved order.

    PySCF creates |det> as (alpha creators ascending)(beta creators ascending)|0>; SBD orders all
    creators by interleaved index 2*spatial+spin. The permutation between the two is a product of
    transpositions moving each beta creator past the alpha creators of higher spatial index, giving
    (-1)^{#(p in alpha_occ, q in beta_occ : p>q)}."""
    aocc = [p for p in range(norb) if (det >> (2 * p)) & 1]
    bocc = [q for q in range(norb) if (det >> (2 * q + 1)) & 1]
    inv = sum(1 for q in bocc for p in aocc if p > q)
    return -1.0 if (inv & 1) else 1.0


def _build_reference(atom: str, basis: str, spin: int):
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = scf.UHF(mol).run()
    mo_a, mo_b = mf.mo_coeff
    norb = mo_a.shape[1]
    hcore = mf.get_hcore()
    h1a = mo_a.T @ hcore @ mo_a
    h1b = mo_b.T @ hcore @ mo_b
    eri_aa = ao2mo.general(mol, (mo_a, mo_a, mo_a, mo_a), compact=False).reshape([norb] * 4)
    eri_ab = ao2mo.general(mol, (mo_a, mo_a, mo_b, mo_b), compact=False).reshape([norb] * 4)
    eri_bb = ao2mo.general(mol, (mo_b, mo_b, mo_b, mo_b), compact=False).reshape([norb] * 4)
    ecore = mol.energy_nuc()
    nelec = mol.nelec  # (na, nb)

    cisolver = fci.direct_uhf.FCI()
    e_dav, ci = cisolver.kernel((h1a, h1b), (eri_aa, eri_ab, eri_bb), norb, nelec, ecore=ecore)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = cisolver.make_rdm12s(ci, norb, nelec)

    # Interleaved-spin-orbital determinant list + coefficients, in PySCF's (stra, strb) order.
    # Each CI coefficient is multiplied by the alpha/beta-ordering Fermi sign so the replica CI
    # matches PySCF's spin-orbital phase (see _reorder_sign).
    na, nb = nelec
    strsa = cistring.gen_strings4orblist(range(norb), na)
    strsb = cistring.gen_strings4orblist(range(norb), nb)
    ci = ci.reshape(len(strsa), len(strsb))
    dets, weights = [], []
    for ia, sa in enumerate(strsa):
        for ib, sb in enumerate(strsb):
            d = 0
            for p in range(norb):
                if (sa >> p) & 1:
                    d |= 1 << (2 * p)
                if (sb >> p) & 1:
                    d |= 1 << (2 * p + 1)
            dets.append(d)
            weights.append(ci[ia, ib] * _reorder_sign(d, norb))
    weights = np.asarray(weights, dtype=float)

    return dict(
        norb=norb, nelec=nelec, ecore=ecore, e_dav=e_dav,
        h1a=h1a, h1b=h1b, eri_aa=eri_aa, eri_ab=eri_ab, eri_bb=eri_bb,
        dm1a=dm1a, dm1b=dm1b, dm2aa=dm2aa, dm2ab=dm2ab, dm2bb=dm2bb,
        dets=dets, weights=weights,
    )


def _energy_from_rdms(ref, dm1a, dm1b, dm2aa, dm2ab, dm2bb) -> float:
    """Standard PySCF UHF energy contraction of spin-resolved RDMs with chemist ERIs."""
    e = ref["ecore"]
    e += np.einsum("pq,pq->", ref["h1a"], dm1a)
    e += np.einsum("pq,pq->", ref["h1b"], dm1b)
    e += 0.5 * np.einsum("pqrs,pqrs->", ref["eri_aa"], dm2aa)
    e += 0.5 * np.einsum("pqrs,pqrs->", ref["eri_bb"], dm2bb)
    e += np.einsum("pqrs,pqrs->", ref["eri_ab"], dm2ab)
    return e


# =============================================================================
# The tests (mirror pyscf/fci/test/test_uhf.py structure)
# =============================================================================
CASES = [
    # (tag, atom, basis, spin) -- small systems whose CI has opposite-spin double excitations with
    # non-negligible amplitude, so the buggy ab/ba term actually contributes.
    # (asymmetric bond lengths matter: a symmetric chain can make the sign error cancel in the
    #  scalar energy even though the RDM elements are wrong.)
    ("H3_asym_doublet", "H 0 0 0; H 0 0 0.8; H 0 0 2.2", "sto-3g", 1),
    ("OH_doublet", "O 0 0 0; H 0 0 0.97", "sto-3g", 1),
]

_TOL = 1e-9
_BUG = 1e-6  # minimum discrepancy we require the original code to exhibit


def _run_case(atom, basis, spin):
    """Build both kernels; shared by pytest and the __main__ runner."""
    ref = _build_reference(atom, basis, spin)
    patched = _RDMKernel(ref["norb"], patched=True)
    patched.build(ref["dets"], ref["weights"])
    original = _RDMKernel(ref["norb"], patched=False)
    original.build(ref["dets"], ref["weights"])
    return ref, patched.blocks(), original.blocks()


@pytest.mark.skipif(not _HAVE_PYSCF, reason="pyscf/numpy not importable in this environment")
@pytest.mark.parametrize("tag,atom,basis,spin", CASES, ids=[c[0] for c in CASES])
def test_rdm2_opposite_spin_sign_vs_pyscf(tag, atom, basis, spin):
    ref, pb, ob = _run_case(atom, basis, spin)
    na, nb = ref["nelec"]

    # --- 1-RDM: unchanged by the patch, matches PySCF, correct traces --------------------
    assert np.allclose(pb["1a"], ob["1a"], atol=_TOL)
    assert np.allclose(pb["1b"], ob["1b"], atol=_TOL)
    assert np.allclose(pb["1a"], ref["dm1a"], atol=_TOL)
    assert np.allclose(pb["1b"], ref["dm1b"], atol=_TOL)
    assert abs(np.trace(pb["1a"]) - na) < _TOL
    assert abs(np.trace(pb["1b"]) - nb) < _TOL

    # --- same-spin blocks: unchanged by the patch, match PySCF ---------------------------
    assert np.allclose(_same_to_pyscf(pb["aa"]), _same_to_pyscf(ob["aa"]), atol=_TOL)
    assert np.allclose(_same_to_pyscf(pb["bb"]), _same_to_pyscf(ob["bb"]), atol=_TOL)
    assert np.allclose(_same_to_pyscf(pb["aa"]), ref["dm2aa"], atol=_TOL)
    assert np.allclose(_same_to_pyscf(pb["bb"]), ref["dm2bb"], atol=_TOL)

    # --- opposite-spin block: PATCHED agrees with PySCF, ORIGINAL does not ---------------
    ab_patched = _opp_to_pyscf(pb["ab"])
    ab_original = _opp_to_pyscf(ob["ab"])
    assert np.allclose(ab_patched, ref["dm2ab"], atol=_TOL), (
        f"[{tag}] patched ab block disagrees with PySCF; "
        f"max|delta|={np.abs(ab_patched - ref['dm2ab']).max():.3e}"
    )
    max_bug = np.abs(ab_original - ref["dm2ab"]).max()
    assert max_bug > _BUG, (
        f"[{tag}] original ab block unexpectedly matched PySCF (max|delta|={max_bug:.3e}); "
        "the sign bug did not manifest for this case -- choose a larger system."
    )

    # The patch changes ONLY the opposite-spin block relative to the original ------------
    assert not np.allclose(ab_patched, ab_original, atol=_BUG), (
        f"[{tag}] patched and original ab blocks are identical -- the patch had no effect."
    )

    # ba block is the (2,3,0,1) transpose of ab; patched must satisfy that identity too.
    assert np.allclose(_opp_to_pyscf(pb["ba"]),
                       np.transpose(ref["dm2ab"], (2, 3, 0, 1)), atol=_TOL)

    # --- energy from the RDMs: patched == Davidson, original deviates --------------------
    e_patched = _energy_from_rdms(ref, pb["1a"], pb["1b"],
                                  _same_to_pyscf(pb["aa"]), ab_patched, _same_to_pyscf(pb["bb"]))
    e_original = _energy_from_rdms(ref, ob["1a"], ob["1b"],
                                   _same_to_pyscf(ob["aa"]), ab_original, _same_to_pyscf(ob["bb"]))
    assert abs(e_patched - ref["e_dav"]) < 1e-8, (
        f"[{tag}] E_recon(patched)={e_patched:.10f} != E_davidson={ref['e_dav']:.10f}"
    )
    assert abs(e_original - ref["e_dav"]) > _BUG, (
        f"[{tag}] E_recon(original)={e_original:.10f} unexpectedly equals "
        f"E_davidson={ref['e_dav']:.10f}; bug did not affect the energy for this case."
    )


if __name__ == "__main__":
    import sys

    if not _HAVE_PYSCF:
        print("pyscf/numpy not importable; run this where the project venv (pyscf) is available.")
        sys.exit(1)

    failures = 0
    for tag, atom, basis, spin in CASES:
        ref, pb, ob = _run_case(atom, basis, spin)
        ab_p = _opp_to_pyscf(pb["ab"]); ab_o = _opp_to_pyscf(ob["ab"])
        e_p = _energy_from_rdms(ref, pb["1a"], pb["1b"],
                                _same_to_pyscf(pb["aa"]), ab_p, _same_to_pyscf(pb["bb"]))
        e_o = _energy_from_rdms(ref, ob["1a"], ob["1b"],
                                _same_to_pyscf(ob["aa"]), ab_o, _same_to_pyscf(ob["bb"]))
        print(f"=== {tag}: norb={ref['norb']} nelec={ref['nelec']} ndet={len(ref['dets'])} ===")
        print(f"  E_davidson              = {ref['e_dav']:.12f}")
        print(f"  E_recon (patched)       = {e_p:.12f}   dE={e_p-ref['e_dav']:+.3e}")
        print(f"  E_recon (original)      = {e_o:.12f}   dE={e_o-ref['e_dav']:+.3e}   <- bug")
        print(f"  max|ab_patched  - pyscf|= {np.abs(ab_p-ref['dm2ab']).max():.3e}")
        print(f"  max|ab_original - pyscf|= {np.abs(ab_o-ref['dm2ab']).max():.3e}   <- bug")
        print(f"  max|dm1a - pyscf|       = {np.abs(pb['1a']-ref['dm1a']).max():.3e}")
        print(f"  max|aa   - pyscf|       = {np.abs(_same_to_pyscf(pb['aa'])-ref['dm2aa']).max():.3e}")
        try:
            test_rdm2_opposite_spin_sign_vs_pyscf(tag, atom, basis, spin)
            print(f"  [PASS] {tag}")
        except AssertionError as exc:
            failures += 1
            print(f"  [FAIL] {tag}: {exc}")

    print(f"\n{len(CASES)-failures}/{len(CASES)} cases passed.")
    sys.exit(1 if failures else 0)
