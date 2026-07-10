"""Correctness tests for the RHF/UHF reference-building pipeline (chem.py).

Before layering anything on top (orbital optimization, CISD injection, ...), these tests lock down
that the classical reference the whole SQD pipeline is seeded from is physically correct:

  * RHF vs UHF are NOT mixed (restricted stays restricted; unrestricted is genuinely unrestricted).
  * UHF reaches the true (broken-symmetry when it exists) minimum for correlated singlets, instead
    of collapsing to the RHF stationary point -- the bug that made "UHF" == "RHF" for the Fe-S
    singlets and made the UCCSD seed/amplitudes restricted in disguise.
  * The variational ordering RHF >= UHF >= FCI holds.
  * Spin: <S^2> is ~0 for a genuine closed-shell singlet, ~2 for a triplet, ~0.75 (+contamination)
    for a doublet, and the (na, nb) sector matches the requested 2Sz.
  * ElectronicProperties shapes/blocks are consistent (UHF carries the beta / ab / bb blocks and
    the UCCSD t2 tuple; RHF carries only the restricted tensors).

Ground-truth energies/spins were established with independent PySCF RHF/UHF/FCI calculations
(sto-3g), so the assertions are anchored to physics, not to the code under test.

All systems are tiny (H4/H5, sto-3g) so the suite runs in seconds with no HPC/quantum hardware.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

try:
    from pyscf import gto, scf, cc, fci  # noqa: F401
    import qcsc_workflow_utility.chem as chem
    _HAVE = True
except Exception as _e:  # pragma: no cover
    _HAVE = False
    _WHY = str(_e)

pytestmark = pytest.mark.skipif(not _HAVE, reason="pyscf / chem not importable")


# H-chain geometries (sto-3g). Stretched chains are strongly correlated (UHF breaks symmetry);
# equilibrium chains are weakly correlated (UHF stays restricted for a singlet).
H4_STRETCHED = "H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0"
H4_EQUILIBRIUM = "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
H4_TRIPLET = "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0"
H5_DOUBLET = "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0; H 0 0 4.0"


@pytest.fixture(autouse=True)
def _silence_prefect_logger(monkeypatch):
    monkeypatch.setattr(chem, "get_run_logger", lambda: mock.MagicMock())


# ── reference energies (independent PySCF) ──────────────────────────────────────────────────
def _rhf(atom, spin=0):
    return scf.RHF(gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)).run(verbose=0)


def _fci_energy(atom, spin):
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    return float(fci.FCI(mf).kernel()[0])


# ── mean-field level: _converge_broken_symmetry_uhf ─────────────────────────────────────────

class TestUHFMeanField:
    def test_stretched_singlet_breaks_symmetry(self):
        """Strongly-correlated H4 singlet: UHF must break spin symmetry and drop well below RHF."""
        mol = gto.M(atom=H4_STRETCHED, basis="sto-3g", spin=0, verbose=0)
        e_rhf = _rhf(H4_STRETCHED).e_tot
        mf = chem._converge_broken_symmetry_uhf(scf.UHF(mol))
        ss, _ = mf.spin_square()
        assert ss > 0.5, f"UHF stayed restricted (<S^2>={ss:.3f}) on a correlated singlet"
        assert mf.e_tot < e_rhf - 1e-2, f"UHF {mf.e_tot:.6f} not below RHF {e_rhf:.6f}"
        assert mf.e_tot >= _fci_energy(H4_STRETCHED, 0) - 1e-8, "UHF below FCI (non-variational!)"

    def test_equilibrium_singlet_stays_restricted(self):
        """Weakly-correlated H4 singlet: NO spin instability exists, so UHF must equal RHF
        (<S^2>=0). The fix must not spuriously break symmetry where it shouldn't."""
        mol = gto.M(atom=H4_EQUILIBRIUM, basis="sto-3g", spin=0, verbose=0)
        e_rhf = _rhf(H4_EQUILIBRIUM).e_tot
        mf = chem._converge_broken_symmetry_uhf(scf.UHF(mol))
        ss, _ = mf.spin_square()
        assert ss < 1e-3, f"UHF spuriously broke symmetry (<S^2>={ss:.3e}) at equilibrium"
        assert abs(mf.e_tot - e_rhf) < 1e-6, f"UHF {mf.e_tot:.8f} != RHF {e_rhf:.8f} at equilibrium"

    def test_triplet_spin(self):
        """H4 triplet (2Sz=2): <S^2> should be ~2 (S=1 -> S(S+1)=2) and energy >= FCI."""
        mol = gto.M(atom=H4_TRIPLET, basis="sto-3g", spin=2, verbose=0)
        mf = chem._converge_broken_symmetry_uhf(scf.UHF(mol))
        ss, _ = mf.spin_square()
        assert abs(ss - 2.0) < 0.15, f"triplet <S^2>={ss:.3f} not ~2.0"
        assert mf.e_tot >= _fci_energy(H4_TRIPLET, 2) - 1e-8, "triplet UHF below FCI"

    def test_doublet_spin(self):
        """H5 doublet (2Sz=1): <S^2> near 0.75 (+ some contamination), energy >= FCI."""
        mol = gto.M(atom=H5_DOUBLET, basis="sto-3g", spin=1, verbose=0)
        mf = chem._converge_broken_symmetry_uhf(scf.UHF(mol))
        ss, _ = mf.spin_square()
        assert 0.7 <= ss < 1.5, f"doublet <S^2>={ss:.3f} out of expected range"
        assert mf.e_tot >= _fci_energy(H5_DOUBLET, 1) - 1e-8, "doublet UHF below FCI"


# ── ElectronicProperties level: compute_molecular_integrals_from_geometry ───────────────────

class TestElectronicPropertiesConsistency:
    def test_rhf_is_restricted_only(self):
        """RHF path: unrestricted=False, no beta/ab/bb blocks, no t2_ab/t2_bb (not mixed)."""
        ep = chem.compute_molecular_integrals_from_geometry.fn(
            atom=H4_EQUILIBRIUM, basis="sto-3g", unrestricted=False
        )
        assert ep.unrestricted is False
        assert ep.one_body_tensor_b is None
        assert ep.two_body_tensor_ab is None and ep.two_body_tensor_bb is None
        assert ep.t2_ab is None and ep.t2_bb is None

    def test_uhf_carries_all_blocks(self):
        """UHF path: unrestricted=True with beta/ab/bb integral blocks and the UCCSD t2 tuple."""
        ep = chem.compute_molecular_integrals_from_geometry.fn(
            atom=H4_STRETCHED, basis="sto-3g", unrestricted=True, spin=0
        )
        norb = ep.num_orbitals
        assert ep.unrestricted is True
        for blk in (ep.one_body_tensor, ep.one_body_tensor_b):
            assert blk.shape == (norb, norb)
        for blk in (ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb):
            assert blk.shape == (norb, norb, norb, norb)
        assert ep.t2.ndim == 4 and ep.t2_ab.ndim == 4 and ep.t2_bb.ndim == 4
        assert len(ep.initial_occupancy[0]) == norb
        assert len(ep.initial_occupancy[1]) == norb

    def test_occupancies_are_fractional_for_correlated_uhf(self):
        """The UHF occupancy seed (from the correlated UCCSD 1-RDM diagonal) must be FRACTIONAL for
        a correlated system -- integer 0/1 occupancies give configuration recovery no signal and
        collapse the SQD subspace to bare HF (a past failure mode)."""
        ep = chem.compute_molecular_integrals_from_geometry.fn(
            atom=H4_STRETCHED, basis="sto-3g", unrestricted=True, spin=0
        )
        occ_a, occ_b = np.asarray(ep.initial_occupancy[0]), np.asarray(ep.initial_occupancy[1])
        frac = np.sum((occ_a > 0.01) & (occ_a < 0.99)) + np.sum((occ_b > 0.01) & (occ_b < 0.99))
        assert frac > 0, "no fractional occupancies -> recovery has no correlation signal"

    def test_electron_count_matches_2sz(self):
        """(na, nb) must match the requested 2Sz for open-shell references."""
        ep_t = chem.compute_molecular_integrals_from_geometry.fn(
            atom=H4_TRIPLET, basis="sto-3g", unrestricted=True, spin=2
        )
        na, nb = ep_t.num_electrons
        assert na - nb == 2, f"triplet (na,nb)=({na},{nb}) does not give 2Sz=2"
        ep_d = chem.compute_molecular_integrals_from_geometry.fn(
            atom=H5_DOUBLET, basis="sto-3g", unrestricted=True, spin=1
        )
        na, nb = ep_d.num_electrons
        assert na - nb == 1, f"doublet (na,nb)=({na},{nb}) does not give 2Sz=1"


# ── end-to-end invariant: RHF >= UHF (>= FCI) via the pipeline, not mixed ───────────────────

class TestVariationalOrdering:
    @pytest.mark.parametrize("atom,spin", [(H4_STRETCHED, 0), (H4_EQUILIBRIUM, 0)])
    def test_uhf_le_rhf_le_none_via_pipeline(self, atom, spin):
        """The pipeline's UHF mean-field energy must be <= its RHF mean-field energy and >= FCI.
        (Uses the mean-field energies the pipeline actually converges, ensuring the reference the
        SQD seed is built from respects the variational ordering and is not restricted-in-disguise
        for the strongly-correlated case.)"""
        e_rhf = _rhf(atom, spin).e_tot
        mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
        mf_u = chem._converge_broken_symmetry_uhf(scf.UHF(mol))
        e_fci = _fci_energy(atom, spin)
        assert mf_u.e_tot <= e_rhf + 1e-9, f"UHF {mf_u.e_tot:.6f} > RHF {e_rhf:.6f}"
        assert mf_u.e_tot >= e_fci - 1e-8, f"UHF {mf_u.e_tot:.6f} < FCI {e_fci:.6f}"
