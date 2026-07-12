"""Tests for broken-symmetry UHF convergence (_converge_broken_symmetry_uhf).

The Fe-S audit found that default UHF + internal-stability-following STAYS spin-pure on systems whose
singlet ground state is actually antiferromagnetic (the RHF determinant is a UHF stationary point;
breaking the spatial symmetry to localize the spins needs a spin-IMBALANCED initial guess). The fix
adds an antiferromagnetic guess and keeps the lower solution. These tests pin that behavior on small
molecules with KNOWN magnetic character:

  * Stretched H2 (0.74 A -> 3.0 A): the textbook broken-symmetry singlet. Default UHF stays at RHF
    (<S^2>=0); the AF guess must find the spin-broken solution (<S^2>~1, ~0.28 Ha LOWER, approaching
    the 2x isolated-H-atom limit -1.0 Ha). THIS is the exact scenario the fix targets.
  * Equilibrium H2 (0.74 A): a genuine closed shell -> UHF must STAY spin-pure (<S^2>~0, == RHF).
    Guards against the fix FALSELY breaking symmetry on well-behaved singlets.
  * O2 (ground triplet): a high-spin control -> <S^2>~2 (S=1), energy below RHF, correct multiplicity.
  * Stretched H4 (AF): default stability already finds the broken solution here; confirms the fix
    does not regress the case that already worked.

Known references: stretched H2 dissociates to two neutral H (each S=1/2), so the BS singlet has
<S^2> -> 1 and E -> -1.0 Ha (2 x -0.5). RHF cannot describe this (stays closed-shell, ~-0.66 Ha).
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto, scf

from qcsc_workflow_utility.chem import _converge_broken_symmetry_uhf


def _rhf_energy(atom, basis="sto-3g", spin=0, charge=0):
    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    return scf.RHF(mol).run().e_tot


def _bs_uhf(atom, basis="sto-3g", spin=0, charge=0):
    """Run the production broken-symmetry UHF driver on a fresh geometry-based UHF."""
    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.conv_tol = 1e-10
    mf = _converge_broken_symmetry_uhf(mf)
    ss, _ = mf.spin_square()
    return mf.e_tot, ss


def test_stretched_h2_finds_broken_symmetry():
    """Stretched H2: the AF guess must find the spin-broken solution the default path misses.

    This is the core scenario of the fix. Default UHF stays at RHF (-0.656, <S^2>=0); the BS solution
    is ~-0.93 (<S^2>~1). If the fix works, _converge_broken_symmetry_uhf returns the BS solution.
    """
    e_rhf = _rhf_energy("H 0 0 0; H 0 0 3.0")
    e_bs, ss = _bs_uhf("H 0 0 0; H 0 0 3.0")
    # BS must be substantially BELOW RHF (the whole point: RHF can't dissociate H2)
    assert e_bs < e_rhf - 0.1, f"BS-UHF {e_bs:.4f} did not drop below RHF {e_rhf:.4f}"
    # and must be genuinely spin-broken (a stretched-H2 singlet -> two localized radicals, <S^2>~1)
    assert ss > 0.5, f"stretched-H2 UHF should break symmetry (<S^2>>0.5), got {ss:.3f}"
    # energy should approach the isolated-atom limit (-1.0 Ha), well below -0.85
    assert e_bs < -0.85, f"BS-UHF {e_bs:.4f} not near the 2x H-atom dissociation limit"


def test_equilibrium_h2_stays_spin_pure():
    """Equilibrium H2 is a genuine closed shell -> the fix must NOT falsely break symmetry."""
    e_rhf = _rhf_energy("H 0 0 0; H 0 0 0.74")
    e_uhf, ss = _bs_uhf("H 0 0 0; H 0 0 0.74")
    assert abs(ss) < 1e-3, f"equilibrium H2 should stay spin-pure, got <S^2>={ss:.4f}"
    assert abs(e_uhf - e_rhf) < 1e-5, f"UHF {e_uhf:.6f} != RHF {e_rhf:.6f} at equilibrium"


def test_o2_triplet_ground_state():
    """O2 ground state is a triplet (S=1): correct <S^2>~2, energy at or below RHF."""
    e_rhf = _rhf_energy("O 0 0 0; O 0 0 1.208", spin=2)
    e_uhf, ss = _bs_uhf("O 0 0 0; O 0 0 1.208", spin=2)
    assert abs(ss - 2.0) < 0.3, f"O2 triplet should have <S^2>~2, got {ss:.3f}"
    assert e_uhf <= e_rhf + 1e-6, f"UHF {e_uhf:.6f} above RHF {e_rhf:.6f}"


def test_stretched_h4_no_regression():
    """Stretched H4: default stability ALREADY finds the broken solution; the fix must not regress it
    (must still return a spin-broken solution well below RHF)."""
    e_rhf = _rhf_energy("H 0 0 0; H 0 0 2.5; H 0 0 5.0; H 0 0 7.5")
    e_bs, ss = _bs_uhf("H 0 0 0; H 0 0 2.5; H 0 0 5.0; H 0 0 7.5")
    assert e_bs < e_rhf - 0.1, f"H4 BS-UHF {e_bs:.4f} not below RHF {e_rhf:.4f}"
    assert ss > 1.0, f"stretched H4 should be strongly spin-broken (<S^2>>1), got {ss:.3f}"


@pytest.mark.parametrize(
    "atom,spin,expect_broken",
    [
        ("H 0 0 0; H 0 0 3.0", 0, True),    # stretched singlet -> broken
        ("H 0 0 0; H 0 0 0.74", 0, False),  # equilibrium singlet -> not broken
    ],
)
def test_broken_symmetry_only_when_physical(atom, spin, expect_broken):
    """The fix breaks symmetry ONLY when it lowers the energy (stretched), never spuriously
    (equilibrium). This is the accept/reject guard: keep the AF solution only if it is lower."""
    e_bs, ss = _bs_uhf(atom, spin=spin)
    if expect_broken:
        assert ss > 0.5, f"expected broken symmetry, got <S^2>={ss:.3f}"
    else:
        assert abs(ss) < 1e-2, f"expected spin-pure, got <S^2>={ss:.3f}"
