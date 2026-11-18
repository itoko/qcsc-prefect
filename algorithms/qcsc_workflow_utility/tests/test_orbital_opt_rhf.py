"""RHF orbital optimization (one rotation U for both spins, spin-summed RDMs).

The RDMs come from a TRUNCATED selected-CI subspace (HF + alpha/beta singles): a full-FCI RDM is
stationary under orbital rotations (zero gradient), so it could not show the optimizer moving.
Stretched H2O/sto-3g (7 orbitals, O-H ~1.9 A, where the singles matter) keeps every test to a
fraction of a second.

  * spin_summed_rdms reproduces PySCF's spin-summed make_rdm12 in both storages (open-shell state,
    so a wrong ab/ba pairing would show).
  * optimize_orbitals_rhf: E(U=I) equals the CI energy, the objective decreases, U is orthogonal,
    the JAX gradient matches finite differences, and re-diagonalizing the SAME subspace in the
    rotated basis does not raise the energy.
  * RHF == UHF with Ua = Ub (restricted=True) on an unrestricted copy of the same integrals.
  * optimize_orbitals on RHF integrals returns one rotation (Ua == Ub) even for spin-asymmetric
    RDMs; it used to optimize Ua and Ub separately and the rotation step then dropped Ub.
  * OO_INIT=1 on the RHF path keeps the CCSD-RDM rotation only when the refreshed CCSD energy
    does not rise, and skips open-shell references.
"""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("qiskit_addon_sqd")

from pyscf import ao2mo, fci, gto, scf  # noqa: E402
from qiskit_addon_sqd.fermion import solve_fermion  # noqa: E402

from qcsc_workflow_utility.chem import ElectronicProperties  # noqa: E402
from qcsc_workflow_utility.orbital_opt import (  # noqa: E402
    _make_jax_rhf_obj_and_grad,
    _rhf_energy,
    optimize_orbitals,
    optimize_orbitals_rhf,
    rotate_electronic_properties,
    spin_summed_rdms,
)

# At equilibrium HF+singles barely moves (Brillouin); stretched, the OO gain is ~0.1 mHa.
H2O = "O 0 0 0; H 0 1.5 1.16; H 0 -1.5 1.16"


def _pqrs_to_prqs(a):
    return np.asarray(a).transpose(0, 2, 1, 3)


@pytest.fixture(scope="module")
def h2o():
    """Stretched H2O/sto-3g RHF integrals in the MO basis: (h1, h2, nuc, norb)."""
    mol = gto.M(atom=H2O, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    norb = mo.shape[1]
    h1 = mo.T @ mf.get_hcore() @ mo
    h2 = ao2mo.full(mol, mo, compact=False).reshape((norb,) * 4)
    return h1, h2, float(mol.energy_nuc()), norb


def _props(h1, h2, nuc, norb, nelec, *, unrestricted=False):
    extra = {}
    if unrestricted:
        extra = dict(
            unrestricted=True,
            one_body_tensor_b=h1.copy(),
            two_body_tensor_ab=h2.copy(),
            two_body_tensor_bb=h2.copy(),
        )
    return ElectronicProperties(
        one_body_tensor=h1,
        two_body_tensor=h2,
        t2=np.zeros((0, 0, 0, 0)),
        initial_occupancy=(np.zeros(norb), np.zeros(norb)),
        nuclear_repulsion_energy=nuc,
        num_orbitals=norb,
        num_electrons=nelec,
        open_shell=nelec[0] != nelec[1],
        spin_sq=0.0,
        **extra,
    )


def _hf_and_singles(norb, n):
    """Integer CI strings of the index-aufbau determinant and all its single excitations."""
    hf = (1 << n) - 1
    dets = {hf}
    for i, a in itertools.product(range(n), range(n, norb)):
        dets.add(hf ^ (1 << i) ^ (1 << a))
    return np.array(sorted(dets), dtype=np.int64)


def _truncated_ci(h1, h2, norb, nelec):
    """Selected CI on HF + singles (per spin): (E_electronic, SCIState, ci_strings)."""
    ci = (_hf_and_singles(norb, nelec[0]), _hf_and_singles(norb, nelec[1]))
    e, sci, _, _ = solve_fermion(ci, h1, h2, open_shell=nelec[0] != nelec[1])
    return float(e), sci, ci


@pytest.fixture(scope="module")
def closed_shell(h2o):
    """RHF props + truncated-CI spin-summed RDMs (pqrs storage) of neutral H2O (5, 5)."""
    h1, h2, nuc, norb = h2o
    nelec = (5, 5)
    e, sci, ci = _truncated_ci(h1, h2, norb, nelec)
    ep = _props(h1, h2, nuc, norb, nelec)
    return ep, e, ci, sci.rdm(rank=1, spin_summed=True), sci.rdm(rank=2, spin_summed=True), sci


@pytest.fixture(scope="module")
def open_shell(h2o):
    """H2O+ (5, 4): per-spin truncated-CI RDMs (pqrs) with genuinely different alpha / beta."""
    h1, h2, nuc, norb = h2o
    nelec = (5, 4)
    _, sci, _ = _truncated_ci(h1, h2, norb, nelec)
    r1a, r1b = sci.rdm(rank=1, spin_summed=False)
    r2aa, r2ab, r2bb = sci.rdm(rank=2, spin_summed=False)
    return (h1, h2, nuc, norb, nelec), (r1a, r1b, r2aa, r2ab, r2bb)


# ─── spin_summed_rdms ────────────────────────────────────────────────────────────────────────

class TestSpinSummedRdms:
    @pytest.fixture
    def fci_rdms(self, h2o):
        h1, h2, _, norb = h2o
        nelec = (5, 4)
        solver = fci.direct_spin1.FCI()
        _, c = solver.kernel(h1, h2, norb, nelec)
        (d1a, d1b), (d2aa, d2ab, d2bb) = solver.make_rdm12s(c, norb, nelec)
        d1, d2 = solver.make_rdm12(c, norb, nelec)
        return (d1a, d1b, d2aa, d2ab, d2bb), (d1, d2)

    def test_pqrs_matches_pyscf(self, fci_rdms):
        blocks, (d1, d2) = fci_rdms
        g1, g2 = spin_summed_rdms(*blocks, rdm2_notation="pqrs")
        np.testing.assert_allclose(g1, d1, atol=1e-12)
        np.testing.assert_allclose(g2, d2, atol=1e-12)

    def test_prqs_matches_pyscf(self, fci_rdms):
        (d1a, d1b, d2aa, d2ab, d2bb), (d1, d2) = fci_rdms
        g1, g2 = spin_summed_rdms(
            d1a, d1b, _pqrs_to_prqs(d2aa), _pqrs_to_prqs(d2ab), _pqrs_to_prqs(d2bb),
            rdm2_notation="prqs",
        )
        np.testing.assert_allclose(g1, d1, atol=1e-12)
        np.testing.assert_allclose(g2, _pqrs_to_prqs(d2), atol=1e-12)


# ─── optimize_orbitals_rhf ───────────────────────────────────────────────────────────────────

class TestOptimizeOrbitalsRhf:
    def test_energy_at_identity_is_ci_energy(self, closed_shell):
        ep, e_ci, _, g1, g2, _ = closed_shell
        n_p = ep.num_orbitals * (ep.num_orbitals - 1) // 2
        e0 = _rhf_energy(
            np.zeros(n_p), ep.num_orbitals, g1, _pqrs_to_prqs(g2),
            ep.one_body_tensor, ep.two_body_tensor, ep.nuclear_repulsion_energy,
        )
        assert e0 == pytest.approx(e_ci + ep.nuclear_repulsion_energy, abs=1e-9)

    def test_lowers_objective_with_orthogonal_rotation(self, closed_shell):
        ep, e_ci, _, g1, g2, _ = closed_shell
        U, e_opt, grad = optimize_orbitals_rhf(ep, g1, g2, rdm2_notation="pqrs", trust_radius=0.1)
        e_before = e_ci + ep.nuclear_repulsion_energy
        assert U.shape == (ep.num_orbitals,) * 2
        np.testing.assert_allclose(U.T @ U, np.eye(ep.num_orbitals), atol=1e-10)
        assert e_opt < e_before - 1e-5, (e_opt, e_before)
        assert np.isfinite(grad)

    def test_jax_gradient_matches_finite_difference(self, closed_shell):
        pytest.importorskip("jax")
        ep, _, _, g1, g2, _ = closed_shell
        norb = ep.num_orbitals
        args = (norb, g1, _pqrs_to_prqs(g2), ep.one_body_tensor, ep.two_body_tensor,
                ep.nuclear_repulsion_energy)
        obj, grad = _make_jax_rhf_obj_and_grad(*args)
        x = np.random.default_rng(0).normal(scale=0.05, size=norb * (norb - 1) // 2)
        assert obj(x) == pytest.approx(_rhf_energy(x, *args), abs=1e-10)
        h = 1e-5
        fd = np.array([
            (_rhf_energy(x + h * e, *args) - _rhf_energy(x - h * e, *args)) / (2 * h)
            for e in np.eye(x.size)
        ])
        np.testing.assert_allclose(grad(x), fd, atol=1e-7)

    def test_rotated_basis_does_not_raise_subspace_energy(self, closed_shell):
        """Variational check: the SAME determinant subspace re-diagonalized after the rotation."""
        ep, e_ci, ci, g1, g2, _ = closed_shell
        U, _, _ = optimize_orbitals_rhf(ep, g1, g2, rdm2_notation="pqrs", trust_radius=0.02)
        ep_rot = rotate_electronic_properties(ep, U)
        e_rot, _, _, _ = solve_fermion(ci, ep_rot.one_body_tensor, ep_rot.two_body_tensor)
        assert e_rot < e_ci - 1e-5, (e_rot, e_ci)

    def test_rejects_unrestricted_props(self, h2o, closed_shell):
        h1, h2, nuc, norb = h2o
        _, _, _, g1, g2, _ = closed_shell
        ep_u = _props(h1, h2, nuc, norb, (5, 5), unrestricted=True)
        with pytest.raises(ValueError, match="restricted"):
            optimize_orbitals_rhf(ep_u, g1, g2)


# ─── one rotation through optimize_orbitals ──────────────────────────────────────────────────

class TestSingleRotation:
    def test_rhf_equals_restricted_uhf(self, open_shell):
        """Same integrals as RHF and as UHF(h_b = h_a): RHF path == UHF path with Ua = Ub."""
        (h1, h2, nuc, norb, nelec), blocks = open_shell
        ep_r = _props(h1, h2, nuc, norb, nelec)
        ep_u = _props(h1, h2, nuc, norb, nelec, unrestricted=True)
        g1, g2 = spin_summed_rdms(*blocks, rdm2_notation="pqrs")

        U, e_r, _ = optimize_orbitals_rhf(ep_r, g1, g2, rdm2_notation="pqrs", trust_radius=0.1)
        Ua, Ub, e_u, _ = optimize_orbitals(
            ep_u, *blocks, rdm2_notation="pqrs", trust_radius=0.1, restricted=True,
        )
        assert Ua is Ub
        assert e_u == pytest.approx(e_r, abs=1e-8)
        np.testing.assert_allclose(Ua, U, atol=1e-5)

    def test_rhf_props_give_one_rotation_for_asymmetric_rdms(self, open_shell):
        (h1, h2, nuc, norb, nelec), blocks = open_shell
        ep_r = _props(h1, h2, nuc, norb, nelec)
        Ua, Ub, e, _ = optimize_orbitals(ep_r, *blocks, rdm2_notation="pqrs", trust_radius=0.1)
        assert Ua is Ub
        U, e_rhf, _ = optimize_orbitals_rhf(
            ep_r, *spin_summed_rdms(*blocks, rdm2_notation="pqrs"),
            rdm2_notation="pqrs", trust_radius=0.1,
        )
        assert e == pytest.approx(e_rhf, abs=1e-10)
        np.testing.assert_allclose(Ua, U, atol=1e-10)

    def test_unrestricted_default_keeps_independent_rotations(self, open_shell):
        (h1, h2, nuc, norb, nelec), blocks = open_shell
        ep_u = _props(h1, h2, nuc, norb, nelec, unrestricted=True)
        Ua, Ub, e_free, _ = optimize_orbitals(ep_u, *blocks, rdm2_notation="pqrs", trust_radius=0.1)
        assert not np.allclose(Ua, Ub, atol=1e-6)
        _, _, e_restr, _ = optimize_orbitals(
            ep_u, *blocks, rdm2_notation="pqrs", trust_radius=0.1, restricted=True,
        )
        # The restricted rotation is a special case of the independent pair.
        assert e_free <= e_restr + 1e-9


# ─── OO_INIT on the RHF path (chem._apply_initial_oo_rhf) ─────────────────────────────────────

class TestInitialOoRhf:
    @pytest.fixture
    def chem_log(self, monkeypatch):
        import qcsc_workflow_utility.chem as chem

        log = mock.MagicMock()
        monkeypatch.setattr(chem, "get_run_logger", lambda: log)
        return chem, log

    @staticmethod
    def _messages(log):
        calls = log.info.call_args_list + log.warning.call_args_list
        return [c.args[0] for c in calls if c.args and isinstance(c.args[0], str)]

    def test_keeps_rotation_only_if_ccsd_energy_does_not_rise(self, chem_log, monkeypatch):
        chem, log = chem_log
        monkeypatch.delenv("OO_INIT", raising=False)
        ep0 = chem.compute_molecular_integrals_from_geometry.fn(atom=H2O, basis="sto-3g")
        _, e0 = chem.refresh_cc_seed(ep0, return_energy=True)

        monkeypatch.setenv("OO_INIT", "1")
        ep1 = chem.compute_molecular_integrals_from_geometry.fn(atom=H2O, basis="sto-3g")
        msgs = self._messages(log)
        accepted = any(m.startswith("OO_INIT accepted") for m in msgs)
        rejected = any(m.startswith("OO_INIT rejected") for m in msgs)
        assert accepted != rejected, msgs

        _, e1 = chem.refresh_cc_seed(ep1, return_energy=True)
        assert e1 <= e0 + 1e-6, (e1, e0)
        if accepted:
            assert not np.allclose(ep1.one_body_tensor, ep0.one_body_tensor)
        else:
            np.testing.assert_allclose(ep1.one_body_tensor, ep0.one_body_tensor)

    def test_skips_open_shell(self, chem_log, h2o):
        """Defensive guard: the RHF CC refresh builds a doubly occupied reference."""
        chem, log = chem_log
        h1, h2, nuc, norb = h2o
        props = _props(h1, h2, nuc, norb, (5, 4))
        mycc = mock.MagicMock()
        assert chem._apply_initial_oo_rhf(props, mycc) is props
        mycc.make_rdm1.assert_not_called()
        assert any(m.startswith("OO_INIT skipped") for m in self._messages(log))
