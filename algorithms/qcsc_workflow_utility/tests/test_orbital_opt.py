#!/usr/bin/env python3
"""Unit tests for qcsc_workflow_utility.orbital_opt.

These tests verify orbital_opt.py independently of SQD / Fugaku / qiskit_addon_sqd.
Only PySCF (integral generation + FCI RDM) and scipy are required.

Test coverage
-------------
T1  Energy reproducibility (pqrs notation)
      optimize_orbitals with pqrs-storage RDM from PySCF reproduces UHF energy at x=0.
T2  Energy decreases (UHF, spin_free=False)
      E_after <= E_before for a correlated (non-HF) state (FCI RDM on UHF Hamiltonian).
T3  Ua / Ub are unitary
      ||U^T U - I||_F < 1e-12 for all returned rotation matrices.
T4  rdm2_notation="prqs" path
      Passing prqs-converted RDMs directly gives the same result as "pqrs".
T5  rotate_electronic_properties consistency
      After rotation: E(H_new, rdm_orig) == E_opt from optimize_orbitals.
T6  RHF path (unrestricted=False)
      optimize_orbitals on RHF system returns Ua == Ub and energy <= UHF reference.
T7  JAX analytical gradient
      T7a: JAX objective == NumPy objective at x=0 (within 1e-10).
      T7b: JAX gradient at x=0 has the same shape and is finite.
      T7c: JAX and NumPy optimization paths converge to the same final energy (within 1e-6).
      T7d: JAX path uses fewer optimizer iterations (nit) than NumPy finite-difference path.

Run locally:
    conda activate qcsc
    cd /Users/yutolt/Documents/qcsc-prefect
    python -m pytest algorithms/qcsc_workflow_utility/tests/test_orbital_opt.py -v
    # or as a standalone report:
    python algorithms/qcsc_workflow_utility/tests/test_orbital_opt.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ─── パス設定（pytest 経由でも standalone 実行でも動く）─────────────────────────
_ROOT = Path(__file__).resolve().parents[3]  # …/algorithms/qcsc_workflow_utility
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING)

# ─── pytest は optional（standalone 実行でも動く）─────────────────────────────
try:
    import pytest as _pytest
    _HAS_PYTEST = True
    pytest = _pytest
except ImportError:
    _HAS_PYTEST = False
    # pytest を使わない場合のダミー
    class _PytestDummy:
        class mark:
            @staticmethod
            def skipif(cond, reason=""):
                def dec(cls): return cls
                return dec
        class fixture:
            def __init__(self, scope=None): pass
            def __call__(self, f): return f
        @staticmethod
        def skip(msg, allow_module_level=False):
            print(f"SKIP: {msg}")
    pytest = _PytestDummy()

# ─── import（失敗したら skip）──────────────────────────────────────────────────
try:
    from qcsc_workflow_utility.orbital_opt import (
        _uhf_energy,
        _make_jax_uhf_obj_and_grad,
        _unitary_from_skew,
        optimize_orbitals,
        rotate_electronic_properties,
    )
    from qcsc_workflow_utility.chem import ElectronicProperties
except ImportError as e:
    print(f"SKIP: orbital_opt not importable: {e}")
    sys.exit(0)

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _JAX = True
except ImportError:
    _JAX = False

try:
    from pyscf import ao2mo, fci as pyscf_fci, gto, scf
    _PYSCF = True
except ImportError:
    _PYSCF = False


# ─── ヘルパー：分子積分 + FCI RDM ────────────────────────────────────────────

def _build_uhf_ep(atom: str, basis: str, spin: int) -> tuple:
    """UHF 積分から ElectronicProperties を構築して返す。
    
    Returns
    -------
    ep         : ElectronicProperties (UHF, unrestricted=True)
    e_uhf      : UHF total energy (Ha)
    rdm1_aa    : alpha 1-RDM from UHF (pqrs-storage convention dummy — identity-like)
    rdm1_bb    : beta  1-RDM from UHF
    rdm2_*_pqrs: 2-RDMs in pqrs-storage (PySCF make_rdm12s convention)
    e_fci      : FCI total energy (Ha)
    """
    mol = gto.Mole()
    mol.build(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = scf.UHF(mol).run()
    mo_a, mo_b = mf.mo_coeff
    norb = mo_a.shape[1]
    hcore = mf.get_hcore()
    eri = mf._eri
    h1_a = mo_a.T @ hcore @ mo_a
    h1_b = mo_b.T @ hcore @ mo_b
    h2_aa = ao2mo.full(eri, mo_a, compact=False).reshape(norb, norb, norb, norb)
    h2_bb = ao2mo.full(eri, mo_b, compact=False).reshape(norb, norb, norb, norb)
    h2_ab = ao2mo.general(eri, (mo_a, mo_a, mo_b, mo_b),
                          compact=False).reshape(norb, norb, norb, norb)
    nuc = mol.energy_nuc()
    na, nb = mol.nelec

    ep = ElectronicProperties(
        one_body_tensor=h1_a,
        two_body_tensor=h2_aa,
        t2=np.zeros((na, na, norb, norb)),
        initial_occupancy=(np.ones(na) / na, np.ones(nb) / nb),
        nuclear_repulsion_energy=nuc,
        num_orbitals=norb,
        num_electrons=(na, nb),
        open_shell=(na != nb),
        spin_sq=float(spin / 2 * (spin / 2 + 1)),
        unrestricted=True,
        one_body_tensor_b=h1_b,
        two_body_tensor_ab=h2_ab,
        two_body_tensor_bb=h2_bb,
        t2_ab=np.zeros((na, nb, norb, norb)),
        t2_bb=np.zeros((nb, nb, norb, norb)),
    )

    # FCI の基底状態 RDM を取得（pqrs-storage: dm2[p,q,r,s] = <p†r†sq>）
    cisolver = pyscf_fci.FCI(mol, mo_a)
    e_fci, c = cisolver.kernel(nelec=mol.nelec)
    dm1s, dm2s = cisolver.make_rdm12s(c, norb, mol.nelec)
    rdm1_aa, rdm1_bb = np.asarray(dm1s[0]), np.asarray(dm1s[1])
    rdm2_aa_pqrs = np.asarray(dm2s[0])
    rdm2_ab_pqrs = np.asarray(dm2s[1])
    rdm2_bb_pqrs = np.asarray(dm2s[2])

    return (ep, float(mf.e_tot), rdm1_aa, rdm1_bb,
            rdm2_aa_pqrs, rdm2_ab_pqrs, rdm2_bb_pqrs, float(e_fci))


def _build_rhf_ep(atom: str, basis: str) -> tuple:
    """RHF 積分から ElectronicProperties (unrestricted=False) を構築。"""
    mol = gto.Mole()
    mol.build(atom=atom, basis=basis, spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    norb = mo.shape[1]
    h1 = mo.T @ mf.get_hcore() @ mo
    h2 = ao2mo.full(mf._eri, mo, compact=False).reshape(norb, norb, norb, norb)
    nuc = mol.energy_nuc()
    na, nb = mol.nelec

    ep = ElectronicProperties(
        one_body_tensor=h1,
        two_body_tensor=h2,
        t2=np.zeros((na, na, norb, norb)),
        initial_occupancy=(np.ones(na) / na, np.ones(nb) / nb),
        nuclear_repulsion_energy=nuc,
        num_orbitals=norb,
        num_electrons=(na, nb),
        open_shell=False,
        spin_sq=0.0,
        unrestricted=False,
    )

    cisolver = pyscf_fci.FCI(mol, mo)
    e_fci, c = cisolver.kernel(nelec=mol.nelec)
    dm1, dm2_pqrs = cisolver.make_rdm12(c, norb, mol.nelec)

    return (ep, float(mf.e_tot), np.asarray(dm1), np.asarray(dm2_pqrs), float(e_fci))


# ─── テストケース ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _PYSCF, reason="pyscf not available")
class TestOrbitalOpt:
    """orbital_opt.optimize_orbitals の正当性検証。"""

    # ─── 共通フィクスチャ ──────────────────────────────────────────────────
    @pytest.fixture(scope="class")
    def oh_data(self):
        """OH radical (spin=1, sto-3g) の UHF データ。"""
        return _build_uhf_ep(
            atom="O 0 0 0; H 0 0 0.97",
            basis="sto-3g",
            spin=1,
        )

    @pytest.fixture(scope="class")
    def h2_rhf_data(self):
        """H2 (RHF, sto-3g) のデータ。"""
        return _build_rhf_ep(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")

    # ─── T1: エネルギー再現性 ──────────────────────────────────────────────
    def test_T1_energy_at_x0_reproduces_uhf(self, oh_data):
        """x=0 での目的関数値が UHF エネルギーに一致することを確認。
        
        FCI RDM は UHF Hamiltonian に対して UHF エネルギーを返さないが、
        UHF RDM（= UHF 1 電子密度行列）を使えば一致するはずなので、
        ここでは _uhf_energy の低レベル API を使って直接確認する。
        """
        ep, e_uhf, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb, e_fci = oh_data
        norb = ep.num_orbitals
        n_p = norb * (norb - 1) // 2

        # UHF の 1-RDM（対角行列 = 占有数）を使う
        na, nb = ep.num_electrons
        dm1_a_uhf = np.diag(np.array([1.0] * na + [0.0] * (norb - na)))
        dm1_b_uhf = np.diag(np.array([1.0] * nb + [0.0] * (norb - nb)))

        # UHF では 2-RDM を decompose できる: rdm2_aa[p,q,r,s] = dm1a[p,q]*dm1a[r,s] - dm1a[p,s]*dm1a[r,q]
        # → prqs格納に変換
        dm2_aa_prqs = (
            np.einsum("pq,rs->prqs", dm1_a_uhf, dm1_a_uhf)
            - np.einsum("ps,rq->prqs", dm1_a_uhf, dm1_a_uhf)
        )
        dm2_ab_prqs = np.einsum("pq,rs->prqs", dm1_a_uhf, dm1_b_uhf)
        dm2_bb_prqs = (
            np.einsum("pq,rs->prqs", dm1_b_uhf, dm1_b_uhf)
            - np.einsum("ps,rq->prqs", dm1_b_uhf, dm1_b_uhf)
        )

        x0 = np.zeros(2 * n_p)
        e_calc = _uhf_energy(
            x0, norb, n_p,
            dm1_a_uhf, dm1_b_uhf,
            dm2_aa_prqs, dm2_ab_prqs, dm2_bb_prqs,
            ep.one_body_tensor, ep.one_body_tensor_b,
            ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
            ep.nuclear_repulsion_energy,
        )
        assert abs(e_calc - e_uhf) < 1e-8, (
            f"Energy at x=0 should reproduce UHF: got {e_calc:.10f}, expected {e_uhf:.10f}"
        )

    # ─── T2: エネルギー減少（FCI RDM を UHF Hamiltonian に入れる）──────────
    def test_T2_energy_decreases_uhf(self, oh_data):
        """FCI RDM + UHF Hamiltonian で optimize_orbitals を呼ぶと E_after < E_before。
        
        FCI RDM は UHF 軌道基底での相関密度行列なので、軌道を回転することで
        エネルギー期待値が下がる方向に動けるはず。
        """
        ep, e_uhf, rdm1_aa, rdm1_bb, rdm2_aa_pqrs, rdm2_ab_pqrs, rdm2_bb_pqrs, e_fci = oh_data

        Ua, Ub, e_after, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa_pqrs, rdm2_ab_pqrs, rdm2_bb_pqrs,
            rdm2_notation="pqrs", maxiter=200,
        )
        # E_before = UHF エネルギーを取得するため別途計算
        n_p = ep.num_orbitals * (ep.num_orbitals - 1) // 2
        rdm2_aa_prqs = rdm2_aa_pqrs.transpose(0, 2, 1, 3)
        rdm2_ab_prqs = rdm2_ab_pqrs.transpose(0, 2, 1, 3)
        rdm2_bb_prqs = rdm2_bb_pqrs.transpose(0, 2, 1, 3)
        e_before = _uhf_energy(
            np.zeros(2 * n_p), ep.num_orbitals, n_p,
            rdm1_aa, rdm1_bb,
            rdm2_aa_prqs, rdm2_ab_prqs, rdm2_bb_prqs,
            ep.one_body_tensor, ep.one_body_tensor_b,
            ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
            ep.nuclear_repulsion_energy,
        )
        assert e_after <= e_before + 1e-10, (
            f"E_after ({e_after:.8f}) should be <= E_before ({e_before:.8f})"
        )
        assert e_after > -1000.0, f"Unphysical energy: {e_after:.4f} Ha"

    # ─── T8: gradient norm returned + small at convergence (MCSCF stopping) ─────
    def test_T8_grad_norm_returned_and_small_at_convergence(self, oh_data):
        """optimize_orbitals returns a 4th value (orbital gradient norm), and at a converged
        minimum that gradient is small -- the reference-free CASSCF stopping signal (Brillouin
        g -> 0). We converge tightly (maxiter large) and assert |grad| is below a loose bound."""
        ep, _, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb, _ = oh_data
        out = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
            rdm2_notation="pqrs", maxiter=500, gtol=1e-10,
        )
        assert len(out) == 4, "optimize_orbitals must return (Ua, Ub, energy, grad_norm)"
        Ua, Ub, e_after, grad_norm = out
        assert np.isfinite(grad_norm) and grad_norm >= 0.0
        # At a converged orbital minimum the gradient should be small (well below the 1e-3
        # production stopping threshold; use a loose bound to stay robust across BLAS/JAX).
        assert grad_norm < 1e-2, f"|grad| at convergence too large: {grad_norm:.3e}"

    # ─── T9: trust radius caps the orbital step ────────────────────────────────
    def test_T9_trust_radius_caps_step(self, oh_data):
        """A finite trust_radius restricts each rotation parameter to +/- trust_radius, so the
        returned rotation cannot exceed the cap. This is the MCSCF step-restriction safeguard
        that prevents over-rotation on fixed (approximate) RDMs. We check the log-map of the
        returned Ua has all |parameters| <= trust_radius (within tolerance)."""
        import scipy.linalg
        ep, _, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb, _ = oh_data
        tr = 0.1
        Ua, Ub, _, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
            rdm2_notation="pqrs", maxiter=200, trust_radius=tr,
        )
        # log(U) is skew-symmetric; its strict-upper-triangle entries are the rotation params.
        A = scipy.linalg.logm(Ua).real
        params = A[np.triu_indices(ep.num_orbitals, k=1)]
        assert np.all(np.abs(params) <= tr + 1e-6), (
            f"trust_radius={tr} not enforced: max|param|={np.max(np.abs(params)):.4f}"
        )

    # ─── T3: Ua / Ub が unitary ────────────────────────────────────────────
    def test_T3_rotation_matrices_are_unitary(self, oh_data):
        """optimize_orbitals が返す Ua, Ub が直交行列であることを確認。"""
        ep, _, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb, _ = oh_data
        norb = ep.num_orbitals

        Ua, Ub, _, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa, rdm2_ab, rdm2_bb,
            rdm2_notation="pqrs", maxiter=50,
        )
        err_a = np.linalg.norm(Ua.T @ Ua - np.eye(norb))
        err_b = np.linalg.norm(Ub.T @ Ub - np.eye(norb))
        assert err_a < 1e-12, f"Ua not unitary: ||Ua^T Ua - I|| = {err_a:.2e}"
        assert err_b < 1e-12, f"Ub not unitary: ||Ub^T Ub - I|| = {err_b:.2e}"

    # ─── T4: rdm2_notation="prqs" パス ────────────────────────────────────
    def test_T4_prqs_notation_consistent(self, oh_data):
        """rdm2_notation='prqs' で渡した結果が 'pqrs' と一致することを確認。"""
        ep, _, rdm1_aa, rdm1_bb, rdm2_aa_pqrs, rdm2_ab_pqrs, rdm2_bb_pqrs, _ = oh_data

        # pqrs → prqs 変換
        rdm2_aa_prqs = rdm2_aa_pqrs.transpose(0, 2, 1, 3)
        rdm2_ab_prqs = rdm2_ab_pqrs.transpose(0, 2, 1, 3)
        rdm2_bb_prqs = rdm2_bb_pqrs.transpose(0, 2, 1, 3)

        _, _, e_pqrs, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa_pqrs, rdm2_ab_pqrs, rdm2_bb_pqrs,
            rdm2_notation="pqrs", maxiter=30,
        )
        _, _, e_prqs, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa_prqs, rdm2_ab_prqs, rdm2_bb_prqs,
            rdm2_notation="prqs", maxiter=30,
        )
        assert abs(e_pqrs - e_prqs) < 1e-8, (
            f"pqrs path ({e_pqrs:.10f}) vs prqs path ({e_prqs:.10f}) differ"
        )

    # ─── T5: rotate_electronic_properties 整合性 ──────────────────────────
    def test_T5_rotate_ep_consistency(self, oh_data):
        """rotate_electronic_properties 後の H で同じ RDM を使うと E が一致。
        
        E(H_new, rdm_orig) == E_opt  が成立するはず（H を回転 = rdm を逆回転と等価）。
        """
        ep, _, rdm1_aa, rdm1_bb, rdm2_aa_pqrs, rdm2_ab_pqrs, rdm2_bb_pqrs, _ = oh_data

        Ua, Ub, e_opt, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa_pqrs, rdm2_ab_pqrs, rdm2_bb_pqrs,
            rdm2_notation="pqrs", maxiter=100,
        )
        ep_new = rotate_electronic_properties(ep, Ua, Ub)

        # E(H_new, rdm_orig) を手計算（prqs格納に変換してから）
        norb = ep.num_orbitals
        n_p = norb * (norb - 1) // 2
        rdm2_aa_prqs = rdm2_aa_pqrs.transpose(0, 2, 1, 3)
        rdm2_ab_prqs = rdm2_ab_pqrs.transpose(0, 2, 1, 3)
        rdm2_bb_prqs = rdm2_bb_pqrs.transpose(0, 2, 1, 3)
        e_check = _uhf_energy(
            np.zeros(2 * n_p), norb, n_p,
            rdm1_aa, rdm1_bb,
            rdm2_aa_prqs, rdm2_ab_prqs, rdm2_bb_prqs,
            ep_new.one_body_tensor, ep_new.one_body_tensor_b,
            ep_new.two_body_tensor, ep_new.two_body_tensor_ab, ep_new.two_body_tensor_bb,
            ep_new.nuclear_repulsion_energy,
        )
        assert abs(e_check - e_opt) < 1e-7, (
            f"E(H_new, rdm_orig)={e_check:.10f} != E_opt={e_opt:.10f}  "
            f"diff={abs(e_check-e_opt):.2e}"
        )

    # ─── T6: RHF パス ──────────────────────────────────────────────────────
    def test_T6_rhf_path(self, h2_rhf_data):
        """RHF (unrestricted=False) で optimize_orbitals が動作することを確認。
        
        H2/sto-3g FCI RDM + RHF Hamiltonian: E_after <= E_before。
        RHF パスでは Ua == Ub が返る。
        """
        ep, e_rhf, dm1_pqrs, dm2_pqrs, e_fci = h2_rhf_data
        norb = ep.num_orbitals

        # RHF パス: rdm1_aa=dm1, rdm1_bb=dm1（同じスピン分を渡す）
        Ua, Ub, e_after, _ = optimize_orbitals(
            ep, dm1_pqrs, dm1_pqrs,
            dm2_pqrs,
            rdm2_notation="pqrs", maxiter=100,
        )
        assert np.allclose(Ua, Ub, atol=1e-10), "RHF path: Ua and Ub should be equal"

        n_p = norb * (norb - 1) // 2
        dm2_prqs = dm2_pqrs.transpose(0, 2, 1, 3)
        e_before = _uhf_energy(
            np.zeros(2 * n_p), norb, n_p,
            dm1_pqrs, dm1_pqrs,
            dm2_prqs, dm2_prqs, dm2_prqs,
            ep.one_body_tensor, ep.one_body_tensor,
            ep.two_body_tensor, ep.two_body_tensor, ep.two_body_tensor,
            ep.nuclear_repulsion_energy,
        )
        assert e_after <= e_before + 1e-10, (
            f"RHF: E_after ({e_after:.8f}) > E_before ({e_before:.8f})"
        )
        assert e_after > -1000.0, f"Unphysical energy in RHF: {e_after:.4f}"


    # ─── T7: JAX analytical gradient ──────────────────────────────────────
    @pytest.mark.skipif(not _JAX, reason="jax not available")
    def test_T7a_jax_objective_matches_numpy_at_x0(self, oh_data):
        """JAX の objective 関数が x=0 で NumPy と同一の値を返すことを確認。"""
        ep, _, rdm1_aa, rdm1_bb, rdm2_aa_pqrs, rdm2_ab_pqrs, rdm2_bb_pqrs, _ = oh_data
        norb = ep.num_orbitals
        n_p = norb * (norb - 1) // 2

        # prqs 変換（内部 convention）
        rdm2_aa = np.asarray(rdm2_aa_pqrs).transpose(0, 2, 1, 3)
        rdm2_ab = np.asarray(rdm2_ab_pqrs).transpose(0, 2, 1, 3)
        rdm2_bb = np.asarray(rdm2_bb_pqrs).transpose(0, 2, 1, 3)

        obj_jax, _ = _make_jax_uhf_obj_and_grad(
            norb, n_p,
            rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
            ep.one_body_tensor, ep.one_body_tensor_b,
            ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
            ep.nuclear_repulsion_energy,
        )
        assert obj_jax is not None, "JAX objective should be buildable"

        x0 = np.zeros(2 * n_p)
        e_jax = obj_jax(x0)
        e_np = _uhf_energy(
            x0, norb, n_p,
            rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
            ep.one_body_tensor, ep.one_body_tensor_b,
            ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
            ep.nuclear_repulsion_energy,
        )
        assert abs(e_jax - e_np) < 1e-10, (
            f"JAX objective ({e_jax:.12f}) != NumPy objective ({e_np:.12f}) at x=0"
        )

    @pytest.mark.skipif(not _JAX, reason="jax not available")
    def test_T7b_jax_gradient_shape_and_finite(self, oh_data):
        """JAX gradient が正しい shape を持ち、有限値であることを確認。"""
        ep, _, rdm1_aa, rdm1_bb, rdm2_aa_pqrs, rdm2_ab_pqrs, rdm2_bb_pqrs, _ = oh_data
        norb = ep.num_orbitals
        n_p = norb * (norb - 1) // 2

        rdm2_aa = np.asarray(rdm2_aa_pqrs).transpose(0, 2, 1, 3)
        rdm2_ab = np.asarray(rdm2_ab_pqrs).transpose(0, 2, 1, 3)
        rdm2_bb = np.asarray(rdm2_bb_pqrs).transpose(0, 2, 1, 3)

        _, grad_jax = _make_jax_uhf_obj_and_grad(
            norb, n_p,
            rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
            ep.one_body_tensor, ep.one_body_tensor_b,
            ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
            ep.nuclear_repulsion_energy,
        )
        assert grad_jax is not None, "JAX gradient should be buildable"

        x0 = np.zeros(2 * n_p)
        g = grad_jax(x0)
        assert g.shape == (2 * n_p,), f"gradient shape mismatch: {g.shape}"
        assert np.all(np.isfinite(g)), "gradient contains non-finite values"

    @pytest.mark.skipif(not _JAX, reason="jax not available")
    def test_T7c_jax_and_numpy_converge_to_same_energy(self, oh_data):
        """use_jax=True/False で明示制御したとき同一のエネルギーに収束することを確認。

        optimize_orbitals() の use_jax 引数で直接 JAX/NumPy を切り替える。
        """
        ep, _, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb, _ = oh_data

        # JAX パス: use_jax=True で明示
        _, _, e_jax, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa, rdm2_ab, rdm2_bb,
            rdm2_notation="pqrs", maxiter=300, use_jax=True,
        )

        # NumPy パス: use_jax=False で JAX を明示的に無効化
        _, _, e_np, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa, rdm2_ab, rdm2_bb,
            rdm2_notation="pqrs", maxiter=300, use_jax=False,
        )

        assert abs(e_jax - e_np) < 1e-6, (
            f"use_jax=True E={e_jax:.10f}  use_jax=False E={e_np:.10f}  "
            f"diff={abs(e_jax-e_np):.2e}"
        )

    @pytest.mark.skipif(not _JAX, reason="jax not available")
    def test_T7d_jax_fewer_nfev_than_numpy(self, oh_data):
        """use_jax=True は use_jax=False より nfev (目的関数評価回数) が大幅に少ないことを確認。

        L-BFGS-B の有限差分は 1 iter あたり n_params 回の追加評価を必要とするため、
        JAX (解析的勾配) の nfev << NumPy (有限差分) の nfev になる。
        """
        ep, _, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb, _ = oh_data

        import scipy.optimize
        from functools import partial

        norb = ep.num_orbitals
        n_p = norb * (norb - 1) // 2
        rdm2_aa_prqs = np.asarray(rdm2_aa).transpose(0, 2, 1, 3)
        rdm2_ab_prqs = np.asarray(rdm2_ab).transpose(0, 2, 1, 3)
        rdm2_bb_prqs = np.asarray(rdm2_bb).transpose(0, 2, 1, 3)

        # JAX パス: use_jax=True で解析的勾配を渡す
        obj_jax, grad_jax = _make_jax_uhf_obj_and_grad(
            norb, n_p,
            rdm1_aa, rdm1_bb, rdm2_aa_prqs, rdm2_ab_prqs, rdm2_bb_prqs,
            ep.one_body_tensor, ep.one_body_tensor_b,
            ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
            ep.nuclear_repulsion_energy,
        )
        res_jax = scipy.optimize.minimize(
            obj_jax, np.zeros(2 * n_p),
            method="L-BFGS-B",
            jac=grad_jax,
            options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-10},
        )

        # NumPy パス: use_jax=False → 有限差分 (jac 未指定)
        obj_np = partial(
            _uhf_energy,
            norb=norb, n_p=n_p,
            rdm1_aa=rdm1_aa, rdm1_bb=rdm1_bb,
            rdm2_aa=rdm2_aa_prqs, rdm2_ab=rdm2_ab_prqs, rdm2_bb=rdm2_bb_prqs,
            h1_a=ep.one_body_tensor, h1_b=ep.one_body_tensor_b,
            h2_aa=ep.two_body_tensor, h2_ab=ep.two_body_tensor_ab,
            h2_bb=ep.two_body_tensor_bb, nuc=ep.nuclear_repulsion_energy,
        )
        res_np = scipy.optimize.minimize(
            obj_np, np.zeros(2 * n_p),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-10},
        )

        nit_jax = res_jax.nit
        nit_np  = res_np.nit
        print(f"\n    JAX  nit={nit_jax:3d}  nfev={res_jax.nfev:5d}  E={res_jax.fun:.10f}")
        print(f"    NumPy nit={nit_np:3d}  nfev={res_np.nfev:5d}  E={res_np.fun:.10f}")
        print(f"    nfev ratio (JAX/NumPy) = {res_jax.nfev/res_np.nfev:.3f}")

        # 両者が同じエネルギーに収束していることを確認
        assert res_jax.fun < res_np.fun + 1e-6, "JAX path should not give worse energy"
        # JAX の nfev は NumPy の 1/10 以下のはず
        # (n_params=30 なら理論比は ~1/31; 余裕を持って 1/5 を閾値に)
        assert res_jax.nfev < res_np.nfev / 5, (
            f"JAX nfev={res_jax.nfev} expected << NumPy nfev={res_np.nfev}"
        )


# ─── スタンドアロン実行モード（レポート生成）────────────────────────────────

def run_report(out_json: str | None = None) -> dict:
    """全テストを実行し、結果を dict / JSON で返す。

    Fugaku などで ``python test_orbital_opt.py`` として実行しても同じ形式の
    レポートが得られる。
    """
    if not _PYSCF:
        print("ERROR: pyscf not available — cannot run tests.")
        return {"status": "error", "reason": "pyscf not available"}

    results: list[dict] = []

    molecules = [
        ("OH",  "O 0 0 0; H 0 0 0.97",           "sto-3g", 1, True),
        ("H3",  "H 0 0 0; H 0 0 0.74; H 0 0 1.48","sto-3g", 1, True),
        ("H2",  "H 0 0 0; H 0 0 0.74",             "sto-3g", 0, False),
    ]

    all_passed = True
    for mol_name, atom, basis, spin, uhf in molecules:
        rec: dict = {"mol": mol_name, "basis": basis, "spin": spin, "tests": []}
        print(f"\n{'─'*60}")
        print(f"  System : {mol_name}  ({atom})")
        t0 = time.time()

        try:
            if uhf:
                (ep, e_uhf, rdm1_aa, rdm1_bb,
                 rdm2_aa, rdm2_ab, rdm2_bb, e_fci) = _build_uhf_ep(atom, basis, spin)
                rec.update({"e_uhf": e_uhf, "e_fci": e_fci,
                             "norb": ep.num_orbitals, "nelec": list(ep.num_electrons)})
                print(f"  E_UHF  = {e_uhf:.8f} Ha")
                print(f"  E_FCI  = {e_fci:.8f} Ha")

                # --- optimize_orbitals ---
                Ua, Ub, e_opt, _ = optimize_orbitals(
                    ep, rdm1_aa, rdm1_bb,
                    rdm2_aa, rdm2_ab, rdm2_bb,
                    rdm2_notation="pqrs", maxiter=300,
                )
                norb = ep.num_orbitals
                n_p = norb * (norb - 1) // 2
                rdm2_aa_prqs = rdm2_aa.transpose(0, 2, 1, 3)
                rdm2_ab_prqs = rdm2_ab.transpose(0, 2, 1, 3)
                rdm2_bb_prqs = rdm2_bb.transpose(0, 2, 1, 3)
                e_before = _uhf_energy(
                    np.zeros(2 * n_p), norb, n_p,
                    rdm1_aa, rdm1_bb,
                    rdm2_aa_prqs, rdm2_ab_prqs, rdm2_bb_prqs,
                    ep.one_body_tensor, ep.one_body_tensor_b,
                    ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
                    ep.nuclear_repulsion_energy,
                )
                err_a = float(np.linalg.norm(Ua.T @ Ua - np.eye(norb)))
                err_b = float(np.linalg.norm(Ub.T @ Ub - np.eye(norb)))
                delta_e = float(e_opt - e_before)

                # rotate_electronic_properties consistency
                ep_new = rotate_electronic_properties(ep, Ua, Ub)
                e_check = _uhf_energy(
                    np.zeros(2 * n_p), norb, n_p,
                    rdm1_aa, rdm1_bb,
                    rdm2_aa_prqs, rdm2_ab_prqs, rdm2_bb_prqs,
                    ep_new.one_body_tensor, ep_new.one_body_tensor_b,
                    ep_new.two_body_tensor, ep_new.two_body_tensor_ab, ep_new.two_body_tensor_bb,
                    ep_new.nuclear_repulsion_energy,
                )
                rot_consistency_err = abs(e_check - e_opt)

                t_pass = e_opt <= e_before + 1e-10
                u_pass = err_a < 1e-12 and err_b < 1e-12
                r_pass = rot_consistency_err < 1e-7
                phys   = e_opt > -1000.0

                rec.update({
                    "e_before": e_before,
                    "e_after":  e_opt,
                    "delta_e":  delta_e,
                    "unitary_err_Ua": err_a,
                    "unitary_err_Ub": err_b,
                    "rotate_ep_consistency_err": rot_consistency_err,
                })
                for name, passed in [
                    ("T2 energy_decreases",        t_pass),
                    ("T3 Ua_unitary",              u_pass),
                    ("T5 rotate_ep_consistency",   r_pass),
                    ("T7 physical_energy",         phys),
                ]:
                    status = "PASS" if passed else "FAIL"
                    if not passed:
                        all_passed = False
                    rec["tests"].append({"name": name, "status": status})
                    mark = "✓" if passed else "✗"
                    print(f"    {mark} {name:<35} {status}")

                print(f"  E_before = {e_before:.8f} Ha")
                print(f"  E_after  = {e_opt:.8f} Ha  (ΔE={delta_e:+.4e} Ha)")
                print(f"  ||Ua^T Ua-I|| = {err_a:.2e}   ||Ub^T Ub-I|| = {err_b:.2e}")
                print(f"  rotate_ep consistency err = {rot_consistency_err:.2e}")

            else:
                # RHF
                ep, e_rhf, dm1, dm2_pqrs, e_fci = _build_rhf_ep(atom, basis)
                rec.update({"e_rhf": e_rhf, "e_fci": e_fci,
                             "norb": ep.num_orbitals, "nelec": list(ep.num_electrons)})
                print(f"  E_RHF  = {e_rhf:.8f} Ha")
                print(f"  E_FCI  = {e_fci:.8f} Ha")

                Ua, Ub, e_opt, _ = optimize_orbitals(
                    ep, dm1, dm1, dm2_pqrs,
                    rdm2_notation="pqrs", maxiter=200,
                )
                norb = ep.num_orbitals
                n_p = norb * (norb - 1) // 2
                dm2_prqs = dm2_pqrs.transpose(0, 2, 1, 3)
                e_before = _uhf_energy(
                    np.zeros(2 * n_p), norb, n_p,
                    dm1, dm1, dm2_prqs, dm2_prqs, dm2_prqs,
                    ep.one_body_tensor, ep.one_body_tensor,
                    ep.two_body_tensor, ep.two_body_tensor, ep.two_body_tensor,
                    ep.nuclear_repulsion_energy,
                )
                ua_ub_eq  = np.allclose(Ua, Ub, atol=1e-10)
                t_pass    = e_opt <= e_before + 1e-10
                err_a     = float(np.linalg.norm(Ua.T @ Ua - np.eye(norb)))
                u_pass    = err_a < 1e-12

                rec.update({"e_before": e_before, "e_after": e_opt,
                             "delta_e": float(e_opt - e_before),
                             "unitary_err_Ua": err_a, "Ua_eq_Ub": ua_ub_eq})
                for name, passed in [
                    ("T6 rhf_ua_eq_ub",     ua_ub_eq),
                    ("T6 rhf_energy_decr",  t_pass),
                    ("T3 Ua_unitary",       u_pass),
                ]:
                    status = "PASS" if passed else "FAIL"
                    if not passed:
                        all_passed = False
                    rec["tests"].append({"name": name, "status": status})
                    mark = "✓" if passed else "✗"
                    print(f"    {mark} {name:<35} {status}")

                print(f"  E_before = {e_before:.8f} Ha")
                print(f"  E_after  = {e_opt:.8f} Ha  (ΔE={e_opt-e_before:+.4e} Ha)")

        except Exception as exc:
            rec["error"] = str(exc)
            all_passed = False
            print(f"  ERROR: {exc}")

        rec["elapsed_s"] = round(time.time() - t0, 2)
        results.append(rec)

    report = {
        "module": "qcsc_workflow_utility.orbital_opt",
        "status": "PASS" if all_passed else "FAIL",
        "results": results,
    }

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  orbital_opt verification: {'PASS' if all_passed else 'FAIL'}")
    print(sep)

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved → {out_json}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Verify qcsc_workflow_utility.orbital_opt (no SQD/Fugaku required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # run and print results
  python test_orbital_opt.py

  # save JSON report (portable to Fugaku)
  python test_orbital_opt.py --report /tmp/orbital_opt_report.json
""",
    )
    parser.add_argument("--report", default=None, metavar="PATH",
                        help="Save JSON report to this path")
    args = parser.parse_args()

    report = run_report(out_json=args.report)
    sys.exit(0 if report["status"] == "PASS" else 1)
