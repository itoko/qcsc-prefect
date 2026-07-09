"""Integration tests: orbital_opt.py wired into the sbd pipeline.

These tests verify:
  1. The notation contract between solver_job (writes prqs-storage RDMs) and
     optimize_orbitals (must be called with rdm2_notation="prqs").
  2. The full local pipeline: PySCF integrals → solve_fermion → optimize_orbitals
     → rotate_electronic_properties gives a physically meaningful result.

No Fugaku, no SQD binary, no Prefect server required.
PySCF + qiskit_addon_sqd + scipy suffice.

Run:
    cd qcsc-prefect
    conda activate qcsc
    pytest algorithms/sbd/tests/test_orbital_opt_integration.py -v
    # or standalone:
    python algorithms/sbd/tests/test_orbital_opt_integration.py --report /tmp/orbopt_integration.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# ─── path setup ───────────────────────────────────────────────────────────────
_SBD_ROOT = Path(__file__).resolve().parents[1]
_UTIL_SRC = _SBD_ROOT.parent / "qcsc_workflow_utility" / "src"
sys.path.insert(0, str(_UTIL_SRC))

# Silence Prefect logger before importing sbd modules
import qcsc_workflow_utility.chem as _chem_mod
_chem_mod.get_run_logger = lambda: mock.MagicMock()

try:
    from sbd import solver_job as _sj
    _sj.get_run_logger = lambda: mock.MagicMock()
except Exception:
    pass

# ─── optional imports ─────────────────────────────────────────────────────────
try:
    from pyscf import ao2mo, fci as pyscf_fci, gto, scf
    _PYSCF = True
except ImportError:
    _PYSCF = False

try:
    from qiskit_addon_sqd.fermion import solve_fermion
    _SQD = True
except ImportError:
    _SQD = False

from qcsc_workflow_utility.orbital_opt import (
    _uhf_energy,
    optimize_orbitals,
    rotate_electronic_properties,
)
from qcsc_workflow_utility.chem import ElectronicProperties


# ─── helpers ──────────────────────────────────────────────────────────────────

def _build_ep(atom: str, basis: str, spin: int) -> tuple:
    """UHF ElectronicProperties + integrals for small test molecules."""
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
        one_body_tensor=h1_a, two_body_tensor=h2_aa,
        t2=np.zeros((na, na, norb, norb)),
        initial_occupancy=(np.ones(na) / na, np.ones(nb) / nb),
        nuclear_repulsion_energy=nuc, num_orbitals=norb,
        num_electrons=(na, nb), open_shell=(na != nb),
        spin_sq=float(spin / 2 * (spin / 2 + 1)),
        unrestricted=True,
        one_body_tensor_b=h1_b, two_body_tensor_ab=h2_ab, two_body_tensor_bb=h2_bb,
        t2_ab=np.zeros((na, nb, norb, norb)),
        t2_bb=np.zeros((nb, nb, norb, norb)),
    )
    return ep, float(mf.e_tot), nuc, na, nb, norb


def _fci_rdm_pqrs(ep: ElectronicProperties) -> tuple:
    """FCI RDMs in pqrs-storage (PySCF/solve_fermion convention)."""
    mol = gto.Mole()
    mol.build(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = scf.UHF(mol).run()
    norb = ep.num_orbitals
    cisolver = pyscf_fci.FCI(mol, mf.mo_coeff[0])
    e_fci, c = cisolver.kernel(nelec=mol.nelec)
    dm1s, dm2s = cisolver.make_rdm12s(c, norb, mol.nelec)
    return (float(e_fci),
            np.asarray(dm1s[0]), np.asarray(dm1s[1]),
            np.asarray(dm2s[0]),  # aa, pqrs
            np.asarray(dm2s[1]),  # ab, pqrs
            np.asarray(dm2s[2]))  # bb, pqrs


# ─── tests ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _PYSCF, reason="pyscf not available")
class TestNotationContract:
    """Verify that prqs↔pqrs notation gives physically correct energies."""

    @pytest.fixture(scope="class")
    def oh_ep(self):
        return _build_ep("O 0 0 0; H 0 0 0.97", "sto-3g", 1)

    def test_pqrs_energy_reproduces_uhf(self, oh_ep):
        """E at x=0 with pqrs-storage RDM reproduces UHF energy."""
        ep, e_uhf, nuc, na, nb, norb = oh_ep
        cisolver = pyscf_fci.FCI(
            gto.Mole().build(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0),
        )
        # Use UHF 1-RDM (idempotent) to check
        dm1_a = np.diag([1.0]*na + [0.0]*(norb-na))
        dm1_b = np.diag([1.0]*nb + [0.0]*(norb-nb))
        dm2_aa_prqs = (np.einsum("pq,rs->prqs", dm1_a, dm1_a)
                       - np.einsum("ps,rq->prqs", dm1_a, dm1_a))
        dm2_ab_prqs = np.einsum("pq,rs->prqs", dm1_a, dm1_b)
        dm2_bb_prqs = (np.einsum("pq,rs->prqs", dm1_b, dm1_b)
                       - np.einsum("ps,rq->prqs", dm1_b, dm1_b))
        n_p = norb * (norb - 1) // 2
        e_calc = _uhf_energy(
            np.zeros(2 * n_p), norb, n_p,
            dm1_a, dm1_b,
            dm2_aa_prqs, dm2_ab_prqs, dm2_bb_prqs,
            ep.one_body_tensor, ep.one_body_tensor_b,
            ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
            nuc,
        )
        assert abs(e_calc - e_uhf) < 1e-7, (
            f"UHF energy at x=0: calc={e_calc:.10f} ref={e_uhf:.10f}"
        )

    def test_wrong_notation_gives_unphysical_energy(self, oh_ep):
        """pqrs-storage RDM passed as prqs-storage gives E << -1000 Ha (the previous bug)."""
        ep, e_uhf, nuc, na, nb, norb = oh_ep
        mol = gto.Mole()
        mol.build(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
        mf = scf.UHF(mol).run()
        cisolver = pyscf_fci.FCI(mol, mf.mo_coeff[0])
        _, c = cisolver.kernel(nelec=mol.nelec)
        dm1s, dm2s = cisolver.make_rdm12s(c, norb, mol.nelec)
        rdm1_aa = np.asarray(dm1s[0])
        rdm1_bb = np.asarray(dm1s[1])
        # pqrs-storage (PySCF native) passed as prqs — deliberately wrong
        rdm2_aa_pqrs_as_prqs = np.asarray(dm2s[0])  # do NOT transpose
        n_p = norb * (norb - 1) // 2
        e_wrong = _uhf_energy(
            np.zeros(2 * n_p), norb, n_p,
            rdm1_aa, rdm1_bb,
            rdm2_aa_pqrs_as_prqs, rdm2_aa_pqrs_as_prqs, rdm2_aa_pqrs_as_prqs,
            ep.one_body_tensor, ep.one_body_tensor_b,
            ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
            nuc,
        )
        # The wrong-notation energy is far outside the physical range
        assert e_wrong < -100.0, (
            f"Expected unphysical energy with wrong notation, got {e_wrong:.4f} Ha"
        )

    def test_prqs_storage_optimize_orbitals(self, oh_ep):
        """optimize_orbitals with rdm2_notation='prqs' gives physical energy."""
        ep, e_uhf, nuc, na, nb, norb = oh_ep
        mol = gto.Mole()
        mol.build(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
        mf = scf.UHF(mol).run()
        cisolver = pyscf_fci.FCI(mol, mf.mo_coeff[0])
        e_fci, c = cisolver.kernel(nelec=mol.nelec)
        dm1s, dm2s = cisolver.make_rdm12s(c, norb, mol.nelec)
        rdm1_aa = np.asarray(dm1s[0])
        rdm1_bb = np.asarray(dm1s[1])
        # Simulate what solver_job writes: prqs-storage
        rdm2_aa_prqs = np.asarray(dm2s[0]).transpose(0, 2, 1, 3)
        rdm2_ab_prqs = np.asarray(dm2s[1]).transpose(0, 2, 1, 3)
        rdm2_bb_prqs = np.asarray(dm2s[2]).transpose(0, 2, 1, 3)

        Ua, Ub, e_opt, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa_prqs, rdm2_ab_prqs, rdm2_bb_prqs,
            rdm2_notation="prqs", maxiter=200,
        )
        assert e_opt > -1000.0, f"Unphysical energy: {e_opt:.4f}"
        assert e_opt < e_uhf + 0.1, f"E_opt ({e_opt:.6f}) far above UHF ({e_uhf:.6f})"
        err_a = np.linalg.norm(Ua.T @ Ua - np.eye(norb))
        assert err_a < 1e-12, f"||Ua^T Ua - I|| = {err_a:.2e}"


@pytest.mark.skipif(not (_PYSCF and _SQD), reason="pyscf or qiskit_addon_sqd not available")
class TestFullLocalPipeline:
    """PySCF integrals → solve_fermion RDMs → optimize_orbitals → rotate_electronic_properties."""

    @pytest.fixture(scope="class")
    def oh_pipeline(self):
        """OH radical: one SQD iteration with the local solve_fermion diagonalizer."""
        ep, e_uhf, nuc, na, nb, norb = _build_ep("O 0 0 0; H 0 0 0.97", "sto-3g", 1)

        # Tiny CI space — all strings must have the same hamming weight per spin channel
        # HF = na ones in the lowest na bits; single excitation = shift one bit
        hf_a = int((1 << na) - 1)           # 0b11111 for na=5
        # next lowest-energy string: excite HOMO→LUMO (bit na-1 → bit na)
        excited_a = int((hf_a ^ (1 << (na - 1))) | (1 << na))
        ci_a = np.array([hf_a, excited_a], dtype=np.int64)
        ci_b = np.array([int((1 << nb) - 1)], dtype=np.int64)
        hcore_avg = (ep.one_body_tensor + ep.one_body_tensor_b) / 2.0
        eri_avg   = (ep.two_body_tensor + ep.two_body_tensor_ab
                     + ep.two_body_tensor_ab.transpose(2, 3, 0, 1)
                     + ep.two_body_tensor_bb) / 4.0
        spin_sq_t = float((na - nb) / 2 * ((na - nb) / 2 + 1))
        energy_raw, sci_state, _, sq = solve_fermion(
            (ci_a, ci_b), hcore_avg, eri_avg, open_shell=True, spin_sq=spin_sq_t,
        )
        sci_e = float(energy_raw) + nuc
        rdm1_aa, rdm1_bb = sci_state.rdm(rank=1, spin_summed=False)
        rdm2_aa, rdm2_ab, rdm2_bb = sci_state.rdm(rank=2, spin_summed=False)
        return (ep, e_uhf, nuc, norb,
                np.asarray(rdm1_aa), np.asarray(rdm1_bb),
                np.asarray(rdm2_aa), np.asarray(rdm2_ab), np.asarray(rdm2_bb),
                sci_e, float(sq))

    def test_solve_fermion_energy_is_physical(self, oh_pipeline):
        ep, e_uhf, nuc, norb, *_, sci_e, sq = oh_pipeline
        assert sci_e > -1000.0
        assert sci_e < 0.0

    def test_optimize_orbitals_with_pqrs_rdm_from_solve_fermion(self, oh_pipeline):
        """solve_fermion returns pqrs-storage → optimize_orbitals(rdm2_notation='pqrs') is correct."""
        ep, e_uhf, nuc, norb, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb, sci_e, _ = oh_pipeline

        Ua, Ub, e_opt, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa, rdm2_ab, rdm2_bb,
            rdm2_notation="pqrs", maxiter=200,
        )
        # Energy must be physical
        assert e_opt > -1000.0, f"Unphysical: {e_opt:.4f}"
        # Ua, Ub unitary
        assert np.linalg.norm(Ua.T @ Ua - np.eye(norb)) < 1e-12
        assert np.linalg.norm(Ub.T @ Ub - np.eye(norb)) < 1e-12

    def test_rotate_ep_consistency_after_pipeline(self, oh_pipeline):
        """E(H_new, rdm_orig) == E_opt after rotate_electronic_properties."""
        ep, e_uhf, nuc, norb, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb, sci_e, _ = oh_pipeline

        Ua, Ub, e_opt, _ = optimize_orbitals(
            ep, rdm1_aa, rdm1_bb,
            rdm2_aa, rdm2_ab, rdm2_bb,
            rdm2_notation="pqrs", maxiter=100,
        )
        ep_new = rotate_electronic_properties(ep, Ua, Ub)

        n_p = norb * (norb - 1) // 2
        # rdm2 already in pqrs; convert to prqs for _uhf_energy
        rdm2_aa_prqs = rdm2_aa.transpose(0, 2, 1, 3)
        rdm2_ab_prqs = rdm2_ab.transpose(0, 2, 1, 3)
        rdm2_bb_prqs = rdm2_bb.transpose(0, 2, 1, 3)
        e_check = _uhf_energy(
            np.zeros(2 * n_p), norb, n_p,
            rdm1_aa, rdm1_bb,
            rdm2_aa_prqs, rdm2_ab_prqs, rdm2_bb_prqs,
            ep_new.one_body_tensor, ep_new.one_body_tensor_b,
            ep_new.two_body_tensor, ep_new.two_body_tensor_ab, ep_new.two_body_tensor_bb,
            ep_new.nuclear_repulsion_energy,
        )
        assert abs(e_check - e_opt) < 1e-7, (
            f"E(H_new, rdm_orig)={e_check:.10f}  E_opt={e_opt:.10f}  diff={abs(e_check-e_opt):.2e}"
        )


# ─── standalone report mode ───────────────────────────────────────────────────

def run_report(out_json: str | None = None) -> dict:
    """Run all checks and return / save a JSON report."""
    if not _PYSCF:
        print("ERROR: pyscf not available")
        return {"status": "error", "reason": "pyscf not available"}

    records = []
    all_pass = True

    print("\n" + "="*60)
    print("  orbital_opt integration tests (sbd pipeline notation)")
    print("="*60)

    molecules = [("OH", "O 0 0 0; H 0 0 0.97", "sto-3g", 1)]

    for mol_name, atom, basis, spin in molecules:
        print(f"\n  System: {mol_name}")
        rec: dict = {"mol": mol_name, "tests": []}
        t0 = time.time()
        try:
            ep, e_uhf, nuc, na, nb, norb = _build_ep(atom, basis, spin)
            rec.update({"e_uhf": e_uhf, "norb": norb, "nelec": [na, nb]})

            mol = gto.Mole()
            mol.build(atom=atom, basis=basis, spin=spin, verbose=0)
            mf = scf.UHF(mol).run()
            cisolver = pyscf_fci.FCI(mol, mf.mo_coeff[0])
            e_fci, c = cisolver.kernel(nelec=mol.nelec)
            dm1s, dm2s = cisolver.make_rdm12s(c, norb, mol.nelec)
            rdm1_aa = np.asarray(dm1s[0])
            rdm1_bb = np.asarray(dm1s[1])
            rdm2_pqrs = [np.asarray(dm2s[i]) for i in range(3)]
            rdm2_prqs = [d.transpose(0, 2, 1, 3) for d in rdm2_pqrs]
            rec["e_fci"] = float(e_fci)

            # ── T_NOTATION: wrong notation gives unphysical E ──────────────
            n_p = norb * (norb - 1) // 2
            e_wrong = _uhf_energy(
                np.zeros(2 * n_p), norb, n_p,
                rdm1_aa, rdm1_bb,
                rdm2_pqrs[0], rdm2_pqrs[0], rdm2_pqrs[0],   # pqrs passed as prqs
                ep.one_body_tensor, ep.one_body_tensor_b,
                ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb,
                nuc,
            )
            t_notation = e_wrong < -100.0
            mark = "✓" if t_notation else "✗"
            if not t_notation: all_pass = False
            print(f"    {mark} T_NOTATION wrong_notation_unphysical   {'PASS' if t_notation else 'FAIL'}"
                  f"  (E_wrong={e_wrong:.2f} Ha)")
            rec["tests"].append({"name": "T_NOTATION_wrong_is_unphysical",
                                  "status": "PASS" if t_notation else "FAIL",
                                  "e_wrong": float(e_wrong)})

            # ── T_PRQS: prqs-notation path (solver_job convention) ─────────
            Ua_p, Ub_p, e_prqs, _ = optimize_orbitals(
                ep, rdm1_aa, rdm1_bb,
                *rdm2_prqs, rdm2_notation="prqs", maxiter=200,
            )
            t_prqs_phys = e_prqs > -1000.0
            t_prqs_unit = (np.linalg.norm(Ua_p.T @ Ua_p - np.eye(norb)) < 1e-12
                           and np.linalg.norm(Ub_p.T @ Ub_p - np.eye(norb)) < 1e-12)
            for name, passed in [("T_PRQS_physical", t_prqs_phys),
                                  ("T_PRQS_unitary",  t_prqs_unit)]:
                mark = "✓" if passed else "✗"
                if not passed: all_pass = False
                print(f"    {mark} {name:<38} {'PASS' if passed else 'FAIL'}")
                rec["tests"].append({"name": name, "status": "PASS" if passed else "FAIL"})
            rec.update({"e_opt_prqs": float(e_prqs)})
            print(f"      E_opt (prqs) = {e_prqs:.8f} Ha  (FCI = {e_fci:.8f})")

            # ── T_PQRS: pqrs-notation path (solve_fermion convention) ──────
            Ua_q, Ub_q, e_pqrs, _ = optimize_orbitals(
                ep, rdm1_aa, rdm1_bb,
                *rdm2_pqrs, rdm2_notation="pqrs", maxiter=200,
            )
            t_agree = abs(e_pqrs - e_prqs) < 1e-8
            mark = "✓" if t_agree else "✗"
            if not t_agree: all_pass = False
            print(f"    {mark} T_PQRS_consistent_with_prqs           {'PASS' if t_agree else 'FAIL'}"
                  f"  (diff={abs(e_pqrs-e_prqs):.2e})")
            rec["tests"].append({"name": "T_PQRS_consistent",
                                  "status": "PASS" if t_agree else "FAIL"})
            rec["e_opt_pqrs"] = float(e_pqrs)

            # ── T_ROT: rotate_ep consistency ───────────────────────────────
            ep_new = rotate_electronic_properties(ep, Ua_p, Ub_p)
            e_check = _uhf_energy(
                np.zeros(2 * n_p), norb, n_p,
                rdm1_aa, rdm1_bb,
                *rdm2_prqs,
                ep_new.one_body_tensor, ep_new.one_body_tensor_b,
                ep_new.two_body_tensor, ep_new.two_body_tensor_ab, ep_new.two_body_tensor_bb,
                ep_new.nuclear_repulsion_energy,
            )
            t_rot = abs(e_check - e_prqs) < 1e-7
            mark = "✓" if t_rot else "✗"
            if not t_rot: all_pass = False
            print(f"    {mark} T_ROT_consistency                      {'PASS' if t_rot else 'FAIL'}"
                  f"  (err={abs(e_check-e_prqs):.2e})")
            rec["tests"].append({"name": "T_ROT_consistency",
                                  "status": "PASS" if t_rot else "FAIL",
                                  "rotate_ep_err": float(abs(e_check - e_prqs))})

        except Exception as exc:
            rec["error"] = str(exc)
            all_pass = False
            print(f"  ERROR: {exc}")

        rec["elapsed_s"] = round(time.time() - t0, 2)
        records.append(rec)

    print("\n" + "="*60)
    print(f"  Result: {'PASS' if all_pass else 'FAIL'}")
    print("="*60)

    report = {
        "module": "sbd.main + qcsc_workflow_utility.orbital_opt",
        "description": (
            "Notation contract: solver_job writes prqs-storage; "
            "optimize_orbitals must be called with rdm2_notation='prqs'."
        ),
        "status": "PASS" if all_pass else "FAIL",
        "results": records,
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved → {out_json}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Integration tests: orbital_opt + sbd pipeline notation contract",
    )
    parser.add_argument("--report", default=None, metavar="PATH")
    args = parser.parse_args()
    report = run_report(out_json=args.report)
    sys.exit(0 if report["status"] == "PASS" else 1)
