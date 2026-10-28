"""<S^2> from the native-solver RDM layout (orbital_opt.spin_square_from_rdms).

Reference everywhere: PySCF ``fci.direct_uhf`` RDMs (``make_rdm12s``) of the same CI vector and
``fci.spin_op.spin_square_general`` with the true alpha/beta MO overlap.

The solver's file layout ("prqs" for every 2-RDM block, as ``sbd.solver_job._read_rdm_file``
documents) is pinned by ``test_native_kernel_layout``: the patched native-kernel replica in
``algorithms/sbd/native/patches/test_rdm_vs_pyscf.py`` accumulates the flat arrays exactly as the C++
kernel does, and they are laid out exactly as ``main.cc`` writes them (row-major over the native
(io, jo, ia, ja) index). The other tests convert PySCF RDMs to that layout with transpose (0,2,1,3).
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, scf
from pyscf.fci.spin_op import spin_square_general

from qcsc_workflow_utility.orbital_opt import rotate_electronic_properties, spin_square_from_rdms


def _uhf(atom, spin=0, basis="sto-3g", break_symmetry=True):
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    if break_symmetry and spin == 0:
        dm = mf.get_init_guess()
        dm[0][0, 0] += 0.3
        dm[1][0, 0] -= 0.3
        mf.kernel(dm0=dm)
    else:
        mf.kernel()
    return mol, mf


def _ints(mol, mf):
    moa, mob = mf.mo_coeff
    n = moa.shape[1]
    hc = mf.get_hcore()
    g = lambda a, b: ao2mo.general(mol, (a, a, b, b), compact=False).reshape((n,) * 4)
    return (moa.T @ hc @ moa, mob.T @ hc @ mob), (g(moa, moa), g(moa, mob), g(mob, mob)), n


def _to_sbd(d1a, d1b, d2aa, d2ab, d2bb):
    """PySCF make_rdm12s -> solver file layout (prqs for every block; transpose is an involution)."""
    return d1a, d1b, d2aa.transpose(0, 2, 1, 3), d2ab.transpose(0, 2, 1, 3), d2bb.transpose(0, 2, 1, 3)


def _reference(cis, ci, n, nelec, mo, ovlp):
    (d1a, d1b), (d2aa, d2ab, d2bb) = cis.make_rdm12s(ci, n, nelec)
    return (d1a, d1b, d2aa, d2ab, d2bb), spin_square_general(d1a, d1b, d2aa, d2ab, d2bb, mo, ovlp)[0]


CASES = {
    "h4_bs_singlet": ("H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8", 0),
    "oh_doublet": ("O 0 0 0; H 0 0 0.97", 1),
    "h4_triplet": ("H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", 2),
}


@pytest.mark.parametrize("name", CASES)
def test_fci_state_matches_pyscf(name):
    atom, spin = CASES[name]
    mol, mf = _uhf(atom, spin)
    h1, eri, n = _ints(mol, mf)
    moa, mob = mf.mo_coeff
    ovlp = mf.get_ovlp()
    cis = fci.direct_uhf.FCISolver()
    _, ci = cis.kernel(h1, eri, n, mol.nelec)
    rdms, ref = _reference(cis, ci, n, mol.nelec, (moa, mob), ovlp)
    s_ab = moa.T @ ovlp @ mob
    got = spin_square_from_rdms(*_to_sbd(*rdms), ab_overlap=s_ab)
    assert got == pytest.approx(ref, abs=1e-10)
    na, nb = mol.nelec
    s = 0.5 * (na - nb)
    assert got == pytest.approx(s * (s + 1), abs=1e-6)  # exact FCI is a spin eigenstate


def test_truncated_subspace_matches_pyscf_and_needs_overlap():
    """Selected-CI-like state (product subspace of the top strings) in BS-UHF orbitals: spin
    contaminated, and the identity overlap gives a different (wrong) value."""
    mol, mf = _uhf("H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8; H 0 0 6.4; H 0 0 8.0")
    h1, eri, n = _ints(mol, mf)
    moa, mob = mf.mo_coeff
    ovlp = mf.get_ovlp()
    cis = fci.direct_uhf.FCISolver()
    _, ci_full = cis.kernel(h1, eri, n, mol.nelec)
    keep_a = np.argsort(-(ci_full ** 2).sum(1))[:8]
    keep_b = np.argsort(-(ci_full ** 2).sum(0))[:8]
    h2e = cis.absorb_h1e(h1, eri, n, mol.nelec, 0.5)
    basis = [(i, j) for i in keep_a for j in keep_b]
    H = np.zeros((len(basis), len(basis)))
    for c, (i, j) in enumerate(basis):
        v = np.zeros(ci_full.shape)
        v[i, j] = 1.0
        hv = cis.contract_2e(h2e, v, n, mol.nelec)
        H[:, c] = [hv[a, b] for a, b in basis]
    c0 = np.linalg.eigh(H)[1][:, 0]
    ci = np.zeros(ci_full.shape)
    for c, (i, j) in enumerate(basis):
        ci[i, j] = c0[c]
    rdms, ref = _reference(cis, ci, n, mol.nelec, (moa, mob), ovlp)
    got = spin_square_from_rdms(*_to_sbd(*rdms), ab_overlap=moa.T @ ovlp @ mob)
    assert got == pytest.approx(ref, abs=1e-10)
    assert 0.1 < got < 1.0  # genuinely spin contaminated
    wrong = spin_square_from_rdms(*_to_sbd(*rdms), ab_overlap=None)
    assert abs(wrong - got) > 0.05


def test_identity_overlap_for_rhf_orbitals():
    """alpha = beta orbitals (rhf2uhf input): ab_overlap=None is exact."""
    mol = gto.M(atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    n = mo.shape[1]
    hc = mo.T @ mf.get_hcore() @ mo
    g = ao2mo.full(mol, mo, compact=False).reshape((n,) * 4)
    cis = fci.direct_uhf.FCISolver()
    _, ci = cis.kernel((hc, hc), (g, g, g), n, mol.nelec)
    ci = ci + 0.05 * np.random.default_rng(0).standard_normal(ci.shape)  # not an S eigenstate
    ci /= np.linalg.norm(ci)
    rdms, ref = _reference(cis, ci, n, mol.nelec, (mo, mo), mf.get_ovlp())
    assert spin_square_from_rdms(*_to_sbd(*rdms)) == pytest.approx(ref, abs=1e-10)


def test_solver_text_files_roundtrip(tmp_path):
    """Write the SBD layout exactly as main.cc does (r[i + L j + L^2 k + L^3 l], loop i,j,k,l) and
    read it back with sbd.solver_job._read_rdm_file."""
    sj = pytest.importorskip("sbd.solver_job")
    mol, mf = _uhf("O 0 0 0; H 0 0 0.97", spin=1)
    h1, eri, n = _ints(mol, mf)
    moa, mob = mf.mo_coeff
    ovlp = mf.get_ovlp()
    cis = fci.direct_uhf.FCISolver()
    _, ci = cis.kernel(h1, eri, n, mol.nelec)
    rdms, ref = _reference(cis, ci, n, mol.nelec, (moa, mob), ovlp)
    sbd = _to_sbd(*rdms)
    names = ["rdm1_a", "rdm1_b", "rdm2_aa", "rdm2_ab", "rdm2_bb"]
    for name, arr in zip(names, sbd):
        # main.cc: native array r[i + L*j (+ L^2*k + L^3*l)] holds element [i,j(,k,l)] of the
        # block; it loops i outermost and writes r[...] -> file order is row-major over (i,j,k,l).
        flat = arr.reshape(-1)  # row-major == the loop order main.cc writes
        np.savetxt(tmp_path / f"{name}.txt", flat)
    read = [sj._read_rdm_file(tmp_path / f"{name}.txt", n, rank=2 if name.startswith("rdm1") else 4)
            for name in names]
    got = spin_square_from_rdms(*read, ab_overlap=moa.T @ ovlp @ mob)
    assert got == pytest.approx(ref, abs=1e-10)


def _main_cc_layout(flat, L, rank):
    """File array exactly as main.cc writes the native flat r: element [i, j(, k, l)] =
    r[i + L j (+ L^2 k + L^3 l)], loop order i outermost (row-major on read)."""
    if rank == 2:
        return np.array([[flat[i + L * j] for j in range(L)] for i in range(L)])
    out = np.empty((L,) * 4)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for m in range(L):
                    out[i, j, k, m] = flat[i + L * j + L * L * k + L * L * L * m]
    return out


def test_native_kernel_layout():
    """The replica of the (patched) native RDM kernel + main.cc's write order -> <S^2> == PySCF."""
    import sys
    from pathlib import Path

    patches = Path(__file__).resolve().parents[2] / "sbd" / "native" / "patches"
    sys.path.insert(0, str(patches))
    try:
        import test_rdm_vs_pyscf as native
    finally:
        sys.path.remove(str(patches))
    for _tag, atom, basis, spin in native.CASES:
        ref = native._build_reference(atom, basis, spin)
        L = ref["norb"]
        kernel = native._RDMKernel(L, patched=True)
        kernel.build(ref["dets"], ref["weights"])
        files = (
            _main_cc_layout(kernel.onebody[0], L, 2), _main_cc_layout(kernel.onebody[1], L, 2),
            _main_cc_layout(kernel.twobody[0], L, 4),  # aa
            _main_cc_layout(kernel.twobody[2], L, 4),  # ab (main.cc writes two_p_rdm[2])
            _main_cc_layout(kernel.twobody[3], L, 4),  # bb
        )
        mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
        mf = scf.UHF(mol).run()
        moa, mob = mf.mo_coeff
        ovlp = mf.get_ovlp()
        s2_ref = spin_square_general(ref["dm1a"], ref["dm1b"], ref["dm2aa"], ref["dm2ab"],
                                     ref["dm2bb"], (moa, mob), ovlp)[0]
        got = spin_square_from_rdms(*files, ab_overlap=moa.T @ ovlp @ mob)
        assert got == pytest.approx(s2_ref, abs=1e-9), _tag


def test_rotation_tracks_overlap():
    """After rotating both spins' orbitals, <S^2> of the same state expressed in the rotated
    orbitals (with the rotated ab_overlap) is unchanged."""
    from qcsc_workflow_utility.chem import ElectronicProperties

    mol, mf = _uhf("H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8")
    h1, eri, n = _ints(mol, mf)
    moa, mob = mf.mo_coeff
    ovlp = mf.get_ovlp()
    s_ab = moa.T @ ovlp @ mob
    cis = fci.direct_uhf.FCISolver()
    _, ci_full = cis.kernel(h1, eri, n, mol.nelec)
    ci = ci_full + 0.1 * np.random.default_rng(1).standard_normal(ci_full.shape)
    ci /= np.linalg.norm(ci)
    rdms, ref = _reference(cis, ci, n, mol.nelec, (moa, mob), ovlp)

    rng = np.random.default_rng(2)
    ua = np.linalg.qr(rng.standard_normal((n, n)))[0]
    ub = np.linalg.qr(rng.standard_normal((n, n)))[0]
    ep = ElectronicProperties(
        one_body_tensor=h1[0], one_body_tensor_b=h1[1], two_body_tensor=eri[0],
        two_body_tensor_ab=eri[1], two_body_tensor_bb=eri[2], t2=np.zeros((0, 0, 0, 0)),
        initial_occupancy=(np.zeros(n), np.zeros(n)), nuclear_repulsion_energy=0.0,
        num_orbitals=n, num_electrons=mol.nelec, open_shell=False, spin_sq=0.0,
        unrestricted=True, ab_overlap=s_ab,
    )
    ep_rot = rotate_electronic_properties(ep, ua, ub)
    np.testing.assert_allclose(ep_rot.ab_overlap, ua.T @ s_ab @ ub, atol=1e-14)

    # same state in the rotated orbitals: phi'_P = sum_p phi_p U[p,P]  ->  RDMs transform with U
    d1a, d1b, d2aa, d2ab, d2bb = rdms
    r1 = lambda d, u: u.T @ d @ u
    r2 = lambda d, ux, uy: np.einsum("pqrs,pP,qQ,rR,sS->PQRS", d, ux, ux, uy, uy, optimize=True)
    rot = (r1(d1a, ua), r1(d1b, ub), r2(d2aa, ua, ua), r2(d2ab, ua, ub), r2(d2bb, ub, ub))
    got = spin_square_from_rdms(*_to_sbd(*rot), ab_overlap=ep_rot.ab_overlap)
    assert got == pytest.approx(ref, abs=1e-10)
    # loading without the overlap would be wrong here
    assert abs(spin_square_from_rdms(*_to_sbd(*rot)) - ref) > 1e-3


def test_uhf_fcidump_sidecar_loaded(tmp_path, monkeypatch):
    """compute_molecular_integrals_from_uhf_fcidump picks up <fcidump>.ab_overlap.npz."""
    from unittest import mock

    import qcsc_workflow_utility.chem as chem
    from qcsc_workflow_utility.uhf_fcidump import write_uhf_fcidump

    monkeypatch.setattr(chem, "get_run_logger", lambda: mock.MagicMock())
    monkeypatch.delenv("OO_INIT", raising=False)
    mol, mf = _uhf("H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2")
    h1, eri, n = _ints(mol, mf)
    path = tmp_path / "x.fcidump"
    write_uhf_fcidump(path, h1_a=h1[0], h1_b=h1[1], h2_aa=eri[0], h2_ab=eri[1], h2_bb=eri[2],
                      norb=n, nelec=mol.nelec, ecore=mol.energy_nuc())
    ep = chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
    assert ep.ab_overlap is None
    moa, mob = mf.mo_coeff
    s_ab = moa.T @ mf.get_ovlp() @ mob
    np.savez(f"{path}.ab_overlap.npz", ab_overlap=s_ab)
    ep = chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
    np.testing.assert_allclose(ep.ab_overlap, s_ab)


def test_cc_refresh_keeps_overlap():
    """refresh_cc_seed (used by OO_INIT and the OO_TO_LUCJ stagnation refresh) carries ab_overlap."""
    import qcsc_workflow_utility.chem as chem
    from qcsc_workflow_utility.chem import ElectronicProperties

    mol, mf = _uhf("H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2")
    h1, eri, n = _ints(mol, mf)
    moa, mob = mf.mo_coeff
    s_ab = moa.T @ mf.get_ovlp() @ mob
    na, nb = mol.nelec
    ep = ElectronicProperties(
        one_body_tensor=h1[0], one_body_tensor_b=h1[1], two_body_tensor=eri[0],
        two_body_tensor_ab=eri[1], two_body_tensor_bb=eri[2],
        t2=np.zeros((na, na, n - na, n - na)), t2_ab=np.zeros((na, nb, n - na, n - nb)),
        t2_bb=np.zeros((nb, nb, n - nb, n - nb)),
        initial_occupancy=(np.zeros(n), np.zeros(n)), nuclear_repulsion_energy=mol.energy_nuc(),
        num_orbitals=n, num_electrons=mol.nelec, open_shell=False, spin_sq=0.0,
        unrestricted=True, ab_overlap=s_ab,
    )
    new = chem.refresh_cc_seed(ep)
    assert new is not None
    np.testing.assert_allclose(new.ab_overlap, s_ab)
