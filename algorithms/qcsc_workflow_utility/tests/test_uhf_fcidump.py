"""Spin-orbital FCIDUMP I/O (qcsc_workflow_utility.uhf_fcidump) and the UHF input task
(chem.compute_molecular_integrals_from_uhf_fcidump).

Ported from sbd/tests/test_load_uhf_fcidump_reference.py (the examples/ loader it tested is now
the package reader). Every number is checked against a REAL PySCF UHF/UCCSD calculation.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from pyscf import ao2mo, cc, gto, scf

import qcsc_workflow_utility.chem as chem
from qcsc_workflow_utility.uhf_fcidump import (
    aufbau_reference,
    check_aufbau_order,
    read_uhf_fcidump,
    write_uhf_fcidump,
)

CASES = {
    "closed_shell_n2": ("N 0 0 0; N 0 0 1.1", "sto-3g", 0),
    "open_shell_oh": ("O 0 0 0; H 0 0 0.97", "sto-3g", 1),
    # broken-symmetry singlet (<S^2>~1.08); at larger R UCCSD has multiple solutions
    "stretched_h4_bs": ("H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2", "sto-3g", 0),
}


@pytest.fixture(autouse=True)
def _silence_prefect_logger(monkeypatch):
    monkeypatch.setattr(chem, "get_run_logger", lambda: mock.MagicMock())
    monkeypatch.delenv("OO_INIT", raising=False)


def _converged_uhf(atom: str, basis: str, spin: int):
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    if spin == 0:
        # break spin symmetry (stretched H4) so h_alpha != h_beta is exercised
        dm = mf.get_init_guess()
        dm[0][0, 0] += 0.3
        dm[1][0, 0] -= 0.3
        mf.kernel(dm0=dm)
    else:
        mf.kernel()
    mo_a, mo_b = mf.mo_coeff
    norb = mo_a.shape[1]
    hc = mf.get_hcore()
    g = lambda a, b: ao2mo.general(mol, (a, a, b, b), compact=False).reshape((norb,) * 4)
    tensors = dict(
        h1_a=mo_a.T @ hc @ mo_a, h1_b=mo_b.T @ hc @ mo_b,
        h2_aa=g(mo_a, mo_a), h2_ab=g(mo_a, mo_b), h2_bb=g(mo_b, mo_b),
        norb=norb, nelec=mf.nelec, ecore=mol.energy_nuc(),
    )
    return mf, tensors


@pytest.fixture(params=CASES.keys())
def reference(request, tmp_path):
    mf, t = _converged_uhf(*CASES[request.param])
    path = tmp_path / f"{request.param}.uhf.fcidump"
    write_uhf_fcidump(path, **t)
    return path, mf, t


def test_roundtrip_recovers_tensors(reference):
    path, _, t = reference
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore = read_uhf_fcidump(path)
    assert norb == t["norb"] and nelec == tuple(t["nelec"])
    assert ecore == pytest.approx(t["ecore"], abs=1e-14)
    for got, key in ((h1_a, "h1_a"), (h1_b, "h1_b"), (h2_aa, "h2_aa"), (h2_ab, "h2_ab"), (h2_bb, "h2_bb")):
        np.testing.assert_allclose(got, t[key], rtol=0, atol=1e-14)


def test_aufbau_reference_energy_matches_pyscf(reference):
    path, mf, _ = reference
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore = read_uhf_fcidump(path)
    e_ref, fock_a, fock_b = aufbau_reference(h1_a, h1_b, h2_aa, h2_ab, h2_bb, nelec, ecore)
    assert e_ref == pytest.approx(mf.e_tot, abs=1e-10)
    np.testing.assert_allclose(np.diag(fock_a), mf.mo_energy[0], atol=1e-8)
    np.testing.assert_allclose(np.diag(fock_b), mf.mo_energy[1], atol=1e-8)
    assert check_aufbau_order(fock_a, fock_b, nelec) == []


def test_task_matches_real_uccsd(reference):
    """The integral-only task reproduces PySCF UCCSD on the real converged UHF: energy, t2, and
    the diagonal of the correlated 1-RDM (the lambda equations must see the right ab/bb blocks)."""
    path, mf, _ = reference
    ref = cc.UCCSD(mf)
    ref.conv_tol = 1e-10
    ref.kernel()
    dm_a, dm_b = ref.make_rdm1()

    ep = chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
    assert ep.unrestricted is True
    assert ep.num_electrons == tuple(mf.nelec)
    assert ep.nuclear_repulsion_energy == pytest.approx(mf.mol.energy_nuc())
    np.testing.assert_allclose(ep.t2, ref.t2[0], atol=1e-6)
    np.testing.assert_allclose(ep.t2_ab, ref.t2[1], atol=1e-6)
    np.testing.assert_allclose(ep.t2_bb, ref.t2[2], atol=1e-6)
    np.testing.assert_allclose(ep.initial_occupancy[0], np.diag(dm_a), atol=1e-6)
    np.testing.assert_allclose(ep.initial_occupancy[1], np.diag(dm_b), atol=1e-6)


def test_task_logs_reference_line(reference, capsys):
    path, mf, _ = reference
    chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("[uhf-input]"))
    e_ref = float(line.split("E_ref=")[1].split(",")[0])
    assert e_ref == pytest.approx(mf.e_tot, abs=1e-9)
    assert "E_UCCSD=" in line


def test_reader_rejects_spatial_fcidump(tmp_path):
    from pyscf import tools

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    path = tmp_path / "spatial.txt"
    tools.fcidump.from_scf(scf.RHF(mol).run(), str(path))
    with pytest.raises(ValueError, match="rhf2uhf"):
        read_uhf_fcidump(path)


def test_non_aufbau_order_warns(tmp_path):
    """Moving a high virtual into the occupied block must trigger the energy-order warning.

    (A HOMO/LUMO swap is not enough: the Fock matrix of the swapped determinant can still be
    ordered. Swapping the HOMO with the highest orbital makes the occupied block contain an
    orbital far above the virtuals.)
    """
    mf, t = _converged_uhf(*CASES["closed_shell_n2"])
    na, nb = t["nelec"]
    perm = np.arange(t["norb"])
    perm[[na - 1, t["norb"] - 1]] = perm[[t["norb"] - 1, na - 1]]
    sw = dict(t)
    sw["h1_a"] = t["h1_a"][np.ix_(perm, perm)]
    sw["h1_b"] = t["h1_b"][np.ix_(perm, perm)]
    for k in ("h2_aa", "h2_ab", "h2_bb"):
        sw[k] = t[k][np.ix_(perm, perm, perm, perm)]
    path = tmp_path / "swapped.txt"
    write_uhf_fcidump(path, **sw)
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, _, nelec, ecore = read_uhf_fcidump(path)
    _, fock_a, fock_b = aufbau_reference(h1_a, h1_b, h2_aa, h2_ab, h2_bb, nelec, ecore)
    assert check_aufbau_order(fock_a, fock_b, nelec)

    # The warning is raised before UCCSD; UCCSD on this deliberately bad reference may not
    # converge (singular DIIS), so stop right after the check.
    class _Stop(Exception):
        pass

    def _stop(*a, **k):
        raise _Stop

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chem.cc, "UCCSD", _stop)
        with pytest.warns(RuntimeWarning, match="energy ordered"), pytest.raises(_Stop):
            chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))


# ── SEED_CC_FILE: pin the UCCSD seed across runs ─────────────────────────────


def test_seed_cc_file_roundtrip(reference, tmp_path, monkeypatch):
    """First run writes the seed; a later run loads it without running UCCSD and gets the
    identical t2 / occupancies / E_UCCSD."""
    path, _, _ = reference
    seed = tmp_path / "seed" / "cc.npz"
    monkeypatch.setenv("SEED_CC_FILE", str(seed))
    ep1 = chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
    assert seed.exists()
    with mock.patch.object(chem.cc, "UCCSD", side_effect=AssertionError("UCCSD must not run")):
        ep2 = chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
    for a, b in [(ep1.t2, ep2.t2), (ep1.t2_ab, ep2.t2_ab), (ep1.t2_bb, ep2.t2_bb),
                 (ep1.initial_occupancy[0], ep2.initial_occupancy[0]),
                 (ep1.initial_occupancy[1], ep2.initial_occupancy[1])]:
        np.testing.assert_array_equal(a, b)


def test_seed_cc_file_existing_wins(reference, tmp_path, monkeypatch):
    """If the file appears while this run is in UCCSD (another array element wrote it first),
    this run uses the file's seed, not its own."""
    path, _, _ = reference
    seed = tmp_path / "cc.npz"
    monkeypatch.setenv("SEED_CC_FILE", str(seed))
    ep_ref = chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
    # tamper with the stored seed to make it distinguishable, keep its fingerprint
    with np.load(seed) as z:
        d = {k: z[k] for k in z.files}
    d["t2_aa"] = d["t2_aa"] + 1.0
    np.savez(seed, **d)
    real_exists = chem.os.path.exists
    calls = {"n": 0}

    def exists_late(p):  # pretend the file is absent at the start of this run only
        if str(p) == str(seed) and calls["n"] == 0:
            calls["n"] += 1
            return False
        return real_exists(p)

    monkeypatch.setattr(chem.os.path, "exists", exists_late)
    ep = chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
    np.testing.assert_allclose(ep.t2, ep_ref.t2 + 1.0)


def test_seed_cc_file_fingerprint_mismatch(reference, tmp_path, monkeypatch):
    path, _, tensors = reference
    seed = tmp_path / "cc.npz"
    monkeypatch.setenv("SEED_CC_FILE", str(seed))
    chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
    other = tmp_path / "other.txt"
    t = dict(tensors)
    t["h1_a"] = t["h1_a"] + 1e-6 * np.eye(t["norb"])
    write_uhf_fcidump(other, **t)
    with pytest.raises(ValueError, match="different integrals"):
        chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(other))


def test_seed_cc_file_rejects_oo_init(reference, tmp_path, monkeypatch):
    path, _, _ = reference
    monkeypatch.setenv("SEED_CC_FILE", str(tmp_path / "cc.npz"))
    monkeypatch.setenv("OO_INIT", "1")
    with pytest.raises(ValueError, match="OO_INIT"):
        chem.compute_molecular_integrals_from_uhf_fcidump.fn(str(path))
