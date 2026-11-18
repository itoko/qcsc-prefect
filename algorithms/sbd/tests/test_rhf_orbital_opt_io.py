"""RHF orbital-optimization plumbing in the sbd workflow.

1. ``solver_job._read_files`` for the RHF solver keeps ONE spin-summed 2-RDM:
   - the RHF build's ``rdm2.txt`` (aa + ab + ba + bb, prqs storage) is read as is;
   - without it (older binaries, the upstream app of the MPI build) the per-spin
     ``rdm2_aa/ab/bb.txt`` are summed, which must equal PySCF's spin-summed 2-RDM;
   - the UHF solver still reads the per-spin blocks and no spin-summed one.
2. ``main._check_oo_method``: RHF solver + oo_method="uhf" raises; UHF solver + oo_method="rhf"
   warns; matching combinations pass silently.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("pyscf")

from pyscf import ao2mo, fci, gto, scf  # noqa: E402

from sbd import solver_job  # noqa: E402


@pytest.fixture(autouse=True)
def _silence_prefect_logger(monkeypatch):
    """IO helpers call Prefect's get_run_logger; stub it for unit tests."""
    monkeypatch.setattr(solver_job, "get_run_logger", lambda: mock.MagicMock())


def _prqs(a):
    return np.asarray(a).transpose(0, 2, 1, 3)


@pytest.fixture(scope="module")
def fci_rdms():
    """Per-spin and spin-summed FCI RDMs of H2O+ (5, 4)/sto-3g; 2-RDMs in prqs storage."""
    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    norb = mo.shape[1]
    h1 = mo.T @ mf.get_hcore() @ mo
    h2 = ao2mo.full(mol, mo, compact=False).reshape((norb,) * 4)
    nelec = (5, 4)
    solver = fci.direct_spin1.FCI()
    _, c = solver.kernel(h1, h2, norb, nelec)
    (d1a, d1b), (d2aa, d2ab, d2bb) = solver.make_rdm12s(c, norb, nelec)
    _, d2 = solver.make_rdm12(c, norb, nelec)
    per_spin = dict(
        rdm1_a=d1a, rdm1_b=d1b, rdm2_aa=_prqs(d2aa), rdm2_ab=_prqs(d2ab), rdm2_bb=_prqs(d2bb),
    )
    return norb, per_spin, _prqs(d2)


def _write_solver_outputs(work_dir: Path, norb: int, rdms: dict[str, np.ndarray]) -> None:
    """Minimal solver output directory; RDM files as main.cc writes them (one value per line)."""
    np.savetxt(work_dir / "occ_a.txt", np.zeros(norb))
    np.savetxt(work_dir / "occ_b.txt", np.zeros(norb))
    (work_dir / "carryover.bin").write_bytes(b"")
    (work_dir / "davidson_energy.txt").write_text("-1.0\n")
    for name, arr in rdms.items():
        np.savetxt(work_dir / f"{name}.txt", np.asarray(arr).ravel(), fmt="%.16e")


class TestReadFilesRdm:
    def test_rhf_reads_spin_summed_rdm2_txt(self, tmp_path, fci_rdms):
        norb, per_spin, g2 = fci_rdms
        _write_solver_outputs(
            tmp_path, norb, dict(rdm1_a=per_spin["rdm1_a"], rdm1_b=per_spin["rdm1_b"], rdm2=g2),
        )
        r = solver_job._read_files(work_dir=tmp_path, norb=norb, method="rhf", do_rdm=1)
        np.testing.assert_allclose(r.rdm2_spin_summed, g2, atol=1e-14)
        np.testing.assert_allclose(r.rdm1 + r.rdm1_b, per_spin["rdm1_a"] + per_spin["rdm1_b"])
        assert r.rdm2 is None and r.rdm2_ab is None and r.rdm2_bb is None

    def test_rhf_sums_legacy_per_spin_files(self, tmp_path, fci_rdms):
        norb, per_spin, g2 = fci_rdms
        _write_solver_outputs(tmp_path, norb, per_spin)
        r = solver_job._read_files(work_dir=tmp_path, norb=norb, method="rhf", do_rdm=1)
        np.testing.assert_allclose(r.rdm2_spin_summed, g2, atol=1e-12)
        assert r.rdm2 is None

    def test_rhf_without_rdm_files(self, tmp_path, fci_rdms):
        norb, _, _ = fci_rdms
        _write_solver_outputs(tmp_path, norb, {})
        r = solver_job._read_files(work_dir=tmp_path, norb=norb, method="rhf", do_rdm=1)
        assert r.rdm1 is None and r.rdm2_spin_summed is None

    def test_uhf_keeps_per_spin_blocks(self, tmp_path, fci_rdms):
        norb, per_spin, _ = fci_rdms
        _write_solver_outputs(tmp_path, norb, per_spin)
        r = solver_job._read_files(work_dir=tmp_path, norb=norb, method="uhf", do_rdm=1)
        np.testing.assert_allclose(r.rdm2, per_spin["rdm2_aa"], atol=1e-14)
        np.testing.assert_allclose(r.rdm2_ab, per_spin["rdm2_ab"], atol=1e-14)
        np.testing.assert_allclose(r.rdm2_bb, per_spin["rdm2_bb"], atol=1e-14)
        assert r.rdm2_spin_summed is None


class TestCheckOoMethod:
    @pytest.fixture(scope="class")
    def check(self):
        from sbd.main import _check_oo_method

        return _check_oo_method

    def test_rhf_solver_with_uhf_oo_raises(self, check):
        with pytest.raises(ValueError, match="oo_method='uhf'"):
            check("uhf", False, mock.MagicMock())

    def test_uhf_solver_with_rhf_oo_warns(self, check):
        logger = mock.MagicMock()
        check("rhf", True, logger)
        logger.warning.assert_called_once()
        assert "ONE rotation" in logger.warning.call_args.args[0]

    @pytest.mark.parametrize("oo_method, unrestricted", [("rhf", False), ("uhf", True)])
    def test_matching_combinations_are_silent(self, check, oo_method, unrestricted):
        logger = mock.MagicMock()
        check(oo_method, unrestricted, logger)
        logger.warning.assert_not_called()
