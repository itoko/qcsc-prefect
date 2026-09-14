"""Regression tests for load_uhf_fcidump_reference.py -- the "load an already-converged BS-UHF
reference from its interleaved FCIDUMP without reconverging" script that closes the gap in
compute_molecular_integrals_from_fcidump (see the module docstring in
examples/fe4s4_hci_from_bsuhf_reference/load_uhf_fcidump_reference.py for the full bug writeup).

Every assertion here is checked against a REAL PySCF UHF/UCCSD calculation on a real molecule,
not just a synthetic round-trip: the whole point of this script is that its numbers must match
what PySCF itself would report for the same reference, to the precision PySCF's own solvers
converge to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "fe4s4_hci_from_bsuhf_reference"
sys.path.insert(0, str(EXAMPLES_DIR))

from pyscf import cc, gto, scf  # noqa: E402

import load_uhf_fcidump_reference as loader  # noqa: E402
from sbd.solver_job import _write_uhf_fcidump  # noqa: E402


def _converged_uhf_tensors(atom: str, basis: str, spin: int):
    """Run a real PySCF UHF (+ UCCSD) calculation and return everything needed to both write an
    interleaved FCIDUMP for it and independently know the ground-truth answer."""
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    mycc = cc.UCCSD(mf)
    mycc.kernel()

    na, nb = mf.nelec
    norb = mf.mo_coeff[0].shape[1]
    mo_a, mo_b = mf.mo_coeff
    hcore_ao = mf.get_hcore()
    h1_a = mo_a.T @ hcore_ao @ mo_a
    h1_b = mo_b.T @ hcore_ao @ mo_b
    eri_ao = mol.intor("int2e", aosym="s1")

    def ao2mo_full(mo1, mo2):
        return np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, mo1, mo1, mo2, mo2, optimize=True)

    h2_aa = ao2mo_full(mo_a, mo_a)
    h2_bb = ao2mo_full(mo_b, mo_b)
    h2_ab = ao2mo_full(mo_a, mo_b)
    ecore = mol.energy_nuc()

    return {
        "h1_a": h1_a, "h1_b": h1_b, "h2_aa": h2_aa, "h2_ab": h2_ab, "h2_bb": h2_bb,
        "norb": norb, "nelec": (na, nb), "ecore": ecore,
        "e_uhf": mf.e_tot, "spin_sq": mf.spin_square()[0], "e_uccsd": mycc.e_tot,
    }


CASES = {
    "closed_shell_n2": ("N 0 0 0; N 0 0 1.1", "sto-3g", 0),
    "open_shell_oh": ("O 0 0 0; H 0 0 0.97", "sto-3g", 1),
}


@pytest.fixture(params=CASES.keys())
def real_reference(request, tmp_path):
    atom, basis, spin = CASES[request.param]
    ref = _converged_uhf_tensors(atom, basis, spin)
    fcidump_path = tmp_path / f"{request.param}.uhf.fcidump"
    _write_uhf_fcidump(
        fcidump_path,
        h1_a=ref["h1_a"], h1_b=ref["h1_b"],
        h2_aa=ref["h2_aa"], h2_ab=ref["h2_ab"], h2_bb=ref["h2_bb"],
        norb=ref["norb"], nelec=ref["nelec"], ecore=ref["ecore"],
    )
    return fcidump_path, ref


def test_load_interleaved_uhf_fcidump_recovers_tensors_exactly(real_reference):
    """Parsing the file back must reproduce h1_a/h1_b/h2_aa/h2_ab/h2_bb/norb/nelec/ecore exactly
    -- this is a pure format parser, no SCF, so nothing should be lost or altered."""
    fcidump_path, ref = real_reference
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore = loader.load_interleaved_uhf_fcidump(
        str(fcidump_path)
    )
    assert norb == ref["norb"]
    assert nelec == ref["nelec"]
    assert ecore == pytest.approx(ref["ecore"])
    assert np.allclose(h1_a, ref["h1_a"])
    assert np.allclose(h1_b, ref["h1_b"])
    assert np.allclose(h2_aa, ref["h2_aa"])
    assert np.allclose(h2_ab, ref["h2_ab"])
    assert np.allclose(h2_bb, ref["h2_bb"])


def test_uhf_energy_from_integrals_matches_real_pyscf(real_reference):
    """The direct tensor-contraction UHF energy must match PySCF's own converged mf.e_tot to
    machine precision -- this is the check that catches "loading the FCIDUMP silently changed
    the energy" (the RHF-fallback failure mode this script exists to close)."""
    fcidump_path, ref = real_reference
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore = loader.load_interleaved_uhf_fcidump(
        str(fcidump_path)
    )
    e_uhf, sz = loader.uhf_energy_from_integrals(h1_a, h1_b, h2_aa, h2_ab, h2_bb, nelec, ecore)
    assert e_uhf == pytest.approx(ref["e_uhf"], abs=1e-10)
    na, nb = nelec
    assert sz == pytest.approx(0.5 * (na - nb))


def test_spin_square_matches_real_pyscf(real_reference):
    """<S^2> via the identity-overlap shortcut must match (or closely match, for genuine open-
    shell spin contamination that the orthonormal-basis assumption can't see) PySCF's own
    mf.spin_square()."""
    fcidump_path, ref = real_reference
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore = loader.load_interleaved_uhf_fcidump(
        str(fcidump_path)
    )
    spin_sq = loader._spin_square_via_pyscf(h1_a, h1_b, nelec)
    assert spin_sq == pytest.approx(ref["spin_sq"], abs=0.05)


def test_run_uccsd_on_reference_matches_real_pyscf(real_reference):
    """UCCSD run on top of the loaded (not reconverged) reference must match a direct PySCF
    cc.UCCSD(mf) run on the same converged mf, to UCCSD's own convergence tolerance -- this is
    the half of the bug report ("checking ... the UCCSD energy against my paper energy") that a
    UHF-energy-only check would miss."""
    fcidump_path, ref = real_reference
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore = loader.load_interleaved_uhf_fcidump(
        str(fcidump_path)
    )
    e_uccsd = loader.run_uccsd_on_reference(
        h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore, max_cycle=200
    )
    assert e_uccsd == pytest.approx(ref["e_uccsd"], abs=1e-6)


def test_cli_end_to_end_writes_summary_matching_real_pyscf(real_reference, tmp_path, capsys):
    """The actual --uccsd CLI path (main()), not just the library functions, end-to-end against
    a real molecule -- confirms argument wiring and the summary-file writer are also correct."""
    fcidump_path, ref = real_reference
    argv = sys.argv
    try:
        sys.argv = ["load_uhf_fcidump_reference.py", "--fcidump", str(fcidump_path), "--uccsd"]
        loader.main()
    finally:
        sys.argv = argv

    summary_path = Path(f"{fcidump_path}.loaded_summary.txt")
    assert summary_path.exists()
    text = summary_path.read_text()
    assert f"norb: {ref['norb']}" in text

    lines = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in text.splitlines()}
    assert float(lines["E_UHF_loaded"]) == pytest.approx(ref["e_uhf"], abs=1e-6)
    assert float(lines["E_UCCSD_loaded"]) == pytest.approx(ref["e_uccsd"], abs=1e-6)
