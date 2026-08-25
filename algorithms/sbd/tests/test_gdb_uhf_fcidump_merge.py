"""Regression tests for the BS-UHF -> interleaved-spin-orbital FCIDUMP merge used to feed a
`-D_UHF`-compiled `sbd::gdb` binary (`gdb_diag_uhf`) real BS-UHF integrals.

See examples/fe4s4_hci_from_bsuhf_reference/merge_bsuhf_to_uhf_fcidump.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "fe4s4_hci_from_bsuhf_reference"
sys.path.insert(0, str(EXAMPLES_DIR))


@pytest.fixture(autouse=True)
def _silence_prefect_logger(monkeypatch):
    from sbd import solver_job

    monkeypatch.setattr(solver_job, "get_run_logger", lambda: mock.MagicMock())


def _write_synthetic_bsuhf_files(tmp_path: Path, prefix: str):
    """Write a tiny synthetic norb=2 BS-UHF triple in the exact format
    prepare_bsuhf_fcidump.py produces: <prefix>.alpha.fcidump, .beta.fcidump, .mixed.npz."""
    from pyscf import ao2mo, tools

    norb = 2
    na, nb = 1, 1
    ecore = 1.5

    h1_a = np.array([[-1.0, 0.1], [0.1, -0.5]])
    h1_b = np.array([[-1.2, 0.0], [0.0, -0.4]])

    # Off-diagonal same-spin elements (ij != kl, both != 0) are the case that exercises the
    # (ij)<->(kl) swap emit_two_body relies on the reader to reconstruct for same-spin blocks --
    # a purely-diagonal h2_aa/h2_bb would never expose a bug in that reconstruction path.
    from pyscf import ao2mo as _ao2mo

    h2_aa = np.zeros((norb,) * 4)
    h2_aa[0, 0, 0, 0] = 0.7
    h2_aa[1, 0, 0, 0] = 0.2  # off-diagonal (10|00)
    h2_aa = _ao2mo.restore(1, _ao2mo.restore(8, h2_aa, norb), norb)
    h2_bb = np.zeros((norb,) * 4)
    h2_bb[1, 1, 1, 1] = 0.6
    h2_bb[1, 0, 1, 1] = 0.15  # off-diagonal (10|11)
    h2_bb = _ao2mo.restore(1, _ao2mo.restore(8, h2_bb, norb), norb)
    h2_ab = np.zeros((norb,) * 4)
    h2_ab[0, 0, 1, 1] = 0.3  # mixed (00|11)

    alpha_path = tmp_path / f"{prefix}.alpha.fcidump"
    beta_path = tmp_path / f"{prefix}.beta.fcidump"
    mixed_path = tmp_path / f"{prefix}.mixed.npz"

    eri_aa_packed = ao2mo.restore(8, h2_aa, norb)
    eri_bb_packed = ao2mo.restore(8, h2_bb, norb)
    tools.fcidump.from_integrals(str(alpha_path), h1_a, eri_aa_packed, norb, na, nuc=ecore, ms=0)
    tools.fcidump.from_integrals(str(beta_path), h1_b, eri_bb_packed, norb, nb, nuc=0.0, ms=0)
    # Real prepare_bsuhf_fcidump.py writes eri_ab via ao2mo.general(..., compact=False), which
    # PySCF returns flat as (norb**2, norb**2), not (norb,norb,norb,norb) -- mirror that shape
    # here so this test would have caught the real reshape bug found against live Fugaku data.
    np.savez(mixed_path, eri_ab=h2_ab.reshape(norb * norb, norb * norb), norb=norb)

    return h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, (na, nb), ecore


def test_merge_matches_direct_write_uhf_fcidump_call(tmp_path, monkeypatch):
    """The merge script's output must be byte-identical to calling _write_uhf_fcidump directly
    on the same tensors -- it should be a pure format bridge, nothing more."""
    from sbd.solver_job import _write_uhf_fcidump

    prefix = "toy"
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore = _write_synthetic_bsuhf_files(
        tmp_path, prefix
    )

    monkeypatch.chdir(tmp_path)
    import merge_bsuhf_to_uhf_fcidump as merge_mod

    out_merged = tmp_path / "merged.fcidump"
    merge_mod.merge(prefix, str(out_merged))

    out_direct = tmp_path / "direct.fcidump"
    _write_uhf_fcidump(
        out_direct, h1_a=h1_a, h1_b=h1_b, h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
        norb=norb, nelec=nelec, ecore=ecore,
    )

    assert out_merged.read_text() == out_direct.read_text()


def _reconstruct_from_merged_fcidump(path: Path, norb: int):
    """Standalone Python re-implementation of the C++ _UHF SetupIntegrals reader's indexing,
    used only to verify the merge script's output round-trips losslessly -- mirrors
    test_uhf.py::test_uhf_fcidump_spin_block_convention's reference-expansion discipline."""
    h1_a = np.zeros((norb, norb))
    h1_b = np.zeros((norb, norb))
    h2_aa = np.zeros((norb,) * 4)
    h2_ab = np.zeros((norb,) * 4)
    h2_bb = np.zeros((norb,) * 4)
    ecore = 0.0

    def spatial_and_spin(spinorb_1based: int) -> tuple[int, bool]:
        idx0 = spinorb_1based - 1
        return idx0 // 2, bool(idx0 % 2)

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("&") or "ORBSYM" in line or "ISYM" in line:
            continue
        parts = line.split()
        val = float(parts[0])
        i, j, k, l = (int(x) for x in parts[1:])
        if i == j == k == l == 0:
            ecore = val
            continue
        if k == 0 and l == 0:
            pi, beta_i = spatial_and_spin(i)
            pj, beta_j = spatial_and_spin(j)
            assert beta_i == beta_j
            target = h1_b if beta_i else h1_a
            target[pi, pj] = val
            target[pj, pi] = val
            continue
        pi, beta_i = spatial_and_spin(i)
        pj, beta_j = spatial_and_spin(j)
        pk, beta_k = spatial_and_spin(k)
        pl, beta_l = spatial_and_spin(l)
        assert beta_i == beta_j and beta_k == beta_l
        if not beta_i and not beta_k:
            target = h2_aa
            same_spin = True
        elif beta_i and beta_k:
            target = h2_bb
            same_spin = True
        elif not beta_i and beta_k:
            target = h2_ab
            same_spin = False
        else:
            # bb|aa -- transpose of aa|bb, reconstructs the same h2_ab via (kl|ij)=(ij|kl).
            target = h2_ab
            pi, pj, pk, pl = pk, pl, pi, pj
            same_spin = False
        perms = [(pi, pj, pk, pl), (pj, pi, pk, pl), (pi, pj, pl, pk), (pj, pi, pl, pk)]
        if same_spin:
            # emit_two_body only writes i>=j, k>=l, ij<=kl for same-spin blocks -- also fill
            # the (ij)<->(kl) swap the C++ reader reconstructs via (ij|kl)=(kl|ij).
            perms += [(pk, pl, pi, pj), (pl, pk, pi, pj), (pk, pl, pj, pi), (pl, pk, pj, pi)]
        for a, b, c, d in perms:
            target[a, b, c, d] = val

    return h1_a, h1_b, h2_aa, h2_ab, h2_bb, ecore


def test_merge_roundtrips_losslessly(tmp_path, monkeypatch):
    prefix = "toy"
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore = _write_synthetic_bsuhf_files(
        tmp_path, prefix
    )

    monkeypatch.chdir(tmp_path)
    import merge_bsuhf_to_uhf_fcidump as merge_mod

    out_merged = tmp_path / "merged.fcidump"
    merge_mod.merge(prefix, str(out_merged))

    r_h1_a, r_h1_b, r_h2_aa, r_h2_ab, r_h2_bb, r_ecore = _reconstruct_from_merged_fcidump(
        out_merged, norb
    )

    assert np.allclose(r_h1_a, h1_a)
    assert np.allclose(r_h1_b, h1_b)
    assert np.allclose(r_h2_aa, h2_aa)
    assert np.allclose(r_h2_ab, h2_ab)
    assert np.allclose(r_h2_bb, h2_bb)
    assert r_ecore == pytest.approx(ecore)


def test_merge_rejects_mismatched_norb(tmp_path, monkeypatch):
    from pyscf import ao2mo, tools

    prefix = "bad"
    norb = 2
    h1 = np.eye(norb)
    eri = ao2mo.restore(8, np.zeros((norb,) * 4), norb)
    tools.fcidump.from_integrals(str(tmp_path / f"{prefix}.alpha.fcidump"), h1, eri, norb, 1, nuc=0.0, ms=0)
    tools.fcidump.from_integrals(str(tmp_path / f"{prefix}.beta.fcidump"), h1, eri, norb, 1, nuc=0.0, ms=0)
    # mixed.npz claims a different norb -- must be rejected loudly, not silently misused.
    np.savez(tmp_path / f"{prefix}.mixed.npz", eri_ab=np.zeros((3, 3, 3, 3)), norb=3)

    monkeypatch.chdir(tmp_path)
    import merge_bsuhf_to_uhf_fcidump as merge_mod

    with pytest.raises(ValueError, match="norb"):
        merge_mod.merge(prefix, str(tmp_path / "out.fcidump"))
