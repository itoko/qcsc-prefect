"""RDM + carryover I/O validation for the SBD solver (RHF and UHF).

Two independent contracts are locked here, both against the REAL reader code in
``sbd.solver_job`` (not re-implementations):

1. Carryover round-trip (``carryover.bin`` / ``carryover_b.bin``).
   The multi-GPU solver writes surviving determinants with a hand-rolled
   big-endian bit-packing in ``main.cc`` (``write_co``); Python decodes them with
   ``_read_carryover_bin``. If the two disagree on bit order, byte order, word
   layout, or padding, the subspace carried between recovery steps is silently
   corrupted. These tests mirror the C++ packing, write a file, read it back with
   the production decoder, and assert exact determinant recovery + preserved
   electron count (popcount), for RHF (alpha only) and UHF (independent alpha +
   beta), including the norb-not-multiple-of-8 cases (Fe2S2 norb=20, Fe4S4
   norb=36 -> 4 padding bits that must not fabricate occupation).

2. RDM file validation (``rdm1_a/b.txt``, ``rdm2_aa/ab/bb.txt``), read with the
   production ``_read_rdm_file``. Asserts the physical invariants the
   off-diagonal-accumulation bug violated: Tr(gamma^1) == N_e (RHF: Tr(a);
   UHF: Tr(a)+Tr(b)), symmetry, off-diagonal coherence present, correct rank-4
   shape for the 2-RDM blocks, and the energy-from-RDM identity on a tiny case.

The device-side accumulation itself is pinned by the C++ test
``tests/functionality/test_gpu_rdm_correlation.cc``; this file covers the
Python-side format contract and RDM-file invariants for both spin methods.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _silence_prefect_logger(monkeypatch):
    """IO helpers call Prefect's get_run_logger; stub it for unit tests."""
    from sbd import solver_job
    monkeypatch.setattr(solver_job, "get_run_logger", lambda: mock.MagicMock())


# =============================================================================
# Carryover packing: faithful mirror of main.cc::write_co
# =============================================================================
def _pack_carryover(dets_occ, norb):
    """Bytes identical to what the solver writes to carryover.bin.

    C++ reference (main.cc write_co):
      bytes_per_config = (L + 7) / 8            ; L == norb
      for j in 0..L-1:
        rev_idx = L - 1 - j                     # sbd::makestring reversal
        bit     = occupied(rev_idx)
        pb = 7 - (j % 8); bb = j // 8           # big-endian bit within byte
        bytes[bb] |= bit << pb
    So orbital `o` maps to output bit j = L-1-o: the decoder column p (after
    np.unpackbits big-endian, sliced [:, :norb]) corresponds to orbital L-1-p.
    """
    bytes_per_config = (norb + 7) // 8
    out = bytearray()
    for occ in dets_occ:
        occset = set(occ)
        buf = bytearray(bytes_per_config)
        for j in range(norb):
            if (norb - 1 - j) in occset:
                buf[j // 8] |= (1 << (7 - (j % 8)))
        out += buf
    return bytes(out)


def _occ_from_row(row):
    """Occupied orbital indices (0-based) from a decoded bool row
    (column p == orbital norb-1-p, per the packer above)."""
    norb = len(row)
    return {norb - 1 - p for p in range(norb) if row[p]}


CARRYOVER_CASES = [
    # (tag, norb, list-of-occupied-orbital-lists)
    ("small_rhf_alpha", 6, [[0, 1, 2], [0, 2, 4], [1, 3, 5]]),
    ("fe2s2_norb20", 20, [list(range(15)),
                          [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9]]),
    ("fe4s4_norb36_alpha", 36, [list(range(27)), list(range(9, 36))]),
    # UHF beta (carryover_b.bin) uses the same decoder with an independent set:
    ("fe4s4_norb36_beta", 36, [list(range(0, 27)), list(range(2, 29))]),
]


@pytest.mark.parametrize("tag,norb,dets_occ", CARRYOVER_CASES,
                         ids=[c[0] for c in CARRYOVER_CASES])
def test_carryover_roundtrip(tag, norb, dets_occ):
    from sbd.solver_job import _read_carryover_bin

    raw = _pack_carryover(dets_occ, norb)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "carryover.bin"
        path.write_bytes(raw)
        decoded = _read_carryover_bin(path, norb)

    assert decoded.shape == (len(dets_occ), norb)
    for i, occ in enumerate(dets_occ):
        assert _occ_from_row(decoded[i]) == set(occ), f"det {i} mismatch"
        assert int(decoded[i].sum()) == len(occ), f"popcount changed for det {i}"


def test_carryover_empty_file():
    """A zero-length carryover.bin decodes to an empty (0, norb) array."""
    from sbd.solver_job import _read_carryover_bin
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "carryover.bin"
        path.write_bytes(b"")
        out = _read_carryover_bin(path, 36)
    assert out.shape == (0, 36)


def test_carryover_misaligned_raises():
    """A file whose size is not a multiple of bytes-per-config is rejected,
    not silently truncated (guards against a partial/corrupt write)."""
    from sbd.solver_job import _read_carryover_bin
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "carryover.bin"
        path.write_bytes(b"\x01\x02\x03")  # 3 bytes; norb=36 -> 5 bytes/config
        with pytest.raises(ValueError):
            _read_carryover_bin(path, 36)


# =============================================================================
# RDM file validation
# =============================================================================
def _make_1rdm(norb, occ_diag, coherence, seed_orbs):
    g = np.zeros((norb, norb))
    for p, v in enumerate(occ_diag):
        g[p, p] = v
    a, b = seed_orbs
    g[a, b] = g[b, a] = coherence
    return g


def test_rdm1_rhf():
    from sbd.solver_job import _read_rdm_file
    norb, N_e = 4, 6.0
    g = _make_1rdm(norb, [1.9, 1.7, 1.6, 0.8], coherence=0.2, seed_orbs=(1, 2))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rdm1_a.txt"
        np.savetxt(p, g.ravel())
        gr = _read_rdm_file(p, norb, rank=2)
    assert gr is not None and gr.shape == (norb, norb)
    assert abs(np.trace(gr) - N_e) < 1e-9            # Tr == N_e (the bug: ~0)
    assert np.allclose(gr, gr.T)                      # symmetric
    offdiag = np.abs(gr - np.diag(np.diag(gr))).sum()
    assert offdiag > 0                                # coherence present


def test_rdm1_uhf_spin_resolved():
    from sbd.solver_job import _read_rdm_file
    norb, N_a, N_b = 4, 4.0, 2.0
    ga = _make_1rdm(norb, [1.5, 1.2, 0.9, 0.4], 0.18, (0, 2))
    gb = _make_1rdm(norb, [0.9, 0.6, 0.4, 0.1], 0.10, (1, 3))
    Gab = np.zeros((norb,) * 4); Gab[0, 0, 0, 0] = 0.5
    with tempfile.TemporaryDirectory() as d:
        pa = Path(d) / "rdm1_a.txt"; np.savetxt(pa, ga.ravel())
        pb = Path(d) / "rdm1_b.txt"; np.savetxt(pb, gb.ravel())
        pab = Path(d) / "rdm2_ab.txt"; np.savetxt(pab, Gab.ravel())
        gar = _read_rdm_file(pa, norb, rank=2)
        gbr = _read_rdm_file(pb, norb, rank=2)
        Gab_r = _read_rdm_file(pab, norb, rank=4)
    assert abs(np.trace(gar) - N_a) < 1e-9
    assert abs(np.trace(gbr) - N_b) < 1e-9
    assert abs((np.trace(gar) + np.trace(gbr)) - (N_a + N_b)) < 1e-9
    assert np.allclose(gar, gar.T) and np.allclose(gbr, gbr.T)
    assert Gab_r is not None and Gab_r.shape == (norb,) * 4


def test_rdm_missing_file_returns_none():
    """Absent RDM file -> None (a solver build without --rdm degrades cleanly)."""
    from sbd.solver_job import _read_rdm_file
    with tempfile.TemporaryDirectory() as d:
        assert _read_rdm_file(Path(d) / "nope.txt", 4, rank=2) is None


def test_rdm_wrong_size_raises():
    from sbd.solver_job import _read_rdm_file
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rdm1_a.txt"
        np.savetxt(p, np.zeros(9))          # 9 != 4^2
        with pytest.raises(ValueError):
            _read_rdm_file(p, 4, rank=2)


def test_energy_from_rdm_identity_rhf():
    """E = sum_pq h.gamma + 1/2 sum (pq|rs).Gamma reproduces the reference on a
    tiny 2-orbital case read back through the production RDM reader."""
    from sbd.solver_job import _read_rdm_file
    norb = 2
    h = np.array([[-1.0, 0.2], [0.2, -0.5]])
    gamma = np.array([[1.6, 0.1], [0.1, 0.4]])          # Tr = 2
    eri = np.zeros((norb,) * 4); eri[0, 0, 0, 0] = 0.7
    Gamma = np.zeros((norb,) * 4); Gamma[0, 0, 0, 0] = 1.3
    E_ref = np.einsum("pq,pq->", h, gamma) + 0.5 * np.einsum("pqrs,pqrs->", eri, Gamma)
    with tempfile.TemporaryDirectory() as d:
        pg = Path(d) / "rdm1_a.txt"; np.savetxt(pg, gamma.ravel())
        pG = Path(d) / "rdm2_aa.txt"; np.savetxt(pG, Gamma.ravel())
        gr = _read_rdm_file(pg, norb, rank=2)
        Gr = _read_rdm_file(pG, norb, rank=4)
    E = np.einsum("pq,pq->", h, gr) + 0.5 * np.einsum("pqrs,pqrs->", eri, Gr)
    assert abs(E - E_ref) < 1e-12


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
