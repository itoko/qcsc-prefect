"""Regression tests for the SQD sample-pool -> gdb `--detfiles` merge used to build a real
determinant subset for `sbd::gdb`'s heat-bath CI from production sample data.

See examples/fe4s4_hci_from_bsuhf_reference/pool_to_gdb_detfile.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "fe4s4_hci_from_bsuhf_reference"
sys.path.insert(0, str(EXAMPLES_DIR))

import pool_to_gdb_detfile as merge_mod  # noqa: E402


def _pack_bitstrings(bool_rows: list[list[int]]) -> np.ndarray:
    """Pack a list of MSB-first bit rows the same way Qiskit's BitArray does: real data is
    right-aligned in the byte array (left-padded with zeros up to a full byte boundary), so
    np.unpackbits(...)[..., -num_bits:] recovers the original bits. Verified directly against a
    real qiskit.primitives.containers.BitArray.from_samples(...) round-trip."""
    num_bits = len(bool_rows[0])
    pad = (-num_bits) % 8
    padded_rows = [[0] * pad + row for row in bool_rows]
    return np.packbits(np.array(padded_rows, dtype=np.uint8), axis=-1)


def test_load_pool_matches_bitstring_dedup_and_probs(tmp_path):
    """load_pool must reproduce np.unique's (bitstrings, counts/num_shots) exactly, matching
    qiskit_addon_sqd.counts.bit_array_to_arrays's own semantics."""
    norb = 2
    num_bits = 2 * norb  # 4
    # Three shots: two identical, one different.
    rows = [
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
    ]
    packed = _pack_bitstrings(rows)
    pool_path = tmp_path / "pool.npz"
    np.savez(pool_path, packed_bits=packed, num_bits=np.array([num_bits]),
             num_shots=np.array([len(rows)]))

    bitstrings, probs = merge_mod.load_pool(str(pool_path))
    assert bitstrings.shape == (2, num_bits)
    # np.unique sorts lexicographically: [0,1,0,0] before [1,0,1,1].
    assert bitstrings[0].tolist() == [0, 1, 0, 0]
    assert bitstrings[1].tolist() == [1, 0, 1, 1]
    assert probs[0] == pytest.approx(1 / 3)
    assert probs[1] == pytest.approx(2 / 3)


def test_build_merged_determinants_bit_interleaving():
    """Hand-worked reference: alpha=0b0011 (orbitals 0,1 occupied), beta=0b1100 (orbitals 2,3
    occupied), norb=4. Combined bit position 2k=alpha orbital k, 2k+1=beta orbital k.

    alpha bits (orbital index -> occupied): 0->1, 1->1, 2->0, 3->0
    beta  bits (orbital index -> occupied): 0->0, 1->0, 2->1, 3->1
    combined bit 0 (alpha 0) = 1, bit 1 (beta 0) = 0, bit 2 (alpha 1) = 1, bit 3 (beta 1) = 0,
    bit 4 (alpha 2) = 0, bit 5 (beta 2) = 1, bit 6 (alpha 3) = 0, bit 7 (beta 3) = 1
    -> combined = 0b10100101 = 0xA5 = 165
    """
    norb = 4
    # bitstrings columns: [:norb]=beta, [norb:]=alpha.
    beta_bits = [0, 0, 1, 1]  # orbitals 2,3 occupied
    alpha_bits = [1, 1, 0, 0]  # orbitals 0,1 occupied
    row = np.array([beta_bits + alpha_bits], dtype=bool)

    combined = merge_mod.build_merged_determinants(row, norb)
    assert combined == [0b10100101]
    assert combined[0] == 165

    # Round-trip: the emitted string must recover the exact original alpha/beta bit arrays when
    # re-split by character index 2*norb-1-p (MSB-left string, p=0 is the rightmost character).
    width = 2 * norb
    s = format(combined[0], f"0{width}b")
    for k in range(norb):
        alpha_char = s[width - 1 - 2 * k]
        beta_char = s[width - 1 - (2 * k + 1)]
        assert int(alpha_char) == alpha_bits[k]
        assert int(beta_char) == beta_bits[k]


def test_build_merged_determinants_overflows_64_bits():
    """Fe4S4-scale check: norb=36 -> 72-bit combined values, which overflow np.int64/np.longlong.
    Verify the implementation genuinely handles bit 63+ correctly (a naive np.int64 accumulator
    would silently wrap/truncate here)."""
    norb = 36
    # All orbitals occupied for both spins -> combined should be (1 << 72) - 1.
    beta_bits = [1] * norb
    alpha_bits = [1] * norb
    row = np.array([beta_bits + alpha_bits], dtype=bool)

    combined = merge_mod.build_merged_determinants(row, norb)
    assert combined == [(1 << (2 * norb)) - 1]
    assert combined[0] > (1 << 63)  # genuinely exceeds a 64-bit signed integer's range

    # High-orbital-only occupation (orbital 35, both spins) must set high bits, not wrap to 0.
    beta_bits = [0] * norb
    beta_bits[35] = 1
    alpha_bits = [0] * norb
    alpha_bits[35] = 1
    row = np.array([beta_bits + alpha_bits], dtype=bool)
    combined = merge_mod.build_merged_determinants(row, norb)
    expected = (1 << (2 * 35)) | (1 << (2 * 35 + 1))
    assert combined == [expected]
    assert combined[0] > (1 << 63)


def test_write_detfile_dedup_and_format(tmp_path):
    norb = 2
    combined = [0b1011, 0b1011, 0b0100]  # one duplicate
    out_path = tmp_path / "det.txt"
    n_unique = merge_mod.write_detfile(combined, norb, str(out_path))
    assert n_unique == 2
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    assert all(len(line) == 2 * norb for line in lines)
    values = {int(line, 2) for line in lines}
    assert values == {0b1011, 0b0100}


def test_electron_count_sanity_check_flags_mismatch(capsys):
    norb = 4
    # value with alpha popcount=2, beta popcount=2 (matches na=2, nb=2)
    good = 0b10100101  # from the earlier hand-worked reference: alpha={0,1}, beta={2,3}
    # a value with a different electron count (alpha popcount=1)
    bad = 0b00000001

    merge_mod.electron_count_sanity_check([good], norb, na=2, nb=2)
    captured = capsys.readouterr()
    assert "all 1 shots match" in captured.err

    merge_mod.electron_count_sanity_check([bad], norb, na=2, nb=2)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_post_select_by_hamming_weight_filters_wrong_particle_number():
    """Regression: the real saved pool is raw, UNFILTERED hardware data -- most shots do not
    have the correct particle number (confirmed against fe4s4_uhf_5M_zhendongli.npz: the top-5
    highest-probability raw shots have alpha/beta popcounts like 22/8, 19/9, nowhere near the
    correct 27/27). Skipping this step and taking the raw top-N by probability silently produces
    determinants with the wrong electron count. norb=4, target nelec=(2,2)."""
    norb = 4
    # columns [:norb]=beta, [norb:]=alpha.
    rows = np.array([
        [0, 0, 1, 1] + [1, 1, 0, 0],  # beta popcount=2, alpha popcount=2 -- KEEP
        [0, 0, 0, 1] + [1, 1, 0, 0],  # beta popcount=1 -- DROP (wrong nelec_b)
        [0, 0, 1, 1] + [1, 0, 0, 0],  # alpha popcount=1 -- DROP (wrong nelec_a)
        [1, 1, 0, 0] + [0, 0, 1, 1],  # beta popcount=2, alpha popcount=2 -- KEEP
    ], dtype=bool)
    probs = np.array([0.4, 0.3, 0.2, 0.1])

    kept_bits, kept_probs = merge_mod.post_select_by_hamming_weight(
        rows, probs, norb, nelec_a=2, nelec_b=2
    )
    assert kept_bits.shape[0] == 2
    assert np.array_equal(kept_bits[0], rows[0])
    assert np.array_equal(kept_bits[1], rows[3])
    assert kept_probs.tolist() == pytest.approx([0.4, 0.1])


def test_post_select_by_hamming_weight_empty_result():
    norb = 2
    rows = np.array([[1, 1, 0, 0]], dtype=bool)  # beta popcount=2, alpha popcount=0
    probs = np.array([1.0])
    kept_bits, kept_probs = merge_mod.post_select_by_hamming_weight(
        rows, probs, norb, nelec_a=1, nelec_b=1
    )
    assert kept_bits.shape == (0, 2 * norb)
    assert kept_probs.shape == (0,)


def test_select_shots_picks_highest_probability(tmp_path):
    norb = 2
    num_bits = 4
    bitstrings = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]], dtype=bool)
    probs = np.array([0.1, 0.7, 0.2])

    selected = merge_mod.select_shots(bitstrings, probs, norb, n_shots=2)
    assert selected.shape == (2, num_bits)
    # Top 2 by probability: index 1 (0.7), index 2 (0.2).
    assert selected[0].tolist() == [1, 1, 1, 1]
    assert selected[1].tolist() == [0, 1, 0, 1]
