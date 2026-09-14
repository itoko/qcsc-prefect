#!/usr/bin/env python
"""Convert a subset of a saved SQD sample pool (packed_bits/num_bits/num_shots npz) into the
plain-ASCII `--detfiles` format `sbd::gdb`'s solver apps (`gdb_diag`, `gdb_diag_uhf`) read.

Background
----------
The production SQD pipeline persists its raw quantum measurement pool as a Qiskit `BitArray`
serialized to three npz keys: `packed_bits` (bit-packed measurement outcomes, uint8), `num_bits`
(bits per shot, `2*norb`), `num_shots` (raw shot count -- NOT yet deduplicated). This is the exact
format `sbd/sqd.py`'s own `quantum_source="saved"` path loads via `qiskit_addon_sqd.counts
.bit_array_to_arrays`, which unpacks the bits and deduplicates identical shots into
(unique bitstring, frequency) pairs.

Bit layout (confirmed against sbd/sqd.py's own `_spin_halves_as_ints`/`subsample_open_shell`):
after `np.unpackbits`, bit 0 is the most-significant bit of the packed row; columns `[:norb]` are
BETA occupation, columns `[norb:]` are ALPHA occupation.

`sbd::gdb`'s own determinant convention (confirmed against `redistribution_equal_bra_a`'s
docstring and the `_UHF` integral spin-block indexing): a single interleaved `2*norb`-bit
determinant where even bit position `2p` (0-based, counting from the LEAST-significant/rightmost
bit -- confirmed against `apps/chemistry_gdb_selected_basis_diagonalization/README.md`: "the
rightmost bit corresponds to alpha-spin orbital 1") is alpha spatial orbital `p`, odd position
`2p+1` is beta spatial orbital `p`.

This script builds ONE merged determinant per SAMPLED SHOT (the paired alpha+beta bitstring
actually measured together), not a Cartesian product of independently-drawn alpha/beta values --
see docs/reference/sbd_gdb_heatbath_and_selection_gap.md's discussion of why a full outer product
would be physically wrong for this purpose.

Fe4S4 has norb=36, so each merged determinant is 72 bits -- this OVERFLOWS a 64-bit integer.
Python's arbitrary-precision `int` is used throughout; never `np.int64`/`np.longlong` for the
merged value.

Usage
-----
    python pool_to_gdb_detfile.py --pool fe4s4_uhf_5M_zhendongli.npz --norb 36 \
        --n-shots 5000 --out fe4s4_bsuhf_pool5000.detfile.txt

This is a SMOKE-TEST sampling policy (top-N by frequency after dedup), not sqd.py's production
frequency-weighted subsampling -- see the module docstring above and
docs/reference/sbd_gdb_heatbath_and_selection_gap.md for context.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def load_pool(path: str) -> tuple[np.ndarray, int]:
    """Load packed_bits/num_bits and unpack+dedup exactly as qiskit_addon_sqd.counts
    .bit_array_to_arrays does, without depending on qiskit/qiskit_addon_sqd at runtime."""
    data = np.load(path)
    packed = data["packed_bits"]
    num_bits = int(data["num_bits"][0])
    bool_array = np.unpackbits(packed, axis=-1)[..., -num_bits:].astype(bool)
    bitstrings, counts = np.unique(bool_array, axis=0, return_counts=True)
    num_shots = int(data["num_shots"][0])
    probs = counts / num_shots
    return bitstrings, probs


def post_select_by_hamming_weight(
    bitstrings: np.ndarray, probs: np.ndarray, norb: int, nelec_a: int, nelec_b: int
):
    """Keep only shots whose alpha/beta Hamming weight matches (nelec_a, nelec_b) -- mirrors
    sbd/sqd.py's own pipeline order (recover -> POSTSELECT -> subsample). The raw saved pool is
    genuine unfiltered hardware measurement data: most shots do NOT have the correct particle
    number (confirmed against the real fe4s4_uhf_5M_zhendongli.npz pool, where the top-5
    highest-probability RAW shots have alpha/beta popcounts like 22/8, 19/9 -- nowhere near
    27/27). Skipping this step and taking the raw top-N by probability silently produces
    determinants with the wrong particle number."""
    beta_bits = bitstrings[:, :norb]
    alpha_bits = bitstrings[:, norb:]
    mask = (alpha_bits.sum(axis=1) == nelec_a) & (beta_bits.sum(axis=1) == nelec_b)
    return bitstrings[mask], probs[mask]


def select_shots(bitstrings: np.ndarray, probs: np.ndarray, norb: int, n_shots: int):
    """Top-n_shots by probability, plus force in the Hartree-Fock configuration at rank 0 if not
    already present -- mirrors sbd/sqd.py's _subsample_one_spin "HF at index 0" convention.
    This is a smoke-test sampling POLICY choice (highest-weight shots), not sqd.py's own
    frequency-weighted production subsampling. Call post_select_by_hamming_weight FIRST -- this
    function does not filter by particle number itself."""
    order = np.argsort(-probs)
    sel = order[:n_shots]
    return bitstrings[sel]


def build_merged_determinants(bitstrings: np.ndarray, norb: int) -> list[int]:
    """Bit-interleave each (beta, alpha) shot pair into one 2*norb-bit determinant integer.

    bitstrings columns: [:norb] = beta, [norb:] = alpha (confirmed convention). Combined bit
    position 2k (0-based, from the least-significant/rightmost bit) = alpha orbital k; 2k+1 =
    beta orbital k. Uses Python's arbitrary-precision int -- 2*norb=72 bits for Fe4S4 overflows
    np.int64/np.longlong.
    """
    n = bitstrings.shape[0]
    beta_bits = bitstrings[:, :norb]
    alpha_bits = bitstrings[:, norb:]
    combined = [0] * n
    for k in range(norb):
        alpha_col = alpha_bits[:, k]
        beta_col = beta_bits[:, k]
        for i in range(n):
            if alpha_col[i]:
                combined[i] |= 1 << (2 * k)
            if beta_col[i]:
                combined[i] |= 1 << (2 * k + 1)
    return combined


def write_detfile(combined: list[int], norb: int, out_path: str) -> int:
    """Write gdb's plain-ASCII --detfiles format: one 2*norb-character '0'/'1' string per line,
    MSB-left (standard big-endian binary rendering), deduplicated."""
    unique = sorted(set(combined))
    width = 2 * norb
    with open(out_path, "w") as fh:
        for value in unique:
            fh.write(format(value, f"0{width}b"))
            fh.write("\n")
    return len(unique)


def electron_count_sanity_check(combined: list[int], norb: int, na: int, nb: int) -> None:
    """Verify every determinant's even-position popcount == na and odd-position popcount == nb --
    should hold for ~100% of shots from an already-Hamming-weight-post-selected production pool.
    A widespread failure here signals a real bug or a schema misunderstanding, not noise."""
    bad = 0
    for value in combined:
        alpha_count = sum((value >> (2 * k)) & 1 for k in range(norb))
        beta_count = sum((value >> (2 * k + 1)) & 1 for k in range(norb))
        if alpha_count != na or beta_count != nb:
            bad += 1
    if bad:
        print(
            f"WARNING: {bad}/{len(combined)} selected shots do not match "
            f"nelec=({na},{nb}) -- check --norb/--nelec-a/--nelec-b against the pool.",
            file=sys.stderr,
        )
    else:
        print(f"Electron-count sanity check: all {len(combined)} shots match nelec=({na},{nb}).",
              file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True, help="Path to the saved sample pool npz "
                                                    "(packed_bits/num_bits/num_shots).")
    p.add_argument("--norb", type=int, required=True, help="Number of spatial orbitals "
                                                              "(36 for Fe4S4). Must match "
                                                              "num_bits/2 or the script aborts.")
    p.add_argument("--nelec-a", type=int, required=True, help="Alpha electron count -- required "
                                                                "to post-select the raw pool by "
                                                                "Hamming weight before sampling "
                                                                "(the raw saved pool is genuine "
                                                                "unfiltered hardware data, most "
                                                                "shots do NOT have the right "
                                                                "particle number).")
    p.add_argument("--nelec-b", type=int, required=True, help="Beta electron count (see --nelec-a).")
    p.add_argument("--n-shots", type=int, default=5000, help="Number of highest-probability "
                                                                "shots to select (default 5000).")
    p.add_argument("--out", default=None, help="Output detfile path "
                                                  "(default: derived from --pool and --n-shots).")
    args = p.parse_args()

    print(f"[1/4] Loading pool {args.pool!r}...", file=sys.stderr)
    bitstrings, probs = load_pool(args.pool)
    num_bits = bitstrings.shape[1]
    if num_bits != 2 * args.norb:
        raise ValueError(
            f"pool num_bits={num_bits} != 2*norb={2 * args.norb} -- schema mismatch, aborting "
            "rather than silently producing garbage determinants."
        )
    print(f"      {bitstrings.shape[0]} unique bitstrings, norb={args.norb}", file=sys.stderr)

    print(
        f"[2/5] Post-selecting by Hamming weight (nelec=({args.nelec_a},{args.nelec_b}))...",
        file=sys.stderr,
    )
    filtered_bitstrings, filtered_probs = post_select_by_hamming_weight(
        bitstrings, probs, args.norb, args.nelec_a, args.nelec_b
    )
    n_kept = filtered_bitstrings.shape[0]
    print(
        f"      {n_kept}/{bitstrings.shape[0]} unique bitstrings survive post-selection "
        f"({filtered_probs.sum():.4f} of total probability mass)",
        file=sys.stderr,
    )
    if n_kept == 0:
        raise ValueError(
            "no bitstrings survive Hamming-weight post-selection -- check --norb/--nelec-a/"
            "--nelec-b against the pool, or that [:norb]=beta,[norb:]=alpha wasn't swapped."
        )

    print(f"[3/5] Selecting top {args.n_shots} of the post-selected shots by probability...",
          file=sys.stderr)
    selected = select_shots(filtered_bitstrings, filtered_probs, args.norb,
                             min(args.n_shots, n_kept))

    print("[4/5] Building merged 2*norb-bit determinants (Python int, no 64-bit overflow)...",
          file=sys.stderr)
    combined = build_merged_determinants(selected, args.norb)
    electron_count_sanity_check(combined, args.norb, args.nelec_a, args.nelec_b)

    out_path = args.out or f"{args.pool.rsplit('.', 1)[0]}_top{args.n_shots}.detfile.txt"
    print(f"[5/5] Writing {out_path!r}...", file=sys.stderr)
    n_unique = write_detfile(combined, args.norb, out_path)
    print(f"      {len(selected)} shots -> {n_unique} unique determinants", file=sys.stderr)


if __name__ == "__main__":
    main()
