#!/usr/bin/env python
"""Merge the three files `prepare_bsuhf_fcidump.py` writes into one interleaved-spin-orbital
FCIDUMP that a `-D_UHF`-compiled `sbd` binary (`gdb_diag_uhf`, `diag_uhf`) can read directly via
its `--fcidump` flag.

Background
----------
`prepare_bsuhf_fcidump.py` writes the converged BS-UHF reference as THREE separate files:
    <prefix>.alpha.fcidump   -- standard FCIDUMP: h1_a, (aa|aa)
    <prefix>.beta.fcidump    -- standard FCIDUMP: h1_b, (bb|bb)
    <prefix>.mixed.npz       -- eri_ab, the (aa|bb) block (no slot for this in a plain FCIDUMP)

`sbd`'s `_UHF`-compiled integral reader (`include/sbd/chemistry/basic/makeintegrals.h`,
`#ifdef _UHF` branch) does not read this three-file layout -- it reads ONE FCIDUMP whose records
use interleaved 1-based spin-orbital indices (alpha spatial orbital p -> spin-orbital 2p+1, beta
-> 2p+2), with the mixed (aa|bb) block folded in at the same records as everything else.

`algorithms/sbd/sbd/solver_job.py::_write_uhf_fcidump` already implements exactly this interleaved
writer (used for the production tpb/SQD UHF solver) -- this script reuses it directly rather than
reimplementing the interleaving/symmetrization logic.

Usage
-----
    python merge_bsuhf_to_uhf_fcidump.py --prefix fe4s4_bsuhf --out fe4s4_bsuhf.uhf.fcidump
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pyscf import ao2mo, tools


def _import_write_uhf_fcidump():
    # Reuse the already-tested writer (algorithms/sbd/tests/test_uhf.py) rather than
    # reimplementing its interleaving/symmetrization logic here. Default assumes this script
    # still lives under examples/fe4s4_hci_from_bsuhf_reference/ in the qcsc-prefect checkout;
    # override with SBD_PACKAGE_DIR when running from a copy staged elsewhere (e.g. a cluster
    # scratch/data area).
    import os

    sbd_pkg_dir = os.environ.get("SBD_PACKAGE_DIR")
    if sbd_pkg_dir is None:
        sbd_pkg_dir = Path(__file__).resolve().parents[2] / "algorithms" / "sbd"
    sys.path.insert(0, str(sbd_pkg_dir))
    from sbd.solver_job import _write_uhf_fcidump

    return _write_uhf_fcidump


def load_spin_fcidump(path: str) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Parse a standard (single-spin) FCIDUMP into dense (h1, h2, norb, nelec, ecore)."""
    ctx = tools.fcidump.read(path)
    norb = ctx["NORB"]
    nelec = ctx["NELEC"]
    ecore = ctx["ECORE"]
    h1 = ctx["H1"]
    h2 = ao2mo.restore(1, ctx["H2"], norb)
    return h1, h2, norb, nelec, ecore


def merge(prefix: str, out_path: str) -> None:
    write_uhf_fcidump = _import_write_uhf_fcidump()

    alpha_path = f"{prefix}.alpha.fcidump"
    beta_path = f"{prefix}.beta.fcidump"
    mixed_path = f"{prefix}.mixed.npz"

    print(f"[1/3] Reading {alpha_path!r} / {beta_path!r} / {mixed_path!r}...", file=sys.stderr)
    h1_a, h2_aa, norb_a, na, ecore = load_spin_fcidump(alpha_path)
    h1_b, h2_bb, norb_b, nb, ecore_b = load_spin_fcidump(beta_path)
    if norb_a != norb_b:
        raise ValueError(f"alpha norb={norb_a} != beta norb={norb_b}")
    norb = norb_a
    if ecore_b not in (0.0, ecore):
        raise ValueError(
            f"unexpected non-zero core energy in beta file ({ecore_b}); "
            "prepare_bsuhf_fcidump.py writes ecore into the alpha file only"
        )

    mixed = np.load(mixed_path)
    eri_ab = mixed["eri_ab"]
    mixed_norb = int(mixed["norb"])
    if mixed_norb != norb:
        raise ValueError(f"mixed.npz norb={mixed_norb} != alpha/beta norb={norb}")
    # prepare_bsuhf_fcidump.py writes this via ao2mo.general(..., compact=False), which PySCF
    # returns as a flat (norb**2, norb**2) array, not a (norb,norb,norb,norb) tensor -- a plain
    # reshape recovers the correct chemist-notation (ij|kl) tensor (verified against a direct
    # einsum contraction of the AO integrals with the four MO coefficient matrices).
    if eri_ab.shape == (norb * norb, norb * norb):
        eri_ab = eri_ab.reshape(norb, norb, norb, norb)
    if eri_ab.shape != (norb, norb, norb, norb):
        raise ValueError(f"eri_ab shape {eri_ab.shape} != expected {(norb, norb, norb, norb)}")

    print(
        f"      norb={norb}  nelec=({na},{nb})  ecore={ecore:.8f}",
        file=sys.stderr,
    )

    print(f"[2/3] Writing merged UHF FCIDUMP to {out_path!r}...", file=sys.stderr)
    write_uhf_fcidump(
        Path(out_path),
        h1_a=h1_a,
        h1_b=h1_b,
        h2_aa=h2_aa,
        h2_ab=eri_ab,
        h2_bb=h2_bb,
        norb=norb,
        nelec=(na, nb),
        ecore=ecore,
    )
    print("[3/3] Done.", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="fe4s4_bsuhf",
                    help="Prefix used by prepare_bsuhf_fcidump.py's --out-prefix (default: fe4s4_bsuhf)")
    p.add_argument("--out", default=None,
                    help="Output merged FCIDUMP path (default: <prefix>.uhf.fcidump)")
    args = p.parse_args()

    out_path = args.out or f"{args.prefix}.uhf.fcidump"
    merge(args.prefix, out_path)


if __name__ == "__main__":
    main()
