#!/usr/bin/env python
"""Reproduce the exact BS-UHF reference used for the [4Fe-4S] LUCJ/SQD study, and dump the
integrals in that MO basis as a new FCIDUMP, for anyone who wants to run a different solver
(HCI/selected-CI, DMRG, FCIQMC, ...) starting from the same reference instead of SQD.

Background
----------
This repo's SQD pipeline does not generate its own FCIDUMP -- it consumes one. The Fe4S4
(54 electrons, 36 orbitals) FCIDUMP used for the LUCJ circuit sampled on ibm_kingston (5M shots)
was supplied by a collaborator (Zhendong Li), already active-space-carved. Ask for
``fe4s4_zhendongli.txt`` if you don't already have it (sha256 0b840102a4894fee4648944f2bf3bb5d85
828deb4b7e71b964136bbc4c9e01ce, 9,648,600 bytes, header ``NORB=36, NELEC=54, MS2=0``).

What THIS FCIDUMP alone does not give you is the reference orbitals: the raw file is in whatever
MO basis the collaborator's own CASSCF/localization produced, and plain closed-shell RHF/UHF on
it just falls back onto the spin-pure RHF solution (a closed-shell singlet's RHF determinant is a
stationary point of the UHF equations, so unrestricted SCF started from it doesn't move). The
actual reference used for the paper's LUCJ sampling is a broken-symmetry UHF (BS-UHF) solution
reached via an atom-localized antiferromagnetic initial guess plus internal-AND-external
stability-following (the Noodleman recipe) -- described in the paper and implemented in
``algorithms/qcsc_workflow_utility/src/qcsc_workflow_utility/chem.py``
(``_parse_af_groups`` / ``_converge_broken_symmetry_uhf``).

This script reproduces exactly that reference standalone (no Prefect, no qcsc-prefect import),
and writes out the two-electron integrals rotated into the converged BS-UHF MO basis as a new
FCIDUMP -- so a different solver can start from the identical orbitals/reference this repo's SQD
pipeline samples its LUCJ circuit from.

Usage
-----
    pip install pyscf numpy   # only external deps
    python prepare_bsuhf_fcidump.py \
        --fcidump /path/to/fe4s4_zhendongli.txt \
        --af-groups fe4s4 \
        --out-prefix fe4s4_bsuhf_p12

Writes:
    fe4s4_bsuhf_p12.alpha.fcidump   -- alpha-spin MO-basis FCIDUMP (h1_a, (aa|aa))
    fe4s4_bsuhf_p12.beta.fcidump    -- beta-spin MO-basis FCIDUMP  (h1_b, (bb|bb))
    fe4s4_bsuhf_p12.mixed.npz       -- the (aa|bb) mixed-spin two-electron block, since standard
                                       FCIDUMP has no slot for it (needed by any solver that treats
                                       alpha/beta ERIs separately, e.g. a UHF-based determinant CI)
    fe4s4_bsuhf_p12.summary.txt     -- E(BS-UHF), <S^2>, occupations, orbital counts

The default --af-groups fe4s4 reproduces the pairing-(1,2) reference used throughout this work:
up={fe1,fe2}, down={fe3,fe4}, full polarization (pol=1.0) -- E=-327.0809 Ha, <S^2>=8.877. Pass
your own JSON (or "@file.json") to try a different fragment/pairing choice; see
docs/tutorials/run_uhf_bsuhf_any_molecule.md Step 1 for the schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import ao2mo, gto, scf, tools


def parse_af_groups(raw: str) -> dict:
    """Same convenience keyword + schema as chem.py's _parse_af_groups(), standalone (no env var,
    no Prefect import) so this script has zero qcsc-prefect dependency."""
    if raw.strip().lower() == "fe4s4":
        return {
            "l1": list(range(0, 2)),
            "fe1": list(range(2, 7)),
            "fe2": list(range(7, 12)),
            "s": list(range(12, 24)),
            "fe3": list(range(24, 29)),
            "fe4": list(range(29, 34)),
            "l2": list(range(34, 36)),
            # Pairing-(1,2): reaches the deepest verified BS-UHF basin directly at full
            # polarization -- see docs/tutorials/run_uhf_bsuhf_any_molecule.md Step 1.
            "up": ["fe1", "fe2"],
            "down": ["fe3", "fe4"],
            "pol": 1.0,
        }
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text()
    return json.loads(raw)


def af_guess_density(norb: int, af_groups: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build the atom-localized (Noodleman) antiferromagnetic guess density: fragments in "up"
    get alpha-heavy occupation, "down" get beta-heavy, "free" get half-filled, everything else is
    closed (doubly occupied). Mirrors chem.py's _af_guess_uhf inner loop.

    Takes norb directly (from the FCIDUMP header) rather than mol.nao: the Mole built from a bare
    FCIDUMP has no atoms, so mol.nao is 0 -- it is only used here to override get_hcore/_eri."""
    n = norb
    up = set(af_groups.get("up", []))
    down = set(af_groups.get("down", []))
    free = set(af_groups.get("free", []))
    pol = af_groups.get("pol", 1.0)
    frags = {k: v for k, v in af_groups.items() if k not in ("up", "down", "free", "pol")}

    dm0 = np.zeros((2, n, n))
    for name, orbs in frags.items():
        for x in orbs:
            if name in up:
                dm0[0, x, x] = 0.5 + 0.5 * pol
                dm0[1, x, x] = 0.5 - 0.5 * pol
            elif name in down:
                dm0[0, x, x] = 0.5 - 0.5 * pol
                dm0[1, x, x] = 0.5 + 0.5 * pol
            elif name in free:
                dm0[0, x, x] = 0.5
                dm0[1, x, x] = 0.5
            else:  # closed fragment: doubly occupied
                dm0[0, x, x] = 1.0
                dm0[1, x, x] = 1.0
    return dm0[0], dm0[1]


def follow_stability_external(m, n_iter: int = 10):
    """Follow BOTH internal and external instabilities, keeping the lowest-energy iterate seen.
    The Noodleman AF sublattice minima are near-degenerate and separated by EXTERNAL
    (spin-symmetry-breaking) instabilities that plain internal stability-following does not
    cross -- see chem.py's _follow_stability_external docstring for why this matters."""
    best = m
    for _ in range(n_iter):
        res = m.stability(return_status=True, external=True)
        mo = res[0]
        stable_i, stable_e = res[-2], res[-1]
        try:
            m = m.newton().run(mo if isinstance(mo, tuple) else m.mo_coeff)
        except Exception:
            break
        if m.e_tot < best.e_tot:
            best = m
        if stable_i and stable_e:
            break
    return best


def converge_bsuhf(fcidump_path: str, af_groups: dict):
    """Load the FCIDUMP via tools.fcidump.to_scf (the proven recipe from chem.py's
    compute_molecular_integrals_from_fcidump -- a bare gto.Mole() with no atom/basis never gets a
    real mol.nao, so a hand-rolled Mole silently breaks the AF guess sizing), disable symmetry
    (an ORBSYM-carrying FCIDUMP makes to_scf build a symmetry-adapted mol, which blocks the plain
    UHF kernel/stability path), convert to a plain scf.UHF carrying the FCIDUMP integrals across,
    seed the atom-localized AF guess, and stability-follow (internal + external) to the
    broken-symmetry minimum."""
    ctx = tools.fcidump.read(fcidump_path)
    norb, nelec, ms2 = ctx["NORB"], ctx["NELEC"], ctx["MS2"]
    na, nb = (nelec + ms2) // 2, (nelec - ms2) // 2

    mf = tools.fcidump.to_scf(fcidump_path)
    mf.mol.verbose = 0
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = mf._eri
    ecore = mf.mol.energy_nuc()

    base_mol = mf.mol
    if getattr(base_mol, "symmetry", False):
        base_mol.symmetry = False
        base_mol.build(dump_input=False, parse_arg=False)

    m = scf.UHF(base_mol)
    m.get_hcore = lambda *a, **k: hcore
    m.get_ovlp = lambda *a, **k: ovlp
    m._eri = eri
    m.mol.energy_nuc = lambda *a: ecore
    m.symmetry = False
    m.max_cycle = 400
    m.conv_tol = 1e-9

    dma, dmb = af_guess_density(norb, af_groups)
    m.kernel(dm0=(dma, dmb))
    m = follow_stability_external(m)
    return m, norb, (na, nb), ecore


def write_spin_fcidump(path: str, h1: np.ndarray, eri: np.ndarray, norb: int, nelec: int,
                        ecore: float) -> None:
    """Write a same-spin block (h1, (xx|xx)) as a standard FCIDUMP. eri must already be in
    8-fold (or 4-fold) restored form for tools.fcidump.from_integrals."""
    tools.fcidump.from_integrals(path, h1, eri, norb, nelec, nuc=ecore, ms=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fcidump", required=True, help="Path to fe4s4_zhendongli.txt (or your own).")
    p.add_argument("--af-groups", default="fe4s4",
                   help='"fe4s4" (pairing-(1,2), default) or raw JSON / "@file.json".')
    p.add_argument("--out-prefix", default="fe4s4_bsuhf")
    args = p.parse_args()

    af_groups = parse_af_groups(args.af_groups)
    print(f"[1/3] Converging BS-UHF from FCIDUMP {args.fcidump!r} with af_groups="
          f"{ {k: v for k, v in af_groups.items() if k not in ('l1','l2','s','fe1','fe2','fe3','fe4')} }...",
          file=sys.stderr)
    mf, norb, (na, nb), ecore = converge_bsuhf(args.fcidump, af_groups)
    spin_sq, mult = mf.spin_square()
    print(f"      E(BS-UHF) = {mf.e_tot:.6f} Ha   <S^2> = {spin_sq:.4f}   "
          f"nelec=({na},{nb})   norb={norb}", file=sys.stderr)

    print("[2/3] Rotating two-electron integrals into the converged BS-UHF MO basis...",
          file=sys.stderr)
    mo_a, mo_b = mf.mo_coeff
    eri_ao = ao2mo.restore(8, mf._eri, norb)
    eri_aa = ao2mo.general(eri_ao, (mo_a, mo_a, mo_a, mo_a), compact=True)
    eri_bb = ao2mo.general(eri_ao, (mo_b, mo_b, mo_b, mo_b), compact=True)
    eri_ab = ao2mo.general(eri_ao, (mo_a, mo_a, mo_b, mo_b), compact=False)
    h1_ao = mf.get_hcore()
    h1_a = mo_a.T @ h1_ao @ mo_a
    h1_b = mo_b.T @ h1_ao @ mo_b

    print("[3/3] Writing FCIDUMPs / summary...", file=sys.stderr)
    write_spin_fcidump(f"{args.out_prefix}.alpha.fcidump", h1_a, eri_aa, norb, na, ecore)
    write_spin_fcidump(f"{args.out_prefix}.beta.fcidump", h1_b, eri_bb, norb, nb, 0.0)
    np.savez(f"{args.out_prefix}.mixed.npz", eri_ab=eri_ab, norb=norb)
    with open(f"{args.out_prefix}.summary.txt", "w") as fh:
        fh.write(
            f"source_fcidump: {args.fcidump}\n"
            f"af_groups: {json.dumps(af_groups)}\n"
            f"norb: {norb}\nnelec: ({na}, {nb})\necore: {ecore}\n"
            f"E_BSUHF: {mf.e_tot:.8f}\nspin_sq: {spin_sq:.6f}\nmultiplicity: {mult:.6f}\n"
        )
    print(f"Wrote {args.out_prefix}.{{alpha,beta}}.fcidump, .mixed.npz, .summary.txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
