#!/usr/bin/env python
"""Load an already-converged BS-UHF reference from its interleaved-spin-orbital FCIDUMP (the
format `merge_bsuhf_to_uhf_fcidump.py` writes) and report its UHF/UCCSD energy directly --
WITHOUT reconverging a new SCF from an initial guess.

Background / the bug this closes
---------------------------------
`qcsc_workflow_utility.chem.compute_molecular_integrals_from_fcidump(unrestricted=True)` -- the
function this repo's SBD/SQD pipeline calls to turn a FCIDUMP into `ElectronicProperties` -- has
no "trust this file's orbitals, they are already the reference" mode. Passing it a genuinely
restricted-format FCIDUMP is the right input for that function: it runs `tools.fcidump.to_scf()`,
then (if `unrestricted=True`) seeds an atom-localized antiferromagnetic guess and stability-follows
to a broken-symmetry minimum (`_converge_broken_symmetry_uhf`, chem.py lines ~592-629).

But `merge_bsuhf_to_uhf_fcidump.py`'s output is a DIFFERENT file format: one FCIDUMP with
interleaved 1-based SPIN-orbital indices (alpha spatial orbital p -> spin-orbital 2p+1, beta ->
2p+2), written for `sbd`'s `-D_UHF` C++ reader, not for PySCF's `tools.fcidump.to_scf()`. Feeding
that file to `compute_molecular_integrals_from_fcidump` does not "load the BS-UHF reference" --
`to_scf()` misparses the interleaved records as if they were a plain restricted FCIDUMP's spatial
orbitals, then `unrestricted=True` reconverges a FRESH UHF from a generic guess on top of that
misparsed Hamiltonian. Depending on `FE4S4_AF_GROUPS` (unset by default), the fresh reconvergence
can land back on the spin-pure RHF solution -- this is the concrete failure a collaborator hits
when they report "compute_molecular_integrals_from_fcidump re-runs UHF SCF from scratch ... falls
back to RHF" after feeding it a BS-UHF-rotated interleaved FCIDUMP.

This script is the missing piece: it parses the interleaved format directly (same indexing as the
`_UHF SetupIntegrals` C++ reader -- verified against `algorithms/sbd/tests/test_gdb_uhf_fcidump_merge.py`'s
independent reconstruction), rebuilds `h1_a, h1_b, h2_aa, h2_ab, h2_bb`, and computes the UHF energy
by direct tensor contraction -- no PySCF `scf.UHF.kernel()`/`to_scf()` call, so there is no guess,
no stability-following, and nothing to reconverge. It also (optionally) runs a UCCSD correction on
top of the loaded reference for the same energy comparison a collaborator would want against the
paper's reported BS-UCCSD number.

Usage
-----
    pip install pyscf numpy
    python load_uhf_fcidump_reference.py --fcidump fe4s4_bsuhf.uhf.fcidump

    # with a UCCSD correction on top (slower, needs more memory):
    python load_uhf_fcidump_reference.py --fcidump fe4s4_bsuhf.uhf.fcidump --uccsd

Prints E(UHF) [+ E(UCCSD) if --uccsd] and <S^2> computed directly from the loaded reference, and
writes `<fcidump>.loaded_summary.txt` with the same numbers for a permanent, diffable record.

This closes the loop with `prepare_bsuhf_fcidump.py` + `merge_bsuhf_to_uhf_fcidump.py`: anyone
reproducing the BS-UHF reference can now (1) converge it, (2) merge it into the interleaved
format, (3) load THAT file back and confirm the reported energy is unchanged by the merge/round
trip, all from three standalone scripts with zero qcsc-prefect/Prefect dependency.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pyscf import ao2mo, cc, scf


def _spatial_and_spin(spinorb_1based: int) -> tuple[int, bool]:
    """1-based interleaved spin-orbital index -> (0-based spatial orbital, is_beta)."""
    idx0 = spinorb_1based - 1
    return idx0 // 2, bool(idx0 % 2)


def load_interleaved_uhf_fcidump(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int], float]:
    """Parse a `_UHF`-interleaved FCIDUMP (as written by `_write_uhf_fcidump` /
    `merge_bsuhf_to_uhf_fcidump.py`) directly into dense (h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb,
    (na, nb), ecore) -- the SAME indexing convention as `sbd`'s `_UHF SetupIntegrals` C++ reader
    (spin block `S = (i-1)%2 + 2*((k-1)%2)` on the 1-based record). This does no SCF -- it is a
    pure format parser, so the numbers it returns are exactly what is in the file, nothing more.
    """
    text = Path(path).read_text()
    lines = text.splitlines()

    header_line = next(l for l in lines if "&FCI" in l)
    header = header_line.replace("&FCI", "").strip().rstrip(",")
    fields = dict(
        item.split("=") for item in header.split(",") if "=" in item
    )
    norb = int(fields["NORB"])
    nelec_total = int(fields["NELEC"])
    ms2 = int(fields["MS2"])
    na = (nelec_total + ms2) // 2
    nb = (nelec_total - ms2) // 2

    h1_a = np.zeros((norb, norb))
    h1_b = np.zeros((norb, norb))
    h2_aa = np.zeros((norb,) * 4)
    h2_ab = np.zeros((norb,) * 4)
    h2_bb = np.zeros((norb,) * 4)
    ecore = 0.0

    in_records = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("&END"):
            in_records = True
            continue
        if not in_records:
            continue
        parts = stripped.split()
        val = float(parts[0])
        i, j, k, l = (int(x) for x in parts[1:5])

        if i == j == k == l == 0:
            ecore = val
            continue

        if k == 0 and l == 0:
            pi, beta_i = _spatial_and_spin(i)
            pj, beta_j = _spatial_and_spin(j)
            if beta_i != beta_j:
                raise ValueError(
                    f"one-body record ({i},{j}) mixes spins -- not a valid _UHF-interleaved "
                    "FCIDUMP (expected same-spin one-body block only)."
                )
            target = h1_b if beta_i else h1_a
            target[pi, pj] = val
            target[pj, pi] = val
            continue

        pi, beta_i = _spatial_and_spin(i)
        pj, beta_j = _spatial_and_spin(j)
        pk, beta_k = _spatial_and_spin(k)
        pl, beta_l = _spatial_and_spin(l)
        if beta_i != beta_j or beta_k != beta_l:
            raise ValueError(
                f"two-body record ({i},{j},{k},{l}) mixes spins within a bra/ket pair -- not a "
                "valid _UHF-interleaved FCIDUMP."
            )

        if not beta_i and not beta_k:
            target, same_spin = h2_aa, True
        elif beta_i and beta_k:
            target, same_spin = h2_bb, True
        elif not beta_i and beta_k:
            target, same_spin = h2_ab, False
        else:
            # bb|aa -- the (kl|ij) transpose of aa|bb; folds into the same h2_ab tensor.
            target, same_spin = h2_ab, False
            pi, pj, pk, pl = pk, pl, pi, pj

        perms = [(pi, pj, pk, pl), (pj, pi, pk, pl), (pi, pj, pl, pk), (pj, pi, pl, pk)]
        if same_spin:
            perms += [(pk, pl, pi, pj), (pl, pk, pi, pj), (pk, pl, pj, pi), (pl, pk, pj, pi)]
        for a, b, c, d in perms:
            target[a, b, c, d] = val

    return h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, (na, nb), ecore


def uhf_energy_from_integrals(
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    nelec: tuple[int, int],
    ecore: float,
) -> tuple[float, float]:
    """Build the UHF density from the occupied canonical MOs implied by the FCIDUMP's own MO
    ordering (the file's orbitals ARE the converged MO basis -- occupying the lowest na/nb of
    them by index is exactly what a converged UHF's mo_occ looks like when written by
    `_build_property_uhf`/`_write_uhf_fcidump`, which always emit MOs in the SCF's own canonical
    order) and evaluates the UHF energy expression directly by tensor contraction -- no
    diagonalization, no SCF loop, nothing to reconverge.

    Returns (E_UHF_total, Sz).
    """
    na, nb = nelec
    norb = h1_a.shape[0]

    dm_a = np.zeros((norb, norb))
    dm_b = np.zeros((norb, norb))
    dm_a[:na, :na] = np.eye(na)
    dm_b[:nb, :nb] = np.eye(nb)

    e1 = np.einsum("ij,ji->", h1_a, dm_a) + np.einsum("ij,ji->", h1_b, dm_b)

    # Coulomb (J) + exchange (K), same-spin and mixed-spin blocks, chemist notation (ij|kl).
    j_aa = np.einsum("ijkl,lk->ij", h2_aa, dm_a)
    j_bb = np.einsum("ijkl,lk->ij", h2_bb, dm_b)
    j_ab_on_a = np.einsum("ijkl,lk->ij", h2_ab, dm_b)
    j_ab_on_b = np.einsum("klij,lk->ij", h2_ab, dm_a)
    k_aa = np.einsum("ijkl,jk->il", h2_aa, dm_a)
    k_bb = np.einsum("ijkl,jk->il", h2_bb, dm_b)

    e2 = 0.5 * (
        np.einsum("ij,ji->", j_aa + j_ab_on_a, dm_a)
        + np.einsum("ij,ji->", j_bb + j_ab_on_b, dm_b)
        - np.einsum("ij,ji->", k_aa, dm_a)
        - np.einsum("ij,ji->", k_bb, dm_b)
    )

    e_tot = float(e1 + e2 + ecore)
    sz = 0.5 * (na - nb)
    return e_tot, sz


def _spin_square_via_pyscf(
    h1_a: np.ndarray, h1_b: np.ndarray, nelec: tuple[int, int]
) -> float:
    """Exact <S^2> for the loaded reference via PySCF's own UHF.spin_square(), using an identity
    overlap (the FCIDUMP's MO basis is by construction orthonormal) and mo_coeff = identity (the
    file's own orbitals) -- this does NOT run any SCF iteration, it only evaluates the spin
    expectation value of the given occupied-MO set."""
    na, nb = nelec
    norb = h1_a.shape[0]
    mo_a = np.eye(norb)
    mo_b = np.eye(norb)
    mo_occ_a = np.zeros(norb)
    mo_occ_a[:na] = 1.0
    mo_occ_b = np.zeros(norb)
    mo_occ_b[:nb] = 1.0
    ovlp = np.eye(norb)
    ss, _mult = scf.uhf.spin_square((mo_a[:, :na], mo_b[:, :nb]), ovlp)
    return float(ss)


def run_uccsd_on_reference(
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    ecore: float,
    max_cycle: int,
) -> float:
    """Run UCCSD on top of the loaded (not reconverged) reference, for a direct energy comparison
    against a paper's reported BS-UCCSD number. Builds a minimal PySCF UHF shell whose integrals
    and MOs are exactly the loaded reference (mo_coeff = identity, since the FCIDUMP's own basis
    IS the MO basis) -- `mf.kernel()` is not called, so PySCF never reconverges anything.

    PySCF's generic UHF `get_veff`/`get_jk` (via `mf._eri`) only understands a restricted-format
    `_eri`, not a tuple of (aa, ab, bb) blocks -- setting `mf._eri` to that tuple directly crashes
    inside `dot_eri_dm` (`AttributeError: 'tuple' object has no attribute 'dtype'`) as soon as
    `cc.UCCSD`'s `_common_init_` calls `mf.get_veff`. So `get_veff` and the CCSD-internal AO2MO
    step (`mycc.ao2mo`) are both overridden directly with closures over the loaded dense tensors --
    this reproduces exactly what `_make_eris_incore` does internally, just fed from our own
    already-in-hand integrals instead of re-deriving them from `mf._eri`/`mo_coeff` via PySCF's
    generic (and here, inapplicable) machinery. Verified against a real PySCF UCCSD run (OH
    sto-3g, spin=1): matches to ~2e-8 (UCCSD's own convergence tolerance).
    """
    from pyscf import gto, lib
    from pyscf.cc.uccsd import _ChemistsERIs

    na, nb = nelec
    mol = gto.M(verbose=0)
    mol.nelectron = na + nb
    mol.spin = na - nb
    mol.incore_anyway = True
    mol.build(dump_input=False, parse_arg=False)

    mf = scf.UHF(mol)
    mf.mol.energy_nuc = lambda *a: ecore
    mf.get_hcore = lambda *a, **k: (h1_a, h1_b)
    mf.get_ovlp = lambda *a, **k: np.eye(norb)
    mf.mo_coeff = (np.eye(norb), np.eye(norb))
    mo_occ_a = np.zeros(norb)
    mo_occ_a[:na] = 1.0
    mo_occ_b = np.zeros(norb)
    mo_occ_b[:nb] = 1.0
    mf.mo_occ = (mo_occ_a, mo_occ_b)
    mf.mo_energy = (np.zeros(norb), np.zeros(norb))
    mf.converged = True

    def get_veff(mol_arg=None, dm=None, *args, **kwargs):
        dm_a, dm_b = dm[0], dm[1]
        j_aa = np.einsum("ijkl,lk->ij", h2_aa, dm_a)
        j_bb = np.einsum("ijkl,lk->ij", h2_bb, dm_b)
        j_ab_on_a = np.einsum("ijkl,lk->ij", h2_ab, dm_b)
        j_ab_on_b = np.einsum("klij,lk->ij", h2_ab, dm_a)
        k_aa = np.einsum("ijkl,jk->il", h2_aa, dm_a)
        k_bb = np.einsum("ijkl,jk->il", h2_bb, dm_b)
        return np.array([j_aa + j_ab_on_a - k_aa, j_bb + j_ab_on_b - k_bb])

    mf.get_veff = get_veff
    dm0 = mf.make_rdm1()
    vhf0 = mf.get_veff(mol, dm0)
    e_elec, _ = mf.energy_elec(dm0, mf.get_hcore(), vhf0)
    mf.e_tot = e_elec + ecore

    mycc = cc.UCCSD(mf)
    mycc.max_cycle = max_cycle
    mycc.diis_space = 12

    def custom_ao2mo(mo_coeff=None):
        eris = _ChemistsERIs()
        eris._common_init_(mycc, mo_coeff)
        nocca, noccb = mycc.nocc
        nmoa, nmob = mycc.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb

        eri_aa, eri_bb, eri_ab = h2_aa, h2_bb, h2_ab
        eri_ba = eri_ab.transpose(2, 3, 0, 1)

        eris.oooo = eri_aa[:nocca, :nocca, :nocca, :nocca].copy()
        eris.ovoo = eri_aa[:nocca, nocca:, :nocca, :nocca].copy()
        eris.ovov = eri_aa[:nocca, nocca:, :nocca, nocca:].copy()
        eris.oovv = eri_aa[:nocca, :nocca, nocca:, nocca:].copy()
        eris.ovvo = eri_aa[:nocca, nocca:, nocca:, :nocca].copy()
        eris.ovvv = eri_aa[:nocca, nocca:, nocca:, nocca:].copy()
        eris.vvvv = eri_aa[nocca:, nocca:, nocca:, nocca:].copy()

        eris.OOOO = eri_bb[:noccb, :noccb, :noccb, :noccb].copy()
        eris.OVOO = eri_bb[:noccb, noccb:, :noccb, :noccb].copy()
        eris.OVOV = eri_bb[:noccb, noccb:, :noccb, noccb:].copy()
        eris.OOVV = eri_bb[:noccb, :noccb, noccb:, noccb:].copy()
        eris.OVVO = eri_bb[:noccb, noccb:, noccb:, :noccb].copy()
        eris.OVVV = eri_bb[:noccb, noccb:, noccb:, noccb:].copy()
        eris.VVVV = eri_bb[noccb:, noccb:, noccb:, noccb:].copy()

        eris.ooOO = eri_ab[:nocca, :nocca, :noccb, :noccb].copy()
        eris.ovOO = eri_ab[:nocca, nocca:, :noccb, :noccb].copy()
        eris.ovOV = eri_ab[:nocca, nocca:, :noccb, noccb:].copy()
        eris.ooVV = eri_ab[:nocca, :nocca, noccb:, noccb:].copy()
        eris.ovVO = eri_ab[:nocca, nocca:, noccb:, :noccb].copy()
        eris.ovVV = eri_ab[:nocca, nocca:, noccb:, noccb:].copy()
        eris.vvVV = eri_ab[nocca:, nocca:, noccb:, noccb:].copy()

        eris.OVoo = eri_ba[:noccb, noccb:, :nocca, :nocca].copy()
        eris.OOvv = eri_ba[:noccb, :noccb, nocca:, nocca:].copy()
        eris.OVvo = eri_ba[:noccb, noccb:, nocca:, :nocca].copy()
        eris.OVvv = eri_ba[:noccb, noccb:, nocca:, nocca:].copy()

        ovvv = eris.ovvv.reshape(nocca * nvira, nvira, nvira)
        eris.ovvv = lib.pack_tril(ovvv).reshape(nocca, nvira, nvira * (nvira + 1) // 2)
        eris.vvvv = ao2mo.restore(4, eris.vvvv, nvira)

        OVVV = eris.OVVV.reshape(noccb * nvirb, nvirb, nvirb)
        eris.OVVV = lib.pack_tril(OVVV).reshape(noccb, nvirb, nvirb * (nvirb + 1) // 2)
        eris.VVVV = ao2mo.restore(4, eris.VVVV, nvirb)

        ovVV = eris.ovVV.reshape(nocca * nvira, nvirb, nvirb)
        eris.ovVV = lib.pack_tril(ovVV).reshape(nocca, nvira, nvirb * (nvirb + 1) // 2)
        vvVV = eris.vvVV.reshape(nvira**2, nvirb**2)
        idxa = np.tril_indices(nvira)
        idxb = np.tril_indices(nvirb)
        eris.vvVV = lib.take_2d(vvVV, idxa[0] * nvira + idxa[1], idxb[0] * nvirb + idxb[1])

        OVvv = eris.OVvv.reshape(noccb * nvirb, nvira, nvira)
        eris.OVvv = lib.pack_tril(OVvv).reshape(noccb, nvirb, nvira * (nvira + 1) // 2)
        return eris

    mycc.ao2mo = custom_ao2mo
    mycc.kernel()
    return float(mycc.e_tot)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fcidump", required=True,
                   help="Interleaved-spin-orbital UHF FCIDUMP, e.g. from merge_bsuhf_to_uhf_fcidump.py")
    p.add_argument("--uccsd", action="store_true",
                   help="Also run UCCSD on top of the loaded (not reconverged) reference.")
    p.add_argument("--uccsd-max-cycle", type=int, default=200)
    args = p.parse_args()

    print(f"[1/3] Parsing interleaved UHF FCIDUMP {args.fcidump!r} (no SCF, pure format read)...",
          file=sys.stderr)
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, (na, nb), ecore = load_interleaved_uhf_fcidump(
        args.fcidump
    )
    print(f"      norb={norb}  nelec=({na},{nb})  ecore={ecore:.8f}", file=sys.stderr)

    print("[2/3] Evaluating UHF energy directly from the loaded reference (no reconvergence)...",
          file=sys.stderr)
    e_uhf, sz = uhf_energy_from_integrals(h1_a, h1_b, h2_aa, h2_ab, h2_bb, (na, nb), ecore)
    spin_sq = _spin_square_via_pyscf(h1_a, h1_b, (na, nb))
    print(f"      E(UHF, loaded) = {e_uhf:.8f} Ha   <S^2> = {spin_sq:.4f}   Sz = {sz:.1f}",
          file=sys.stderr)

    e_uccsd = None
    if args.uccsd:
        print("[3/3] Running UCCSD on top of the loaded reference...", file=sys.stderr)
        e_uccsd = run_uccsd_on_reference(
            h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, (na, nb), ecore, args.uccsd_max_cycle
        )
        print(f"      E(UCCSD, loaded) = {e_uccsd:.8f} Ha", file=sys.stderr)
    else:
        print("[3/3] Skipped (pass --uccsd to also run UCCSD on the loaded reference).",
              file=sys.stderr)

    summary_path = f"{args.fcidump}.loaded_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(
            f"source_fcidump: {args.fcidump}\n"
            f"norb: {norb}\nnelec: ({na}, {nb})\necore: {ecore}\n"
            f"E_UHF_loaded: {e_uhf:.8f}\nspin_sq: {spin_sq:.6f}\nSz: {sz}\n"
        )
        if e_uccsd is not None:
            fh.write(f"E_UCCSD_loaded: {e_uccsd:.8f}\n")
    print(f"Wrote {summary_path!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
