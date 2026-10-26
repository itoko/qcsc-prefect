"""Interleaved spin-orbital FCIDUMP I/O (the format the SBD ``-D_UHF`` build reads).

This module is the single definition of the spin-orbital FCIDUMP format shared by the workflow
(``sbd.solver_job`` writes it for every UHF solve; ``chem.compute_molecular_integrals_from_uhf_fcidump``
reads it as the workflow input) and by the external orbital converters (``rhf2uhf``, ``rhf2bsuhf``).

Format (matches the C++ ``_UHF SetupIntegrals`` parser):

* Header ``NORB`` is the number of SPATIAL orbitals; records are 1-based over ``1..2*NORB`` with
  alpha spatial orbital ``p`` (0-based) -> ``2p+1`` and beta -> ``2p+2``.
* Two-body records ``(i,j,k,l)`` are chemist ``(ij|kl)``; both indices of a pair share a spin.
  The spin block is ``S = (i-1)%2 + 2*((k-1)%2)``: S=0 aa|aa, S=1 bb|aa (transpose of aa|bb),
  S=2 aa|bb, S=3 bb|bb. Same-spin blocks are emitted with the 8-fold cut, the mixed block with
  the 4-fold cut only (i>=j, k>=l).
* One-body records have ``k=l=0``; the core energy record is ``0 0 0 0``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

_HEADER_INT = re.compile(r"\b(NORB|NELEC|MS2)\s*=\s*(-?\d+)", re.IGNORECASE)


def write_uhf_fcidump(
    path: str | Path,
    *,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    ecore: float = 0.0,
) -> None:
    """Write an interleaved-spin-orbital FCIDUMP for the SBD ``-D_UHF`` build.

    Tensors are chemist-notation ``(ij|kl)`` as returned by PySCF ``ao2mo``, dense ``(norb,)*4``.
    The reader applies its own symmetrization, so each unique record is emitted once: same-spin
    blocks with ``i>=j, k>=l, ij>=kl``; the mixed ``aa|bb`` block with ``i>=j, k>=l`` only (its
    bra<->ket swap is the distinct ``bb|aa`` block, which the reader rebuilds by transposition).
    Zero-valued records are skipped. ``ecore`` is written as the ``0 0 0 0`` record.
    """
    nelec_a, nelec_b = nelec

    lines: list[str] = []
    lines.append(
        f" &FCI NORB={norb},NELEC={nelec_a + nelec_b},MS2={nelec_a - nelec_b},"
    )
    lines.append("  ORBSYM=" + ",".join(["1"] * norb) + ",")
    lines.append("  ISYM=1,")
    lines.append(" &END")

    def spinorb(spatial: int, spin_is_beta: bool) -> int:
        # 1-based spin-orbital index: alpha -> 2*p+1, beta -> 2*p+2 (p 0-based spatial).
        return 2 * spatial + (2 if spin_is_beta else 1)

    def emit_two_body(tensor: np.ndarray, bra_beta: bool, ket_beta: bool) -> None:
        # Chemist (ij|kl). Same-spin blocks (aa|aa, bb|bb) have full 8-fold symmetry, so emit only
        # i>=j, k>=l, (i,j)>=(k,l). The mixed block (aa|bb) only has the 4-fold permutation
        # symmetry i<->j and k<->l (the bra<->ket swap maps it to the distinct bb|aa block, which
        # the reader reconstructs by its own symmetrization), so for it we must NOT apply the
        # (i,j)>=(k,l) cut or we drop valid records like (00|11).
        same_spin = bra_beta == ket_beta
        for i in range(norb):
            for j in range(i + 1):
                ij = i * (i + 1) // 2 + j
                for k in range(norb):
                    for m in range(k + 1):
                        kl = k * (k + 1) // 2 + m
                        if same_spin and ij < kl:
                            continue
                        val = float(tensor[i, j, k, m])
                        if val == 0.0:
                            continue
                        si = spinorb(i, bra_beta)
                        sj = spinorb(j, bra_beta)
                        sk = spinorb(k, ket_beta)
                        sl = spinorb(m, ket_beta)
                        lines.append(f"{val:>28.16e} {si:4d} {sj:4d} {sk:4d} {sl:4d}")

    # Two-body blocks: S=0 aa|aa, S=2 aa|bb, S=3 bb|bb. The S=1 bb|aa block is the (kl|ij)
    # transpose of aa|bb and is recovered by the reader's symmetrization, so we emit aa|bb only.
    emit_two_body(h2_aa, bra_beta=False, ket_beta=False)
    emit_two_body(h2_ab, bra_beta=False, ket_beta=True)
    emit_two_body(h2_bb, bra_beta=True, ket_beta=True)

    # One-body: alpha block at even spin-orbitals, beta at odd. k=l=0 marks a one-body record.
    def emit_one_body(h1: np.ndarray, beta: bool) -> None:
        for i in range(norb):
            for j in range(i + 1):
                val = float(h1[i, j])
                if val == 0.0:
                    continue
                si = spinorb(i, beta)
                sj = spinorb(j, beta)
                lines.append(f"{val:>28.16e} {si:4d} {sj:4d} {0:4d} {0:4d}")

    emit_one_body(h1_a, beta=False)
    emit_one_body(h1_b, beta=True)

    # Core energy record (i=j=k=l=0).
    lines.append(f"{float(ecore):>28.16e} {0:4d} {0:4d} {0:4d} {0:4d}")

    Path(path).write_text("\n".join(lines) + "\n")


def _spatial_and_spin(spinorb_1based: int) -> tuple[int, bool]:
    """1-based interleaved spin-orbital index -> (0-based spatial orbital, is_beta)."""
    idx0 = spinorb_1based - 1
    return idx0 // 2, bool(idx0 % 2)


def read_uhf_fcidump(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int], float]:
    """Parse an interleaved spin-orbital FCIDUMP into dense tensors.

    Returns ``(h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, (na, nb), ecore)`` with chemist-notation
    ``(norb,)*4`` two-body blocks. All permutational symmetries are filled in with the same rules
    as the C++ ``_UHF SetupIntegrals`` reader (8-fold for aa/bb, 4-fold + ``bb|aa`` transpose for
    the mixed block). No SCF is run: the numbers are exactly those in the file.

    Raises ``ValueError`` when the file looks like a spatial-orbital (restricted) FCIDUMP, i.e. no
    record index exceeds ``NORB`` -- such a file must be converted with ``rhf2uhf`` first.
    """
    lines = Path(path).read_text().splitlines()

    end = next(
        (n for n, line in enumerate(lines) if line.strip().upper() in ("&END", "/")), None
    )
    if end is None:
        raise ValueError(f"{path}: no '&END' terminating the FCIDUMP header.")
    fields = {k.upper(): int(v) for k, v in _HEADER_INT.findall(" ".join(lines[:end]))}
    try:
        norb = fields["NORB"]
        nelec_total = fields["NELEC"]
    except KeyError as exc:
        raise ValueError(f"{path}: FCIDUMP header lacks {exc.args[0]}.") from None
    ms2 = fields.get("MS2", 0)
    na = (nelec_total + ms2) // 2
    nb = (nelec_total - ms2) // 2

    records = []
    for line in lines[end + 1 :]:
        parts = line.split()
        if not parts:
            continue
        val = float(parts[0].replace("D", "E").replace("d", "e"))
        records.append((val, *(int(x) for x in parts[1:5])))
    if not records:
        raise ValueError(f"{path}: FCIDUMP has no integral records.")

    max_index = max(max(r[1:]) for r in records)
    if max_index <= norb:
        raise ValueError(
            f"{path}: max orbital index {max_index} <= NORB={norb}, so this looks like a "
            "spatial-orbital (restricted) FCIDUMP, not the interleaved spin-orbital format. "
            "Convert it with rhf2uhf (or rhf2bsuhf) first."
        )
    if max_index > 2 * norb:
        raise ValueError(
            f"{path}: orbital index {max_index} exceeds 2*NORB={2 * norb}; not a valid "
            "interleaved spin-orbital FCIDUMP."
        )

    h1_a = np.zeros((norb, norb))
    h1_b = np.zeros((norb, norb))
    h2_aa = np.zeros((norb,) * 4)
    h2_ab = np.zeros((norb,) * 4)
    h2_bb = np.zeros((norb,) * 4)
    ecore = 0.0

    for val, i, j, k, l in records:
        if i == j == k == l == 0:
            ecore = val
            continue

        if k == 0 and l == 0:
            pi, beta_i = _spatial_and_spin(i)
            pj, beta_j = _spatial_and_spin(j)
            if beta_i != beta_j:
                raise ValueError(
                    f"{path}: one-body record ({i},{j}) mixes spins -- not a valid "
                    "interleaved spin-orbital FCIDUMP."
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
                f"{path}: two-body record ({i},{j},{k},{l}) mixes spins within a bra/ket pair "
                "-- not a valid interleaved spin-orbital FCIDUMP."
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


def aufbau_reference(
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    nelec: tuple[int, int],
    ecore: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Energy and Fock matrices of the aufbau determinant (lowest ``na``/``nb`` orbitals by index).

    Returns ``(E_ref, fock_a, fock_b)``; ``E_ref`` includes ``ecore``. With the file's orbitals
    as the MO basis this is exactly the UHF energy of the reference the converters emit.
    """
    na, nb = nelec
    oa, ob = slice(0, na), slice(0, nb)
    j_a = np.einsum("pqii->pq", h2_aa[:, :, oa, oa]) + np.einsum("pqii->pq", h2_ab[:, :, ob, ob])
    j_b = np.einsum("pqii->pq", h2_bb[:, :, ob, ob]) + np.einsum("iipq->pq", h2_ab[oa, oa, :, :])
    k_a = np.einsum("piiq->pq", h2_aa[:, oa, oa, :])
    k_b = np.einsum("piiq->pq", h2_bb[:, ob, ob, :])
    fock_a = h1_a + j_a - k_a
    fock_b = h1_b + j_b - k_b
    e_ref = 0.5 * (
        np.trace(h1_a[oa, oa] + fock_a[oa, oa]) + np.trace(h1_b[ob, ob] + fock_b[ob, ob])
    )
    return float(e_ref + ecore), fock_a, fock_b


def check_aufbau_order(fock_a: np.ndarray, fock_b: np.ndarray, nelec: tuple[int, int]) -> list[str]:
    """Check that occupied Fock diagonals lie below virtual ones (per spin).

    The workflow occupies the lowest ``na``/``nb`` orbitals BY INDEX; this is the reference only if
    the file's orbitals are energy ordered. Returns a list of human-readable problems (empty if OK).
    """
    problems = []
    for label, fock, n in (("alpha", fock_a, nelec[0]), ("beta", fock_b, nelec[1])):
        diag = np.diag(fock)
        if 0 < n < len(diag) and diag[:n].max() >= diag[n:].min():
            problems.append(
                f"{label}: max occupied Fock diagonal {diag[:n].max():.6f} >= min virtual "
                f"{diag[n:].min():.6f} (orbitals are not energy ordered around the gap)"
            )
    return problems


def warn_if_not_aufbau(fock_a: np.ndarray, fock_b: np.ndarray, nelec: tuple[int, int], source: str = "") -> None:
    for problem in check_aufbau_order(fock_a, fock_b, nelec):
        warnings.warn(
            f"{source}: {problem}; the index-aufbau determinant may not be the intended reference.",
            RuntimeWarning,
            stacklevel=2,
        )
