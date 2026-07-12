"""Molecule geometry definition for quantum algorithms."""

import io
import os
import warnings
from typing import Annotated

import numpy as np
from prefect import get_run_logger, task
from pydantic import BaseModel
from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pyscf import ao2mo, cc, gto, scf, tools

# Pydantic Types
NpStrict1DArrayF64 = Annotated[
    np.ndarray[tuple[int,], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=1, strict_data_typing=True),
]

NpStrict2DArrayF64 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=2, strict_data_typing=True),
]

NpStrict4DArrayF64 = Annotated[
    np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=4, strict_data_typing=True),
]


class ElectronicProperties(BaseModel):
    """Intermediate data representing the electronic properties.

    For restricted (RHF) calculations only the spin-summed ``*_tensor`` and ``t2`` fields are
    populated and ``unrestricted`` is ``False``. For unrestricted (UHF) calculations the alpha
    blocks live in the base fields (``one_body_tensor`` = h1 alpha, ``two_body_tensor`` = (aa|aa),
    ``t2`` = t2aa) and the beta / mixed blocks live in the optional ``*_b`` / ``*_ab`` / ``*_bb``
    fields; ``unrestricted`` is ``True``.
    """

    one_body_tensor: NpStrict2DArrayF64
    two_body_tensor: NpStrict4DArrayF64
    t2: NpStrict4DArrayF64
    initial_occupancy: tuple[NpStrict1DArrayF64, NpStrict1DArrayF64]
    nuclear_repulsion_energy: float
    num_orbitals: int
    num_electrons: tuple[int, int]
    open_shell: bool
    spin_sq: float

    # Whether this is an unrestricted (UHF) calculation. When True, the *_b / *_ab / *_bb
    # fields below are populated. RHF results leave them None and behave exactly as before.
    unrestricted: bool = False

    # Beta one-body integrals (h1 beta); alpha h1 stays in one_body_tensor.
    one_body_tensor_b: NpStrict2DArrayF64 | None = None
    # Mixed (aa|bb) and beta-beta (bb|bb) two-body integrals; (aa|aa) stays in two_body_tensor.
    two_body_tensor_ab: NpStrict4DArrayF64 | None = None
    two_body_tensor_bb: NpStrict4DArrayF64 | None = None
    # UCCSD t2 mixed (t2ab) and beta-beta (t2bb) amplitudes; t2aa stays in t2.
    t2_ab: NpStrict4DArrayF64 | None = None
    t2_bb: NpStrict4DArrayF64 | None = None


def _parse_af_groups() -> dict | None:
    """Parse the FE4S4_AF_GROUPS env var into an atom-localized AF-guess spec, or None.

    Format (JSON): {"fe1":[2,3,4,5,6], "fe2":[7,8,9,10,11], ..., "up":["fe1","fe3"],
    "down":["fe2","fe4"]}. Fragments listed in "up"/"down" are alpha/beta spin-polarized; all other
    fragments are treated as closed (doubly occupied). Enables the Noodleman broken-symmetry guess in
    _converge_broken_symmetry_uhf for Fe-S cubanes (~33 mHa lower reference than energy-ordered).

    Convenience: FE4S4_AF_GROUPS="fe4s4" expands to the standard [Fe4S4(SCH3)4]2- MO grouping
    (l1=0-1, fe1=2-6, fe2=7-11, s=12-23, fe3=24-28, fe4=29-33, l2=34-35) with the Singlet-I pairing
    (Fe1,Fe3 up / Fe2,Fe4 down), matching the reference that reaches -326.801 (<S^2>~7.6).
    """
    import json

    raw = os.environ.get("FE4S4_AF_GROUPS", "").strip()
    if not raw:
        return None
    if raw.lower() == "fe4s4":
        return {
            "l1": list(range(0, 2)),
            "fe1": list(range(2, 7)),
            "fe2": list(range(7, 12)),
            "s": list(range(12, 24)),
            "fe3": list(range(24, 29)),
            "fe4": list(range(29, 34)),
            "l2": list(range(34, 36)),
            "up": ["fe1", "fe3"],
            "down": ["fe2", "fe4"],
        }
    try:
        return json.loads(raw)
    except Exception:
        warnings.warn(f"FE4S4_AF_GROUPS could not be parsed as JSON: {raw!r}; ignoring.")
        return None


def _converge_broken_symmetry_uhf(mf, max_follow: int = 5, af_groups: dict | None = None):
    """Drive a UHF calculation to the genuine (possibly spin-broken) solution.

    For a closed-shell singlet, plain UHF started from the RHF density stays at the RHF solution
    (the RHF determinant is a stationary point of the UHF equations) -- so ``mf.to_uhf().kernel()``
    returns an *effectively restricted* reference (<S^2> = 0, identical energy). For strongly
    correlated systems (e.g. Fe-S clusters) the true UHF is lower in energy and spin-broken; seeding
    the LUCJ ansatz / UCCSD amplitudes from the unbroken solution is a restricted reference in
    disguise and defeats the point of running UHF.

    This runs UHF, then iteratively performs an internal STABILITY analysis: if the current solution
    is unstable, it rebuilds the density from the lowest instability mode and re-converges (via
    Newton), following the instability down to a stable (broken-symmetry when appropriate) minimum.
    This is the standard PySCF recipe for reaching the true UHF solution.

    IMPORTANT (Fe-S cubanes): for [4Fe-4S] the internal stability analysis from a SYMMETRIC guess
    stays in the spin-pure (RHF) basin (<S^2>=0, -326.547) -- it does NOT find the antiferromagnetic
    broken-symmetry minimum, because breaking the spatial symmetry to localize the Fe spins requires
    a spin-IMBALANCED initial density, not just following an internal instability. So we FIRST try an
    antiferromagnetic guess (promote a few beta electrons across the HOMO-LUMO gap, per the Noodleman
    BS recipe), stability-follow that, and keep whichever solution (default vs AF-guess) is LOWER.
    For Fe4S4 the AF guess reaches -326.767 (<S^2>~7.6), 219 mHa below the spin-pure solution.
    """

    def _follow_stability(m, n=max_follow):
        stab = False
        for _ in range(n):
            new_mo = m.stability()[0]
            cur = m.mo_coeff
            if np.allclose(new_mo[0], cur[0]) and np.allclose(new_mo[1], cur[1]):
                stab = True
                break
            m = m.newton().run(m.make_rdm1(new_mo, m.mo_occ))
        return m, stab

    def _follow_stability_external(m, n=10):
        """Follow BOTH internal and EXTERNAL instabilities and keep the LOWEST iterate seen.

        The Noodleman AF sublattice minima (Singlet-I/II/III) are near-degenerate and separated by
        external (spin-symmetry-breaking) instabilities that plain internal following (_follow_
        stability) does not cross -- it declares "stable" at the first internal minimum (~-326.769
        for Fe4S4) and never reaches the lower external basin (~-326.801). Mirrors the collaborator
        recipe: m.stability(return_status=True, external=True), rebuild, repeat, track the minimum."""
        best = m
        for _ in range(n):
            res = m.stability(return_status=True, external=True)
            mo = res[0]
            stable_i, stable_e = res[-2], res[-1]
            try:
                # Pass the (external-)instability MO coefficients DIRECTLY to the Newton solver as
                # the initial guess (not a density rebuilt from them): newton().run(mo_coeff) follows
                # the external spin-symmetry-breaking mode into the lower AF basin, whereas
                # run(make_rdm1(mo)) stalls at the higher internal minimum. This is the collaborator
                # recipe and is what reaches -326.801 (vs -326.771 with the density rebuild).
                m = m.newton().run(mo if isinstance(mo, tuple) else m.mo_coeff)
            except Exception:
                break
            if m.e_tot < best.e_tot:
                best = m
            if stable_i and stable_e:
                break
        return best, True

    def _af_guess_uhf(base, base_mol, base_hcore, base_ovlp, base_eri, base_nuc):
        """A fresh plain scf.UHF (same integrals/mol) started from an AF spin-imbalanced density.
        Returns the stability-followed result, or None if it fails. Tries a couple of promotion
        widths. Built from a PLAIN scf.UHF on the mol (not base.__class__, which may be a
        newton-wrapped SecondOrderUHF lacking a simple mol constructor)."""
        try:
            na, nb = base_mol.nelec
        except Exception:
            return None
        n = base_mol.nao
        best = None
        for shift in (2, 1, 3):
            if nb - shift < 0 or nb + shift > n:
                continue
            m = scf.UHF(base_mol)
            # carry the FCIDUMP integral overrides if present
            if base_hcore is not None:
                m.get_hcore = lambda *a, **k: base_hcore
            if base_ovlp is not None:
                m.get_ovlp = lambda *a, **k: base_ovlp
            if base_eri is not None:
                m._eri = base_eri
            if base_nuc is not None:
                m.mol.energy_nuc = lambda *a: base_nuc
            m.max_cycle = 400
            m.conv_tol = 1e-9
            mo = np.eye(n)
            occ_a = list(range(na))
            occ_b = list(range(nb - shift)) + list(range(nb, nb + shift))
            dma = mo[:, occ_a] @ mo[:, occ_a].T
            dmb = mo[:, occ_b] @ mo[:, occ_b].T
            try:
                m.kernel(dm0=(dma, dmb))
                m, _ = _follow_stability(m)
                if best is None or m.e_tot < best.e_tot:
                    best = m
            except Exception:
                continue

        # Atom-localized (Noodleman) AF guess: when an orbital->fragment map is supplied, seed the
        # density so specific magnetic centers get spin-UP and the counter-set spin-DOWN (the true
        # antiferromagnetic sublattices), instead of a chemistry-blind energy-ordered promotion. For
        # Fe-S cubanes this lands in a ~33 mHa LOWER broken-symmetry basin than the energy-ordered
        # guess (Fe4S4: -326.801 vs -326.768) -- the reference orbitals span the Fe-3d correlation
        # far better, which is what SQD actually recovers on. `af_groups` maps fragment name ->
        # list[orbital idx]; keys "up"/"down" list which fragments are alpha/beta-polarized, closed
        # fragments (doubly occupied) are everything else. We stability-follow (external) and keep
        # the lowest across all guesses.
        if af_groups:
            up = set(af_groups.get("up", []))
            down = set(af_groups.get("down", []))
            frags = {k: v for k, v in af_groups.items() if k not in ("up", "down")}
            dm0 = np.zeros((2, n, n))
            for name, orbs in frags.items():
                for x in orbs:
                    if name in up:
                        dm0[0, x, x] = 1.0
                    elif name in down:
                        dm0[1, x, x] = 1.0
                    else:  # closed fragment: doubly occupied
                        dm0[:, x, x] = 1.0
            m = scf.UHF(base_mol)
            if base_hcore is not None:
                m.get_hcore = lambda *a, **k: base_hcore
            if base_ovlp is not None:
                m.get_ovlp = lambda *a, **k: base_ovlp
            if base_eri is not None:
                m._eri = base_eri
            if base_nuc is not None:
                m.mol.energy_nuc = lambda *a: base_nuc
            m.max_cycle = 400
            m.conv_tol = 1e-9
            try:
                m.kernel(dm0=(dm0[0], dm0[1]))
                # Follow external instabilities (10 rounds): the AF sublattice minima are
                # near-degenerate (Singlet-I/II/III) and only EXTERNAL following reaches the lowest
                # basin (~-326.801 for Fe4S4); internal-only following stalls at ~-326.769.
                m, _ = _follow_stability_external(m, n=10)
                if best is None or m.e_tot < best.e_tot:
                    best = m
            except Exception:
                pass
        return best

    # Capture the mol + any FCIDUMP integral overrides from the ORIGINAL mf before newton-wrapping,
    # so the AF-guess UHF can be rebuilt as a plain scf.UHF with the same Hamiltonian.
    _base_mol = mf.mol
    _base_hcore = mf.get_hcore() if callable(getattr(mf, "get_hcore", None)) else None
    _base_ovlp = mf.get_ovlp() if callable(getattr(mf, "get_ovlp", None)) else None
    _base_eri = getattr(mf, "_eri", None)
    try:
        _base_nuc = float(mf.mol.energy_nuc())
    except Exception:
        _base_nuc = None

    mf.kernel()
    stable = False
    for _ in range(max_follow):
        # stability() returns the (internal) stable MOs; if they differ from the current MOs the
        # solution was unstable and we follow them down.
        new_mo = mf.stability()[0]
        # new_mo is (mo_a, mo_b); detect whether it changed (instability found).
        cur = mf.mo_coeff
        changed = not (
            np.allclose(new_mo[0], cur[0]) and np.allclose(new_mo[1], cur[1])
        )
        if not changed:
            stable = True
            break  # stable -> genuine UHF minimum reached
        dm = mf.make_rdm1(new_mo, mf.mo_occ)
        mf = mf.newton().run(dm)

    # Also try the antiferromagnetic broken-symmetry guess and keep whichever solution is LOWER.
    # For closed-shell/RHF-basin cases the AF guess just returns to the same energy (no harm); for
    # Fe-S cubanes it finds the genuine BS minimum the stability-only path misses.
    af = _af_guess_uhf(mf, _base_mol, _base_hcore, _base_ovlp, _base_eri, _base_nuc)
    if af is not None and af.converged and af.e_tot < mf.e_tot - 1e-6:
        try:
            ss_af, _ = af.spin_square()
        except Exception:
            ss_af = float("nan")
        warnings.warn(
            f"Broken-symmetry UHF found via AF guess: E={af.e_tot:.6f} <S^2>={ss_af:.3f} is "
            f"{(mf.e_tot - af.e_tot) * 1000:.1f} mHa below the stability-only solution "
            f"({mf.e_tot:.6f}). Using the broken-symmetry reference.",
            RuntimeWarning,
            stacklevel=2,
        )
        return af

    # Robustness (audit P4): if the follow loop exhausted max_follow while still unstable, the
    # returned reference is NOT a genuine minimum. Warn loudly and report the final <S^2> so a
    # downstream seed built from a non-stable / spin-contaminated reference is visible, not silent.
    if not stable:
        try:
            ss, _mult = mf.spin_square()
        except Exception:
            ss = float("nan")
        warnings.warn(
            f"UHF stability following did not converge to a stable solution after "
            f"{max_follow} follows; returning the last iterate (<S^2>={ss:.4f}). The UHF "
            f"reference may be non-stationary or spin-contaminated -- t2/occupancy seeds built "
            f"from it should be treated with caution.",
            RuntimeWarning,
            stacklevel=2,
        )
    return mf


def _build_property(
    mf: scf.RHF,
    norb: int,
    spin_sq: float,
    buf: io.StringIO,
) -> ElectronicProperties:
    """Helper function to build ElectronicProperties from molecular calculation results.

    Args:
        mf: PySCF RHF object after calculation
        norb: Number of orbitals
        spin_sq: Target value for the total spin squared
        buf: StringIO buffer containing PySCF logs

    Returns:
        ElectronicProperties object
    """
    # Apply unitary transform with obtained MO coefficient.
    # The FCIdump file already gives you these integrals in MO basis,
    # but the HF calculation may give you correction to these integrals.
    # For example, phase may change with this unitary transform.
    # When started with Mole object, AO → MO transform is performed in here.
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(norb, norb, norb, norb)

    nuclear_repulsion_energy = mf.mol.energy_nuc()
    num_elec_a, num_elec_b = mf.mol.nelec

    # Run CCSD (amplitude seed for the LUCJ ansatz). Raise max_cycle + DIIS for hard active spaces.
    mycc = cc.CCSD(mf)
    mycc.max_cycle = int(os.environ.get("SEED_CCSD_MAX_CYCLE", "200"))
    mycc.diis_space = 12
    mycc.kernel()
    t2 = mycc.t2

    # Per-MO occupancies for configuration recovery. recover_configurations expects the mean
    # occupancy of MO p IN THE BASIS THE BITSTRINGS LIVE IN (the canonical MO basis the circuit and
    # integrals use), i.e. the DIAGONAL of the 1-RDM in that basis -- NOT the sorted eigenvalues
    # (natural-orbital occupation numbers in a different basis), which would misassign the bias to
    # the wrong orbitals. The RDM must come from the *correlated* CCSD (mycc), not the SCF object
    # (mf.make_rdm1() is idempotent -> integer 0/1 occupancies, which give recovery no fractional
    # signal and collapse the SQD subspace to bare Hartree-Fock).
    rdm1_ccsd = mycc.make_rdm1()
    occ_ccsd = np.diag(rdm1_ccsd) / 2.0  # spatial-orbital occ (0-2) -> per-spin (0-1)

    # Get PySCF logs dumped into in-memory buffer
    get_run_logger().info(buf.getvalue())

    return ElectronicProperties(
        one_body_tensor=h1,
        two_body_tensor=h2,
        t2=t2,
        initial_occupancy=(occ_ccsd, occ_ccsd),
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        num_orbitals=norb,
        num_electrons=(num_elec_a, num_elec_b),
        open_shell=num_elec_a != num_elec_b,
        spin_sq=spin_sq,
    )


def _build_property_uhf(
    mf: scf.UHF,
    norb: int,
    spin_sq: float,
    buf: io.StringIO,
) -> ElectronicProperties:
    """Build ElectronicProperties from an unrestricted (UHF) calculation.

    Mirrors :func:`_build_property` but keeps separate alpha/beta MO coefficients, three
    two-body integral blocks (aa, ab, bb), the UCCSD t2 tuple (t2aa, t2ab, t2bb), and per-spin
    occupancies. The alpha blocks go into the base ElectronicProperties fields; beta / mixed
    blocks go into the optional *_b / *_ab / *_bb fields.

    Args:
        mf: PySCF UHF object after calculation.
        norb: Number of (spatial) orbitals.
        spin_sq: Target value for the total spin squared.
        buf: StringIO buffer containing PySCF logs.

    Returns:
        ElectronicProperties object with ``unrestricted=True``.
    """
    mo_a, mo_b = mf.mo_coeff
    hcore = mf.get_hcore()

    # One-body integrals in each spin's MO basis.
    h1_a = mo_a.T @ hcore @ mo_a
    h1_b = mo_b.T @ hcore @ mo_b

    # Two-body integrals: alpha-alpha, beta-beta, and the mixed alpha-beta block.
    eri = mf._eri
    h2_aa = ao2mo.full(eri, mo_a, compact=False).reshape(norb, norb, norb, norb)
    h2_bb = ao2mo.full(eri, mo_b, compact=False).reshape(norb, norb, norb, norb)
    h2_ab = ao2mo.general(eri, (mo_a, mo_a, mo_b, mo_b), compact=False).reshape(
        norb, norb, norb, norb
    )

    nuclear_repulsion_energy = mf.mol.energy_nuc()
    num_elec_a, num_elec_b = mf.mol.nelec

    # Run UCCSD; t2 is the tuple (t2aa, t2ab, t2bb). Strongly-correlated Fe-S active spaces do not
    # converge in the default 50 cycles; raise max_cycle + DIIS so the amplitude seed for the LUCJ
    # ansatz is well-formed (override via SEED_CCSD_MAX_CYCLE).
    mycc = cc.UCCSD(mf)
    mycc.max_cycle = int(os.environ.get("SEED_CCSD_MAX_CYCLE", "200"))
    mycc.diis_space = 12
    mycc.kernel()
    t2_aa, t2_ab, t2_bb = mycc.t2

    # Per-spin natural-orbital occupancies from the *correlated* UCCSD 1-RDM (in MO basis).
    # These must be fractional: configuration recovery downstream uses them to bias sampled
    # bitstrings toward the physically important (near-HF) configurations. The UHF *SCF* RDM is
    # (often) idempotent -> integer 0/1 occupancies, which give recovery no signal and collapse
    # the SQD subspace to bare Hartree-Fock. UCCSD make_rdm1() returns (dm_a, dm_b) in MO basis.
    # Use the DIAGONAL (per-canonical-MO occupancy) of the correlated UCCSD 1-RDM, not the sorted
    # eigenvalues. The bitstrings are measured in the canonical MO basis (PrepareHartreeFockJW +
    # integrals from mf.mo_coeff), so recover_configurations needs occ of MO p in THAT basis; the
    # eigenvalues are natural-orbital occupation numbers in a rotated basis and, once sorted, would
    # misassign the bias to the wrong orbitals (the error lands on the active, partially-occupied
    # orbitals and grows with system size).
    dm_cc_a, dm_cc_b = mycc.make_rdm1()
    occ_a = np.diag(dm_cc_a)
    occ_b = np.diag(dm_cc_b)

    # Get PySCF logs dumped into in-memory buffer
    get_run_logger().info(buf.getvalue())

    return ElectronicProperties(
        one_body_tensor=h1_a,
        two_body_tensor=h2_aa,
        t2=t2_aa,
        one_body_tensor_b=h1_b,
        two_body_tensor_ab=h2_ab,
        two_body_tensor_bb=h2_bb,
        t2_ab=t2_ab,
        t2_bb=t2_bb,
        initial_occupancy=(occ_a, occ_b),
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        num_orbitals=norb,
        num_electrons=(num_elec_a, num_elec_b),
        open_shell=num_elec_a != num_elec_b,
        unrestricted=True,
        spin_sq=spin_sq,
    )


@task(
    persist_result=True,
    result_serializer="compressed/json",
    name="compute_molecular_integrals",
)
def compute_molecular_integrals_from_geometry(
    atom: str,
    basis: str = "6-31g",
    symmetry: str | bool = False,
    spin_sq: float = 0.0,
    unrestricted: bool = False,
    spin: int = 0,
) -> ElectronicProperties:
    """Precompute molecular orbital property from geometry with classical methods.

    Args:
        atom: Definition for molecule structure.
        basis: Name of basis set.
        symmetry: Whether to use symmetry, otherwise string of point group name.
        spin_sq: Target value for the total spin squared for the ground state.
        unrestricted: Run an unrestricted (UHF) calculation instead of restricted (RHF).
        spin: Number of unpaired electrons (nelec_alpha - nelec_beta). Only meaningful for
            open-shell systems; passed to PySCF as ``mol.spin``.

    Returns:
        ElectronicProperties object containing molecular integrals and properties.
    """
    # PySCF doesn't use the standard Python logging and Prefect cannot capture it.
    # The logs are directly written in the stdout or in a file.
    # To forward the logs to the Prefect logging sytem,
    # we set an in-memory buffer to the PySCF logging system and read from there.
    buf = io.StringIO()

    mol = gto.Mole()
    mol.build(
        atom=atom,
        basis=basis,
        symmetry=symmetry,
        spin=spin,
    )
    mol.stdout = buf
    mol.verbose = 4

    if unrestricted:
        mf = scf.UHF(mol)
        # Follow any spin instability to the true (broken-symmetry) UHF minimum (see
        # _converge_broken_symmetry_uhf) so the reference is genuinely unrestricted for singlets.
        mf = _converge_broken_symmetry_uhf(mf)
        norb = mf.mo_coeff[0].shape[1]
        return _build_property_uhf(mf, norb, spin_sq, buf)

    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]

    return _build_property(mf, norb, spin_sq, buf)


@task(
    persist_result=True,
    result_serializer="compressed/json",
    name="compute_molecular_integrals",
)
def compute_molecular_integrals_from_fcidump(
    fcidump_file: str,
    spin_sq: float = 0.0,
    unrestricted: bool = False,
) -> ElectronicProperties:
    """Precompute molecular orbital property from FCIDump file with classical methods.

    Args:
        fcidump_file: Location of FCIDump file storing 1-electron and 2-electron integrals.
        spin_sq: Target value for the total spin squared for the ground state.
        unrestricted: Run an unrestricted (UHF) calculation instead of restricted (RHF). The
            number of unpaired electrons is taken from the FCIDUMP ``MS2`` header.

    Returns:
        ElectronicProperties object containing molecular integrals and properties.
    """
    # PySCF doesn't use the standard Python logging and Prefect cannot capture it.
    # The logs are directly written in the stdout or in a file.
    # To forward the logs to the Prefect logging sytem,
    # we set an in-memory buffer to the PySCF logging system and read from there.
    buf = io.StringIO()

    data = tools.fcidump.read(fcidump_file)
    norb = data["NORB"]

    mf = tools.fcidump.to_scf(fcidump_file)
    mf.mol.verbose = 4
    mf.mol.stdout = buf

    if unrestricted:
        # pyscf 2.13 has no fcidump.to_uhf_scf; convert the RHF shell (which already carries the
        # FCIDUMP integral overrides and mol.spin = MS2) into a UHF object and run it. UHF init
        # guess builds two spin densities from nelec=(na, nb); no manual dm0 needed.
        nuc = mf.mol.energy_nuc()
        # Audit P3: tools.fcidump.to_scf sets mol.symmetry=True with a *guessed* point group when
        # the FCIDUMP carries ORBSYM (both the Fe2S2 and Fe4S4 dumps do: ORBSYM=1). The RHF branch
        # defeats this with mf.symmetry=False (below); the UHF branch previously did NOT, so UHF ran
        # its kernel + stability analysis under a possibly-wrong group -> symmetry-constrained /
        # misconverged reference, and stability following couldn't break the relevant symmetry.
        # Disable symmetry on the mol and REBUILD it before converting, so to_uhf() produces a plain
        # (non-symmetry-adapted) UHF whose kernel()/stability() don't require a point group. Setting
        # .symmetry=False on an already-symmetry-adapted object is insufficient (SymAdaptedUHF.build
        # still raises "mol.symmetry not enabled"); the rebuild is what actually clears it.
        # to_scf builds a SymAdaptedRHF, and mf.to_uhf() preserves that symmetry-adapted class --
        # whose build() still demands mol.symmetry even after we clear the flag. So disable symmetry
        # on the mol, rebuild it, and construct a PLAIN scf.UHF on it (not mf.to_uhf()); carry the
        # FCIDUMP integral overrides (get_hcore/_eri) + nuclear energy across by hand.
        base_mol = mf.mol
        if getattr(base_mol, "symmetry", False):
            base_mol.symmetry = False
            base_mol.build(dump_input=False, parse_arg=False)
        hcore = mf.get_hcore()
        ovlp = mf.get_ovlp()
        eri = mf._eri
        mf = scf.UHF(base_mol)
        mf.get_hcore = lambda *args, **kwargs: hcore
        mf.get_ovlp = lambda *args, **kwargs: ovlp
        mf._eri = eri
        mf.mol.energy_nuc = lambda *args: nuc  # preserved across the conversion
        mf.symmetry = False
        # Follow any spin instability to the genuine (broken-symmetry) UHF minimum -- otherwise a
        # singlet stays at the RHF solution and the "UHF" reference/UCCSD/t2 seed is restricted.
        # For Fe-S cubanes, an atom-localized (Noodleman) AF guess -- supplied via FE4S4_AF_GROUPS --
        # reaches a ~33 mHa lower BS basin than the energy-ordered guess (better Fe-3d orbitals for
        # SQD to recover on). See _parse_af_groups; unset -> plain energy-ordered AF guess (default).
        mf = _converge_broken_symmetry_uhf(mf, af_groups=_parse_af_groups())
        return _build_property_uhf(mf, norb, spin_sq, buf)

    # Run HF calculation with Newton method.
    # HF convergence is important, as we assume
    # the FCIdump file is created with a converged result.
    mf = scf.newton(mf)
    mf.symmetry = False
    dm0 = np.zeros((norb, norb))
    for i in range(mf.mol.nelectron // 2):
        dm0[i, i] = 2.0
    mf.kernel(dm0=dm0)

    return _build_property(mf, norb, spin_sq, buf)
