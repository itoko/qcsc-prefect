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


def _converge_broken_symmetry_uhf(mf, max_follow: int = 5):
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
    """
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
        mf = _converge_broken_symmetry_uhf(mf)
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
