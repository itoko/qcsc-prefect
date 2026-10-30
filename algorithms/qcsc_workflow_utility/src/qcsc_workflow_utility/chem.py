"""Molecule geometry definition for quantum algorithms."""

import io
import os
import warnings
from typing import Annotated

import numpy as np

# Workaround: PySCF DIIS references numpy.linalg.linalg.LinAlgError which was removed
# in numpy 2.x (now numpy.linalg._linalg). Patch it so DIIS exception handling works.
if not hasattr(np.linalg, "linalg"):
    import types as _types
    np.linalg.linalg = _types.SimpleNamespace(LinAlgError=np.linalg.LinAlgError)
from prefect import get_run_logger, task
from pydantic import BaseModel
from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pyscf import ao2mo, cc, gto, scf, tools

from .uhf_fcidump import check_aufbau_order, read_uhf_fcidump

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
    # Overlap between the current alpha and beta orbitals, S_ab[p, q] = <phi_p^alpha | phi_q^beta>,
    # needed for <S^2> (spin_square_from_rdms). UHF only. None means the alpha and beta orbitals
    # coincide at load time (identity; e.g. rhf2uhf input); orbital rotations update it as
    # Ua^T S_ab Ub (rotate_electronic_properties).
    ab_overlap: NpStrict2DArrayF64 | None = None


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


def _build_property_from_uhf_integrals(
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    ecore: float,
    spin_sq: float,
    buf: io.StringIO,
) -> tuple[ElectronicProperties, float, float]:
    """Build unrestricted ElectronicProperties from MO-basis spin-block integrals alone.

    The orbitals are taken as given (they are built OUTSIDE the workflow, e.g. by rhf2uhf /
    rhf2bsuhf): the reference is the index-aufbau determinant (lowest ``na`` alpha / ``nb`` beta
    orbitals), represented by a mock UHF (mo_coeff = I, see :func:`_build_mock_uhf`). UCCSD is run
    cold on that reference to seed the LUCJ ansatz (t2) and configuration recovery (diagonal of
    the correlated 1-RDM). The alpha blocks go into the base ElectronicProperties fields; beta /
    mixed blocks go into the optional *_b / *_ab / *_bb fields. ``OO_INIT=1`` then applies one
    initial orbital-optimization step (:func:`_apply_initial_oo`).

    Returns ``(props, E_ref, E_UCCSD, UCCSD converged)`` (total energies, including ``ecore``).
    """
    num_elec_a, num_elec_b = nelec
    props = ElectronicProperties(
        one_body_tensor=np.ascontiguousarray(h1_a, dtype=np.float64),
        two_body_tensor=np.ascontiguousarray(h2_aa, dtype=np.float64),
        t2=np.zeros((0, 0, 0, 0)),
        one_body_tensor_b=np.ascontiguousarray(h1_b, dtype=np.float64),
        two_body_tensor_ab=np.ascontiguousarray(h2_ab, dtype=np.float64),
        two_body_tensor_bb=np.ascontiguousarray(h2_bb, dtype=np.float64),
        initial_occupancy=(np.zeros(norb), np.zeros(norb)),
        nuclear_repulsion_energy=float(ecore),
        num_orbitals=norb,
        num_electrons=(num_elec_a, num_elec_b),
        open_shell=num_elec_a != num_elec_b,
        unrestricted=True,
        spin_sq=spin_sq,
    )

    mf, make_eris = _build_mock_uhf(props, verbose=4, stdout=buf)
    e_ref = float(mf.e_tot)
    for problem in check_aufbau_order(np.diag(mf.mo_energy[0]), np.diag(mf.mo_energy[1]), nelec):
        warnings.warn(
            f"{problem}; the index-aufbau determinant may not be the intended reference.",
            RuntimeWarning,
            stacklevel=2,
        )

    # SEED_CC_FILE (opt-in): reuse a saved UCCSD seed instead of re-running UCCSD. The UCCSD here
    # often stops at max_cycle unconverged, and from there its iterates depend on the BLAS thread
    # reduction order, so the t2 seed (hence the LUCJ ansatz and the trial-0 subspace) differs
    # run to run even for the same input and seed. One file shared by all runs pins it.
    seed_cc_file = os.environ.get("SEED_CC_FILE", "").strip()
    fingerprint = _integral_fingerprint(h1_a, h1_b, h2_aa, h2_ab, h2_bb, nelec)
    if seed_cc_file and os.path.exists(seed_cc_file):
        if os.environ.get("OO_INIT", "0") == "1":
            raise ValueError("SEED_CC_FILE cannot be combined with OO_INIT=1 (needs the live UCCSD).")
        seed = load_cc_seed(seed_cc_file, fingerprint)
        buf.write(
            f"[seed-cc] loaded UCCSD seed from {os.path.abspath(seed_cc_file)} "
            f"(E_UCCSD={seed['e_cc']:.10f}, converged={seed['converged']}, sha1={seed['sha1']})\n"
        )
        get_run_logger().info(buf.getvalue())
        props = props.model_copy(
            update={
                "t2": seed["t2_aa"],
                "t2_ab": seed["t2_ab"],
                "t2_bb": seed["t2_bb"],
                "initial_occupancy": (seed["occ_a"], seed["occ_b"]),
            }
        )
        return props, e_ref, seed["e_cc"], seed["converged"]

    # Run UCCSD; t2 is the tuple (t2aa, t2ab, t2bb). Strongly-correlated Fe-S active spaces do not
    # converge in the default 50 cycles; raise max_cycle + DIIS so the amplitude seed for the LUCJ
    # ansatz is well-formed (override via SEED_CCSD_MAX_CYCLE).
    mycc = cc.UCCSD(mf)
    mycc.max_cycle = int(os.environ.get("SEED_CCSD_MAX_CYCLE", "200"))
    mycc.diis_space = 12
    eris = make_eris(mycc)
    # The lambda equations (make_rdm1/make_rdm2) call mycc.ao2mo() when no eris are passed, which
    # on the mock rebuilds ALL spin blocks from mf._eri (= h2_aa only). Pin the correct eris.
    mycc.ao2mo = lambda *args, **kwargs: eris
    mycc.kernel(eris=eris)
    t2_aa, t2_ab, t2_bb = mycc.t2

    # Per-spin occupancies from the *correlated* UCCSD 1-RDM (in MO basis). These must be
    # fractional: configuration recovery downstream uses them to bias sampled bitstrings toward the
    # physically important (near-HF) configurations; the SCF RDM is idempotent (integer 0/1) and
    # gives recovery no signal. Use the DIAGONAL (per-MO occupancy in the basis the bitstrings are
    # measured in), not the sorted natural-orbital eigenvalues, which live in a rotated basis.
    dm_cc_a, dm_cc_b = mycc.make_rdm1()
    occ_a = np.diag(dm_cc_a)
    occ_b = np.diag(dm_cc_b)

    e_cc, cc_conv = float(mycc.e_tot), bool(mycc.converged)
    if seed_cc_file:
        # First writer wins; every run then uses the FILE's seed, so array elements that started
        # together (each running its own UCCSD) still end up on one identical seed.
        written = save_cc_seed(
            seed_cc_file, fingerprint, t2_aa, t2_ab, t2_bb, occ_a, occ_b, e_cc, cc_conv
        )
        seed = load_cc_seed(seed_cc_file, fingerprint)
        t2_aa, t2_ab, t2_bb = seed["t2_aa"], seed["t2_ab"], seed["t2_bb"]
        occ_a, occ_b = seed["occ_a"], seed["occ_b"]
        buf.write(
            f"[seed-cc] {'wrote' if written else 'another run wrote'} UCCSD seed "
            f"{os.path.abspath(seed_cc_file)}; using it (E_UCCSD={seed['e_cc']:.10f} vs this run "
            f"{e_cc:.10f}, sha1={seed['sha1']})\n"
        )
        e_cc, cc_conv = seed["e_cc"], seed["converged"]

    # Get PySCF logs dumped into in-memory buffer
    get_run_logger().info(buf.getvalue())

    props = props.model_copy(
        update={
            "t2": t2_aa,
            "t2_ab": t2_ab,
            "t2_bb": t2_bb,
            "initial_occupancy": (occ_a, occ_b),
        }
    )

    # Opt-in initial orbital optimization: reference -> UCCSD -> OO -> UCCSD refresh.
    # Default off (OO_INIT unset/0).
    if os.environ.get("OO_INIT", "0") == "1":
        if seed_cc_file:
            raise ValueError("SEED_CC_FILE cannot be combined with OO_INIT=1 (needs the live UCCSD).")
        props = _apply_initial_oo(props, mycc, dm_cc_a, dm_cc_b)

    return props, e_ref, e_cc, cc_conv


def _integral_fingerprint(h1_a, h1_b, h2_aa, h2_ab, h2_bb, nelec) -> str:
    """sha1 of the integrals and electron counts a saved UCCSD seed belongs to."""
    import hashlib

    h = hashlib.sha1()
    for a in (h1_a, h1_b, h2_aa, h2_ab, h2_bb):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    h.update(np.asarray(nelec, dtype=np.int64).tobytes())
    return h.hexdigest()


def save_cc_seed(path, fingerprint, t2_aa, t2_ab, t2_bb, occ_a, occ_b, e_cc, converged) -> bool:
    """Write a UCCSD seed file unless one already exists. Returns True if this call wrote it.

    Written to a temp file and linked into place, so concurrent writers never leave a partial
    file and the first writer wins (later ones keep the existing file)."""
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(
            fh, fingerprint=np.asarray(fingerprint), t2_aa=t2_aa, t2_ab=t2_ab, t2_bb=t2_bb,
            occ_a=occ_a, occ_b=occ_b, e_cc=np.float64(e_cc), converged=np.bool_(converged),
        )
    try:
        os.link(tmp, path)  # fails if path exists -> first writer wins
        return True
    except FileExistsError:
        return False
    finally:
        os.unlink(tmp)


def load_cc_seed(path, fingerprint: str) -> dict:
    """Read a UCCSD seed file and check it was made for these integrals."""
    import hashlib

    with open(path, "rb") as fh:
        raw = fh.read()
    with np.load(io.BytesIO(raw)) as z:
        if str(z["fingerprint"]) != fingerprint:
            raise ValueError(
                f"SEED_CC_FILE {path} was made for different integrals/electron counts "
                f"(fingerprint {str(z['fingerprint'])[:12]} != {fingerprint[:12]})."
            )
        out = {k: np.asarray(z[k], dtype=np.float64) for k in ("t2_aa", "t2_ab", "t2_bb", "occ_a", "occ_b")}
        out["e_cc"] = float(z["e_cc"])
        out["converged"] = bool(z["converged"])
    out["sha1"] = hashlib.sha1(raw).hexdigest()[:12]
    return out


def _apply_initial_oo(
    props: ElectronicProperties,
    mycc: "cc.uccsd.UCCSD",
    dm_cc_a: np.ndarray,
    dm_cc_b: np.ndarray,
) -> ElectronicProperties:
    """Insert one orbital-optimization step between the initial UHF-reference UCCSD and the t2 seed.

    Opt-in via ``OO_INIT=1`` (UHF path only). Flow:

        reference (FCIDUMP orbitals) -> UCCSD (already run; 1-/2-RDM in hand)
               -> OO (rotate orbitals with the fixed UCCSD RDM, two-step MCSCF)
               -> UCCSD refresh in the rotated basis (fresh t2 / occupancy seed).

    OO needs a *correlated* RDM: a converged SCF density already satisfies the generalized
    Brillouin condition (zero orbital gradient), so an SCF-RDM-driven OO cannot move. UCCSD
    supplies that correlated RDM without needing a CI subspace.

    UCCSD is non-variational and, for large active spaces (e.g. Fe4S4 36o/54e), frequently
    unconverged, so the rotation is deliberately conservative: a single fixed-RDM OO step with
    a small trust radius, guarded by an energy check that *discards* the rotation unless the
    refreshed UCCSD energy is at or below the pre-OO UCCSD energy (safe fallback to the
    un-rotated reference).

    Thresholds are aligned with the DE-loop OO (see sbd/main.py) through one shared knob:
    ``OO_GRAD_TOL`` (default 1e-5) drives the inner L-BFGS-B pgtol on EVERY OO path (this initial
    OO, the DE-loop fixed-RDM OO, and the self-consistent resolve OO) AND the outer Brillouin-freeze
    / SC-macro convergence. ``OO_TRUST`` (0.5) and ``OO_MAXITER`` (300) likewise share env names
    and defaults with the DE-loop OO.
    """
    from qcsc_workflow_utility.orbital_opt import (
        optimize_orbitals,
        rotate_electronic_properties,
    )

    logger = get_run_logger()
    e_cc_before = float(mycc.e_tot)

    # UCCSD spin-block 2-RDM in PySCF "pqrs" (chemist) storage. make_rdm1() already matches the
    # physicist convention optimize_orbitals expects (dm1[p,q] = <q^dag p>), so no transpose.
    d2aa, d2ab, d2bb = mycc.make_rdm2()

    gtol = float(os.environ.get("OO_GRAD_TOL", "1e-5"))
    trust = float(os.environ.get("OO_TRUST", "0.5"))
    maxiter = int(os.environ.get("OO_MAXITER", "300"))

    # Inner L-BFGS-B convergence (scipy pgtol) is driven by the shared OO_GRAD_TOL knob (default
    # 1e-5), identical to the DE-loop and SC OO paths.
    Ua, Ub, e_oo, grad_norm = optimize_orbitals(
        props,
        np.asarray(dm_cc_a),
        np.asarray(dm_cc_b),
        np.asarray(d2aa),
        np.asarray(d2ab),
        np.asarray(d2bb),
        rdm2_notation="pqrs",  # UCCSD make_rdm2 storage (NOT the native solver's "prqs")
        gtol=gtol,
        trust_radius=trust,
        maxiter=maxiter,
    )

    props_rot = rotate_electronic_properties(props, Ua, Ub)
    props_new, e_cc_after = refresh_cc_seed(props_rot, return_energy=True)

    if (
        props_new is not None
        and e_cc_after is not None
        and np.isfinite(e_cc_after)
        and e_cc_after <= e_cc_before + 1e-6
    ):
        logger.info(
            "OO_INIT accepted: E(UCCSD before OO)=%.10f  E_oo(fixed-RDM, total)=%.10f  "
            "|grad|=%.3e  E(UCCSD after OO)=%.10f  dE=%+.3e Ha  (trust=%.3g, gtol=%.1e).",
            e_cc_before, e_oo, grad_norm, e_cc_after, e_cc_after - e_cc_before, trust, gtol,
        )
        return props_new

    logger.warning(
        "OO_INIT rejected (no energy gain / NaN): E(UCCSD before OO)=%.10f  E_oo=%.10f  "
        "|grad|=%.3e  E(UCCSD after OO)=%s  -> keeping pre-OO reference/UCCSD.",
        e_cc_before, e_oo, grad_norm,
        f"{e_cc_after:.10f}" if e_cc_after is not None else "None",
    )
    return props


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
        mf = scf.UHF(mol).run()
        # Follow internal instabilities to a stable UHF minimum. Atom-localized broken-symmetry
        # guesses are out of scope here (orbitals for production runs are built outside the
        # workflow, e.g. by rhf2bsuhf, and fed in as a spin-orbital FCIDUMP).
        for _ in range(5):
            new_mo = mf.stability()[0]
            if np.allclose(new_mo[0], mf.mo_coeff[0]) and np.allclose(new_mo[1], mf.mo_coeff[1]):
                break
            mf = mf.newton().run(mf.make_rdm1(new_mo, mf.mo_occ))
        mo_a, mo_b = mf.mo_coeff
        norb = mo_a.shape[1]
        hcore = mf.get_hcore()
        h2 = lambda c1, c2: ao2mo.general(mol, (c1, c1, c2, c2), compact=False).reshape((norb,) * 4)
        props, *_ = _build_property_from_uhf_integrals(
            mo_a.T @ hcore @ mo_a,
            mo_b.T @ hcore @ mo_b,
            h2(mo_a, mo_a),
            h2(mo_a, mo_b),
            h2(mo_b, mo_b),
            norb,
            mol.nelec,
            mol.energy_nuc(),
            spin_sq,
            buf,
        )
        return props

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
) -> ElectronicProperties:
    """Precompute molecular orbital property from FCIDump file with classical methods.

    Restricted (RHF) only. For unrestricted runs use
    :func:`compute_molecular_integrals_from_uhf_fcidump` with a spin-orbital FCIDUMP (convert a
    spatial-orbital file with rhf2uhf / rhf2bsuhf).

    Args:
        fcidump_file: Location of FCIDump file storing 1-electron and 2-electron integrals.
        spin_sq: Target value for the total spin squared for the ground state.

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


@task(
    persist_result=True,
    result_serializer="compressed/json",
    name="compute_molecular_integrals",
)
def compute_molecular_integrals_from_uhf_fcidump(
    fcidump_file: str,
    spin_sq: float = 0.0,
    ab_overlap_file: str | None = None,
) -> ElectronicProperties:
    """Load a spin-orbital FCIDUMP as the unrestricted reference -- no SCF, no orbital change.

    The file (format: :mod:`qcsc_workflow_utility.uhf_fcidump`, as written by rhf2uhf /
    rhf2bsuhf) already carries the MO-basis spin-block integrals of the chosen reference; the
    reference determinant is the lowest ``na`` alpha / ``nb`` beta orbitals by index. UCCSD (t2
    seed + initial occupancy) and the optional ``OO_INIT=1`` step are run on top of it from the
    integrals alone (mock UHF, see :func:`_build_mock_uhf`).

    Args:
        fcidump_file: Interleaved spin-orbital FCIDUMP (header NORB = spatial orbitals).
        spin_sq: Target value for the total spin squared for the ground state.
        ab_overlap_file: ``.npz`` with key ``ab_overlap`` (norb x norb), the overlap between the
            file's alpha and beta orbitals (rhf2bsuhf writes ``OUT.ab_overlap.npz``). Default: the
            sidecar ``<fcidump_file>.ab_overlap.npz`` if it exists, else identity (alpha and beta
            orbitals coincide, as for rhf2uhf output). Only used for <S^2>.

    Returns:
        ElectronicProperties with ``unrestricted=True``; ``nuclear_repulsion_energy`` is the
        file's core energy.
    """
    buf = io.StringIO()
    h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore = read_uhf_fcidump(fcidump_file)
    props, e_ref, e_cc, cc_conv = _build_property_from_uhf_integrals(
        h1_a, h1_b, h2_aa, h2_ab, h2_bb, norb, nelec, ecore, spin_sq, buf
    )
    if ab_overlap_file is None and os.path.exists(f"{fcidump_file}.ab_overlap.npz"):
        ab_overlap_file = f"{fcidump_file}.ab_overlap.npz"
    if ab_overlap_file is not None:
        s_ab = np.asarray(np.load(ab_overlap_file)["ab_overlap"], dtype=np.float64)
        if s_ab.shape != (norb, norb):
            raise ValueError(f"{ab_overlap_file}: ab_overlap shape {s_ab.shape} != ({norb}, {norb})")
        props = props.model_copy(update={"ab_overlap": s_ab})
        print(f"[uhf-input] alpha/beta orbital overlap from {os.path.abspath(ab_overlap_file)}", flush=True)
    # Reference line parsed by the analysis tooling (check_logs.py / notebooks).
    print(
        f"[uhf-input] loaded spin-orbital FCIDUMP {os.path.abspath(fcidump_file)} "
        f"(norb={norb}, nelec={tuple(int(n) for n in nelec)}, E_ref={e_ref:.10f}, "
        f"E_UCCSD={e_cc:.10f}, UCCSD_converged={cc_conv})",
        flush=True,
    )
    return props


def refresh_cc_seed(
    elec_props: ElectronicProperties,
    return_energy: bool = False,
):
    """Re-run CCSD/UCCSD on the current (rotated) integrals and return updated ElectronicProperties.

    This is the CC-refresh step for OO_TO_LUCJ=1: after orbital optimization rotates the
    Hamiltonian, solve CC in the new basis so that t2 and initial_occupancy match the rotated
    integrals. Uses the previous t2 amplitudes (from elec_props) as warm-start initial guess
    so that UCCSD converges faster in the rotated basis. Returns a new ElectronicProperties
    with updated t2/occupancy fields, or None if amplitudes contain NaN.

    If ``return_energy`` is True, returns ``(props_or_None, e_tot_or_None)`` instead: the
    refreshed UCCSD/CCSD total energy accompanies the properties (used by the OO_INIT energy
    guard). Default False preserves the original single-value return for existing callers.
    """
    import logging

    log = logging.getLogger(__name__)
    norb = elec_props.num_orbitals
    nelec_a, nelec_b = elec_props.num_electrons

    max_cycle = int(os.environ.get("SEED_CCSD_MAX_CYCLE", "200"))

    if elec_props.unrestricted:
        return _refresh_cc_seed_uhf(elec_props, log, max_cycle, return_energy)
    else:
        return _refresh_cc_seed_rhf(elec_props, log, max_cycle, return_energy)


def _refresh_cc_seed_rhf(
    elec_props: ElectronicProperties,
    log,
    max_cycle: int,
    return_energy: bool = False,
) -> ElectronicProperties | None:
    """CC refresh for RHF."""
    norb = elec_props.num_orbitals
    nelec_a, nelec_b = elec_props.num_electrons
    h1 = np.asarray(elec_props.one_body_tensor)
    h2 = np.asarray(elec_props.two_body_tensor)

    mol = gto.M()
    mol.nelectron = nelec_a + nelec_b
    mol.spin = 0
    mol.verbose = 0
    mol.incore_anyway = True

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a, **k: h1
    mf.get_ovlp = lambda *a, **k: np.eye(norb)
    mf._eri = ao2mo.restore(8, h2.reshape(norb, norb, norb, norb), norb)
    mf.mol.energy_nuc = lambda *a: elec_props.nuclear_repulsion_energy
    mf.mol.nao_nr = lambda *a: norb

    mo_coeff = np.eye(norb)
    mo_occ = np.zeros(norb)
    mo_occ[:nelec_a] = 2.0

    # Compute proper Fock matrix diagonal for orbital energies.
    # F_pq = h1_pq + sum_i [2*(pi|qi) - (pq|ii)] where i are occupied.
    nocc = nelec_a
    j_diag = 2.0 * np.einsum("piqi->pq", h2[:, :nocc, :, :nocc])
    k_diag = -np.einsum("pqii->pq", h2[:, :, :nocc, :nocc])
    fock = h1 + j_diag + k_diag
    mo_energy = np.diag(fock)

    # HF energy: E = sum_i h_ii + 0.5 * sum_ij [2*(ii|jj) - (ij|ji)] + E_nuc
    e_hf = (
        2.0 * np.einsum("ii->", h1[:nocc, :nocc])
        + 2.0 * np.einsum("iijj->", h2[:nocc, :nocc, :nocc, :nocc])
        - np.einsum("ijji->", h2[:nocc, :nocc, :nocc, :nocc])
        + elec_props.nuclear_repulsion_energy
    )

    mf.mo_coeff = mo_coeff
    mf.mo_occ = mo_occ
    mf.mo_energy = mo_energy
    mf.e_tot = e_hf
    mf.converged = True

    mycc = cc.CCSD(mf)
    mycc.max_cycle = max_cycle
    mycc.diis_space = 12

    # Warm-start: use old t2 as initial guess for faster convergence in rotated basis.
    t2_old = np.asarray(elec_props.t2)
    t1_init = np.zeros((nelec_a, norb - nelec_a))
    log.info("CC refresh (RHF): warm-start with old ||t2||=%.4e", np.linalg.norm(t2_old))
    mycc.kernel(t1=t1_init, t2=t2_old)

    if not mycc.converged:
        log.warning("CC refresh (RHF): CCSD did not converge after %d cycles. Using unconverged amplitudes.", max_cycle)

    t1_new, t2_new = mycc.t1, mycc.t2
    log.info(
        "CC refresh (RHF): converged. E_corr=%.6f ||t1||=%.4e ||t2||=%.4e",
        mycc.e_corr, np.linalg.norm(t1_new), np.linalg.norm(t2_new),
    )

    rdm1_cc = mycc.make_rdm1()
    occ_new = np.diag(rdm1_cc) / 2.0

    if np.any(np.isnan(t2_new)) or np.any(np.isnan(occ_new)):
        log.warning("CC refresh (RHF): NaN detected in amplitudes or occupancy. Discarding.")
        return (None, None) if return_energy else None

    props = ElectronicProperties(
        one_body_tensor=elec_props.one_body_tensor,
        two_body_tensor=elec_props.two_body_tensor,
        t2=t2_new,
        initial_occupancy=(occ_new, occ_new),
        nuclear_repulsion_energy=elec_props.nuclear_repulsion_energy,
        num_orbitals=norb,
        num_electrons=elec_props.num_electrons,
        open_shell=elec_props.open_shell,
        spin_sq=elec_props.spin_sq,
    )
    return (props, float(mycc.e_tot)) if return_energy else props


def _build_mock_uhf(
    elec_props: ElectronicProperties,
    verbose: int = 4,
    stdout=None,
):
    """Mock UHF SCF object for MO-basis spin-block integrals (mo_coeff = I, aufbau occupation).

    ``get_jk`` / ``get_veff`` handle the separate aa/ab/bb integral blocks so PySCF UCCSD works
    natively on it; ``mo_energy`` is the Fock diagonal and ``e_tot`` the energy of the index-aufbau
    determinant (lowest ``na`` alpha / ``nb`` beta orbitals).

    Returns ``(mf, make_eris)``; ``make_eris(mycc)`` builds the UCCSD ``_ChemistsERIs`` from the
    three blocks directly (``mf._eri`` only carries the aa block, so ``mycc.ao2mo()`` on this mock
    would be wrong for the ab / bb blocks).
    """
    norb = elec_props.num_orbitals
    nelec_a, nelec_b = elec_props.num_electrons
    h1_a = np.asarray(elec_props.one_body_tensor)
    h1_b = np.asarray(elec_props.one_body_tensor_b)
    h2_aa = np.asarray(elec_props.two_body_tensor)
    h2_ab = np.asarray(elec_props.two_body_tensor_ab)
    h2_bb = np.asarray(elec_props.two_body_tensor_bb)
    nuc = elec_props.nuclear_repulsion_energy

    mol = gto.M()
    mol.nelectron = nelec_a + nelec_b
    mol.spin = nelec_a - nelec_b
    mol.verbose = verbose
    if stdout is not None:
        mol.stdout = stdout
    mol.incore_anyway = True

    mf = scf.UHF(mol)
    ovlp = np.eye(norb)
    mf.get_ovlp = lambda *a, **k: ovlp
    mf.mol.nao_nr = lambda *a: norb
    mf.mol.energy_nuc = lambda *a: nuc
    mf.get_hcore = lambda *a, **k: h1_a

    _eri_aa_8fold = ao2mo.restore(8, h2_aa.reshape(norb, norb, norb, norb), norb)
    _eri_bb_8fold = ao2mo.restore(8, h2_bb.reshape(norb, norb, norb, norb), norb)
    _h2_ab = h2_ab
    mf._eri = _eri_aa_8fold

    def _uhf_get_jk(mol_arg=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if dm is None:
            dm = mf.make_rdm1()
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            from pyscf.scf import hf
            return hf.dot_eri_dm(_eri_aa_8fold, dm, hermi, with_j, with_k)
        dm_a, dm_b = dm[0], dm[1]
        from pyscf.scf import hf
        vj_aa, vk_aa = hf.dot_eri_dm(_eri_aa_8fold, dm_a, hermi, with_j, with_k)
        vj_bb, vk_bb = hf.dot_eri_dm(_eri_bb_8fold, dm_b, hermi, with_j, with_k)
        vj_ab = np.einsum("pqrs,rs->pq", _h2_ab, dm_b) if with_j else np.zeros_like(vj_aa)
        vj_ba = np.einsum("rspq,rs->pq", _h2_ab, dm_a) if with_j else np.zeros_like(vj_bb)
        vj_a = vj_aa + vj_ab
        vj_b = vj_bb + vj_ba
        vj = np.array((vj_a, vj_b))
        vk = np.array((vk_aa, vk_bb))
        return vj, vk
    mf.get_jk = _uhf_get_jk

    def _uhf_get_veff(mol_arg=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if dm is None:
            dm = mf.make_rdm1()
        vj, vk = _uhf_get_jk(dm=dm, hermi=hermi)
        # PySCF UHF computes fock = get_hcore() + get_veff(). get_hcore returns h1_a
        # (single matrix), so fock_b = h1_a + veff_b. To get the correct fock_b = h1_b + J_b - K_b,
        # we must include the (h1_b - h1_a) correction in veff_b.
        veff_a = vj[0] - vk[0]
        veff_b = vj[1] - vk[1] + (h1_b - h1_a)
        return np.array((veff_a, veff_b))
    mf.get_veff = _uhf_get_veff

    mo_a = np.eye(norb)
    mo_b = np.eye(norb)
    occ_a = np.zeros(norb)
    occ_a[:nelec_a] = 1.0
    occ_b = np.zeros(norb)
    occ_b[:nelec_b] = 1.0

    mf.mo_coeff = (mo_a, mo_b)
    mf.mo_occ = (occ_a, occ_b)

    nocc_a, nocc_b = nelec_a, nelec_b
    dm_a_hf = np.zeros((norb, norb))
    np.fill_diagonal(dm_a_hf[:nocc_a, :nocc_a], 1.0)
    dm_b_hf = np.zeros((norb, norb))
    np.fill_diagonal(dm_b_hf[:nocc_b, :nocc_b], 1.0)
    vj, vk = _uhf_get_jk(dm=(dm_a_hf, dm_b_hf))
    fock_a = h1_a + vj[0] - vk[0]
    fock_b = h1_b + vj[1] - vk[1]
    mf.mo_energy = (np.diag(fock_a), np.diag(fock_b))

    e_uhf = (
        np.einsum("ii->", h1_a[:nocc_a, :nocc_a])
        + np.einsum("ii->", h1_b[:nocc_b, :nocc_b])
        + 0.5 * (np.einsum("iijj->", h2_aa[:nocc_a, :nocc_a, :nocc_a, :nocc_a])
                 - np.einsum("ijji->", h2_aa[:nocc_a, :nocc_a, :nocc_a, :nocc_a]))
        + 0.5 * (np.einsum("iijj->", h2_bb[:nocc_b, :nocc_b, :nocc_b, :nocc_b])
                 - np.einsum("ijji->", h2_bb[:nocc_b, :nocc_b, :nocc_b, :nocc_b]))
        + np.einsum("iijj->", h2_ab[:nocc_a, :nocc_a, :nocc_b, :nocc_b])
        + nuc
    )
    mf.e_tot = e_uhf
    mf.converged = True

    # Supply correct spin-block MO integrals via ao2mofn passed to _make_eris_incore.
    # PySCF _make_eris_incore calls ao2mofn(moa), ao2mofn(mob), ao2mofn((moa,moa,mob,mob)).
    # With mo_coeff=I the integrals are already in MO basis; return them directly.
    def make_eris(mycc):
        def _ao2mofn(mo_or_tuple):
            if isinstance(mo_or_tuple, tuple):
                return h2_ab.reshape(-1)
            _ao2mofn._n += 1
            return (h2_aa if _ao2mofn._n == 1 else h2_bb).reshape(-1)
        _ao2mofn._n = 0

        from pyscf.cc.uccsd import _make_eris_incore
        return _make_eris_incore(mycc, mycc.mo_coeff, ao2mofn=_ao2mofn)

    return mf, make_eris


def _refresh_cc_seed_uhf(
    elec_props: ElectronicProperties,
    log,
    max_cycle: int,
    return_energy: bool = False,
) -> ElectronicProperties | None:
    """CC refresh for UHF.

    Runs UCCSD on the mock UHF of :func:`_build_mock_uhf`, warm-started from the previous t2.
    """
    norb = elec_props.num_orbitals
    nelec_a, nelec_b = elec_props.num_electrons
    h2_aa = np.asarray(elec_props.two_body_tensor)
    h2_ab = np.asarray(elec_props.two_body_tensor_ab)
    h2_bb = np.asarray(elec_props.two_body_tensor_bb)

    mf, make_eris = _build_mock_uhf(elec_props, verbose=4)
    e_uhf = mf.e_tot

    mycc = cc.UCCSD(mf)
    mycc.max_cycle = max_cycle
    mycc.diis_space = 12
    mycc.diis_start_cycle = 20

    log.info(
        "CC refresh (UHF): e_uhf=%.6f norb=%d nelec=(%d,%d) "
        "||h2_aa||=%.4e ||h2_ab||=%.4e ||h2_bb||=%.4e",
        e_uhf, norb, nelec_a, nelec_b,
        np.linalg.norm(h2_aa), np.linalg.norm(h2_ab), np.linalg.norm(h2_bb),
    )

    eris = make_eris(mycc)

    # Warm-start: use old t1/t2 as initial guess for faster convergence.
    t2_aa_old = np.asarray(elec_props.t2)
    t2_ab_old = np.asarray(elec_props.t2_ab)
    t2_bb_old = np.asarray(elec_props.t2_bb)
    nocc_a, nocc_b = nelec_a, nelec_b
    nvir_a, nvir_b = norb - nocc_a, norb - nocc_b
    t1_a_init = np.zeros((nocc_a, nvir_a))
    t1_b_init = np.zeros((nocc_b, nvir_b))
    log.info(
        "CC refresh (UHF): warm-start with old ||t2aa||=%.4e ||t2ab||=%.4e ||t2bb||=%.4e",
        np.linalg.norm(t2_aa_old), np.linalg.norm(t2_ab_old), np.linalg.norm(t2_bb_old),
    )
    mycc.kernel(t1=(t1_a_init, t1_b_init), t2=(t2_aa_old, t2_ab_old, t2_bb_old), eris=eris)

    if not mycc.converged:
        log.warning("CC refresh (UHF): UCCSD did not converge after %d cycles. Using unconverged amplitudes.", max_cycle)

    t2_aa_new, t2_ab_new, t2_bb_new = mycc.t2
    t1_a_new, t1_b_new = mycc.t1
    log.info(
        "CC refresh (UHF): E_corr=%.6f converged=%s ||t1a||=%.4e ||t1b||=%.4e "
        "||t2aa||=%.4e ||t2ab||=%.4e ||t2bb||=%.4e",
        mycc.e_corr, mycc.converged,
        np.linalg.norm(t1_a_new), np.linalg.norm(t1_b_new),
        np.linalg.norm(t2_aa_new), np.linalg.norm(t2_ab_new), np.linalg.norm(t2_bb_new),
    )

    # NaN check BEFORE make_rdm1: lambda equation crashes on NaN amplitudes
    if any(np.any(np.isnan(x)) or np.any(np.isinf(x)) for x in [t2_aa_new, t2_ab_new, t2_bb_new]):
        log.warning("CC refresh (UHF): NaN/Inf detected in amplitudes. Discarding.")
        return (None, None) if return_energy else None

    dm_a, dm_b = mycc.make_rdm1()
    occ_a_new = np.diag(dm_a)
    occ_b_new = np.diag(dm_b)

    if any(np.any(np.isnan(x)) for x in [occ_a_new, occ_b_new]):
        log.warning("CC refresh (UHF): NaN detected in occupancy. Discarding.")
        return (None, None) if return_energy else None

    props = ElectronicProperties(
        one_body_tensor=elec_props.one_body_tensor,
        two_body_tensor=elec_props.two_body_tensor,
        t2=t2_aa_new,
        one_body_tensor_b=elec_props.one_body_tensor_b,
        two_body_tensor_ab=elec_props.two_body_tensor_ab,
        two_body_tensor_bb=elec_props.two_body_tensor_bb,
        t2_ab=t2_ab_new,
        t2_bb=t2_bb_new,
        initial_occupancy=(occ_a_new, occ_b_new),
        nuclear_repulsion_energy=elec_props.nuclear_repulsion_energy,
        num_orbitals=elec_props.num_orbitals,
        num_electrons=elec_props.num_electrons,
        open_shell=elec_props.open_shell,
        unrestricted=True,
        spin_sq=elec_props.spin_sq,
        ab_overlap=elec_props.ab_overlap,
    )
    return (props, float(mycc.e_tot)) if return_energy else props


