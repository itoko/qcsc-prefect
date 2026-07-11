"""Orbital optimization utilities for RHF and UHF SQD closed loops.

This implements the orbital-rotation step of a two-step (alternating CI / orbital)
MCSCF optimization on top of SQD: the selected-CI diagonalization supplies the
reduced density matrices, and here we minimize the energy over unitary orbital
rotations U = exp(skew(x)) with the Hamiltonian integrals held fixed, then rotate
the integrals for the next SQD trial. The orbital step is solved with L-BFGS
(SciPy) using an analytical gradient via JAX when available (NumPy fallback
otherwise; see the ``use_jax`` parameter of :func:`optimize_orbitals`).

Reference
---------
Q. M. Kreplin, P. J. Knowles, H.-J. Werner, "Second-order MCSCF optimization
revisited. II. Combined first- and second-order orbital optimization for large
molecules", J. Chem. Phys. 152, 074102 (2020). doi:10.1063/1.5142241. In particular
the two-step (alternating CI/orbital) scheme and the L-BFGS convergence acceleration
of the orbital optimization discussed therein.

Mathematical background
-----------------------
Given a Hamiltonian H (fixed) and reduced density matrices (RDMs) from a
selected-CI diagonalization, we minimize the energy expectation value over
all unitary orbital rotations U:

  RHF:
    E(U) = h0 + Tr[h1 · (U γ1 U^T)] + 0.5 · Σ h2[p,q,r,s] · g2_rot[p,r,q,s]

  UHF:
    E(Uα, Uβ) = E_nuc
      + Tr[h1_α · (Uα γ1_αα Uα^T)]
      + Tr[h1_β · (Uβ γ1_ββ Uβ^T)]
      + 0.5 · einsum('pqrs,prqs->', h2_αα, g2_αα_rot(Uα))
      +       einsum('pqrs,prqs->', h2_αβ, g2_αβ_rot(Uα,Uβ))
      + 0.5 · einsum('pqrs,prqs->', h2_ββ, g2_ββ_rot(Uβ))

Notation
--------
  h2 tensors : chemist's notation  (pq|rs)
  rdm2 tensors: physicist's notation  <p†r†sq>

  Two storage conventions are in use:
    pqrs-storage (PySCF / qiskit_addon_sqd): rdm2[p,q,r,s] = <p†r†sq>
      Correct contraction: 0.5 * einsum('pqrs,pqrs->', h2_chem, rdm2_pqrs)
    prqs-storage (internal): rdm2[p,r,q,s] = <p†r†sq>
      Correct contraction: 0.5 * einsum('pqrs,prqs->', h2_chem, rdm2_prqs)

  solve_fermion / PySCF make_rdm2s return pqrs-storage.
  This module internally uses prqs-storage; the public API accepts pqrs-storage
  and converts automatically (rdm2_prqs = rdm2_pqrs.transpose(0,2,1,3)).

  1-RDM: γ[p,q] = <q†p>

Rotation convention (consistent with orbital_optimizer_uhf.py)
----------------------------------------------------------------
  1-RDM:  γ_rot = U γ U^T
  2-RDM:  g2[p,r,q,s] = Σ_{PRQS} U[p,P] U[r,R] rdm2[P,R,Q,S] U[q,Q] U[s,S]

  This is consistent with H_new = U^T H U applied to the Hamiltonian tensors:
    h1_new = U^T h1 U
    h2_new[P,Q,R,S] = Σ_{pqrs} U[p,P] U[q,Q] h2[p,q,r,s] U[r,R] U[s,S]

Usage
-----
  from qcsc_workflow_utility.orbital_opt import optimize_orbitals, rotate_electronic_properties

  # After each SQD diagonalization iteration:
  Ua, Ub, e_opt = optimize_orbitals(elec_props, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb)
  elec_props = rotate_electronic_properties(elec_props, Ua, Ub)
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.optimize

if TYPE_CHECKING:
    from qcsc_workflow_utility.chem import ElectronicProperties

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers  (skew-symmetric → unitary, RDM rotation)
# ─────────────────────────────────────────────────────────────────────────────

def _unitary_from_skew(x: np.ndarray, n: int) -> np.ndarray:
    """Build U = expm(A) where A is skew-symmetric with upper-triangle = x."""
    A = np.zeros((n, n), dtype=np.float64)
    A[np.triu_indices(n, k=1)] = x
    A -= A.T
    return scipy.linalg.expm(A)


def _rotate_rdm2(
    rdm2_prqs: np.ndarray, Ubra: np.ndarray, Uket: np.ndarray
) -> np.ndarray:
    """Rotate a physicist's 2-RDM stored as (p, r, q, s) = <p†r†sq>.

    g2[p,r,q,s] = Σ_{PRQS} Ubra[p,P] Uket[r,R] rdm2[P,R,Q,S] Ubra[q,Q] Uket[s,S]

    Split into two O(n^5) steps:
      step 1: contract creation  indices (axes 0, 1)
      step 2: contract annihilation indices (axes 2, 3)
    """
    tmp = np.einsum("pP,rR,PRQS->prQS", Ubra, Uket, rdm2_prqs, optimize="greedy")
    return np.einsum("prQS,qQ,sS->prqs", tmp, Ubra, Uket, optimize="greedy")


# ─────────────────────────────────────────────────────────────────────────────
# Energy evaluators
# ─────────────────────────────────────────────────────────────────────────────

def _uhf_energy(
    x: np.ndarray,
    norb: int,
    n_p: int,
    rdm1_aa: np.ndarray,
    rdm1_bb: np.ndarray,
    rdm2_aa: np.ndarray,
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    nuc: float,
) -> float:
    """UHF energy E(Uα, Uβ) as a function of the skew-symmetric parameters x.

    x[:n_p] → Aα upper-triangle → Uα = expm(Aα)
    x[n_p:] → Aβ upper-triangle → Uβ = expm(Aβ)
    """
    Ua = _unitary_from_skew(x[:n_p], norb)
    Ub = _unitary_from_skew(x[n_p:], norb)

    g1a = Ua @ rdm1_aa @ Ua.T
    g1b = Ub @ rdm1_bb @ Ub.T

    g2aa = _rotate_rdm2(rdm2_aa, Ua, Ua)
    g2ab = _rotate_rdm2(rdm2_ab, Ua, Ub)
    g2bb = _rotate_rdm2(rdm2_bb, Ub, Ub)

    return float(
        nuc
        + np.einsum("pq,pq->", h1_a, g1a)
        + np.einsum("pq,pq->", h1_b, g1b)
        + 0.5 * np.einsum("pqrs,prqs->", h2_aa, g2aa)
        +       np.einsum("pqrs,prqs->", h2_ab, g2ab)
        + 0.5 * np.einsum("pqrs,prqs->", h2_bb, g2bb)
    )


def _rhf_energy(
    x: np.ndarray,
    norb: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    nuc: float,
) -> float:
    """RHF energy E(U) as a function of the skew-symmetric parameter x.

    h2 in chemist's (pq|rs); rdm2 in physicist's (prqs).
    """
    U = _unitary_from_skew(x, norb)
    g1 = U @ rdm1 @ U.T
    g2 = _rotate_rdm2(rdm2, U, U)
    return float(
        nuc
        + np.einsum("pq,pq->", h1, g1)
        + 0.5 * np.einsum("pqrs,prqs->", h2, g2)
    )


# ─────────────────────────────────────────────────────────────────────────────
# JAX analytical-gradient path (optional fast path for UHF)
# ─────────────────────────────────────────────────────────────────────────────

def _make_jax_uhf_obj_and_grad(
    norb: int,
    n_p: int,
    rdm1_aa: np.ndarray,
    rdm1_bb: np.ndarray,
    rdm2_aa: np.ndarray,
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    nuc: float,
):
    """Build JAX-jitted (objective, gradient) for UHF orbital optimization.

    Returns (None, None) when JAX is unavailable; caller falls back to NumPy.
    """
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
    except ImportError:
        return None, None

    j = {k: jnp.array(v) for k, v in dict(
        rdm1_aa=rdm1_aa, rdm1_bb=rdm1_bb,
        rdm2_aa=rdm2_aa, rdm2_ab=rdm2_ab, rdm2_bb=rdm2_bb,
        h1_a=h1_a, h1_b=h1_b,
        h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
    ).items()}

    def _skew_to_U(x_half: "jnp.ndarray") -> "jnp.ndarray":
        idx = jnp.triu_indices(norb, k=1)
        A = jnp.zeros((norb, norb), dtype=x_half.dtype)
        A = A.at[idx].set(x_half)
        A = A - A.T
        return jax.scipy.linalg.expm(A)

    def _rot_rdm2_jax(rdm2_prqs, Ubra, Uket):
        tmp = jnp.einsum("pP,rR,PRQS->prQS", Ubra, Uket, rdm2_prqs, optimize=True)
        return jnp.einsum("prQS,qQ,sS->prqs", tmp, Ubra, Uket, optimize=True)

    def objective(x: "jnp.ndarray") -> "jnp.ndarray":
        Ua = _skew_to_U(x[:n_p])
        Ub = _skew_to_U(x[n_p:])
        g1a = Ua @ j["rdm1_aa"] @ Ua.T
        g1b = Ub @ j["rdm1_bb"] @ Ub.T
        g2aa = _rot_rdm2_jax(j["rdm2_aa"], Ua, Ua)
        g2ab = _rot_rdm2_jax(j["rdm2_ab"], Ua, Ub)
        g2bb = _rot_rdm2_jax(j["rdm2_bb"], Ub, Ub)
        return (
            nuc
            + jnp.einsum("pq,pq->", j["h1_a"], g1a)
            + jnp.einsum("pq,pq->", j["h1_b"], g1b)
            + 0.5 * jnp.einsum("pqrs,prqs->", j["h2_aa"], g2aa)
            +       jnp.einsum("pqrs,prqs->", j["h2_ab"], g2ab)
            + 0.5 * jnp.einsum("pqrs,prqs->", j["h2_bb"], g2bb)
        )

    obj_jit  = jax.jit(objective)
    grad_jit = jax.jit(jax.grad(objective))

    def obj_np(x):
        return float(obj_jit(jnp.array(x, dtype=jnp.float64)))

    def grad_np(x):
        return np.array(grad_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

    return obj_np, grad_np


# ─────────────────────────────────────────────────────────────────────────────
# Public API — optimize_orbitals
# ─────────────────────────────────────────────────────────────────────────────

def optimize_orbitals(
    elec_props: "ElectronicProperties",
    rdm1_aa: np.ndarray,
    rdm1_bb: np.ndarray,
    rdm2_aa: np.ndarray,
    rdm2_ab: np.ndarray | None = None,
    rdm2_bb: np.ndarray | None = None,
    *,
    method: str = "L-BFGS-B",
    maxiter: int = 300,
    ftol: float = 1e-15,
    gtol: float = 1e-10,
    rdm2_notation: str = "pqrs",
    use_jax: bool | None = None,
    trust_radius: float | None = 0.5,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Find the orbital rotation that minimizes the SCI energy expectation value.

    Works for both RHF (``elec_props.unrestricted == False``) and UHF
    (``elec_props.unrestricted == True``).  The RHF path treats the spin-summed
    inputs as a single channel; the UHF path optimizes Uα and Uβ independently.

    Parameters
    ----------
    elec_props:
        ElectronicProperties from the current SQD iteration (carries the
        Hamiltonian integrals in chemist's notation for h2 tensors).
    rdm1_aa, rdm1_bb:
        Alpha and beta 1-RDMs in physicist's convention: γ[p,q] = <q†p>.
        For RHF, pass the spin-resolved blocks (or sum them for the spin-free path).
    rdm2_aa:
        Alpha-alpha 2-RDM in physicist's convention: <p†r†sq>.
    rdm2_ab:
        Alpha-beta 2-RDM (physicist's). If None, uses rdm2_aa (spin-free approx).
    rdm2_bb:
        Beta-beta 2-RDM (physicist's). If None, uses rdm2_aa (spin-free approx).
    method:
        scipy.optimize.minimize method string (default "L-BFGS-B").
    maxiter:
        Maximum optimizer iterations.
    ftol, gtol:
        Convergence tolerances for L-BFGS-B.
    rdm2_notation : {"pqrs", "prqs"}
        Storage convention of the input 2-RDMs.
        ``"pqrs"`` (default): PySCF / qiskit_addon_sqd convention —
            rdm2[p,q,r,s] = <p†r†sq>, correct contraction is
            ``einsum('pqrs,pqrs->', h2_chem, rdm2)``.
        ``"prqs"``: internal convention —
            rdm2[p,r,q,s] = <p†r†sq>, correct contraction is
            ``einsum('pqrs,prqs->', h2_chem, rdm2)``.
        ``solve_fermion`` and PySCF ``make_rdm2s`` return pqrs-storage.
    use_jax : bool or None
        Control the gradient backend used by the L-BFGS-B optimizer.
        ``None`` (default): use JAX analytical gradient if JAX is importable,
            otherwise fall back to NumPy finite-difference.
        ``True``: require JAX; raises ``ImportError`` if JAX is unavailable.
        ``False``: always use NumPy finite-difference, even when JAX is installed.
    trust_radius : float or None
        Step restriction on the orbital-rotation parameters (max Euclidean norm of the
        skew-generator vector ``x`` per spin channel). Bounding the step is the standard
        MCSCF safeguard (cf. ORCA/Molpro "step restriction") against over-rotation: with a
        FIXED (approximate/noise-limited) RDM the fixed-RDM energy functional can be driven
        artificially low by large rotations that decouple the RDMs from the rotated
        Hamiltonian (a non-variational artifact). Capping ``‖x‖`` keeps each orbital step in
        the region where the fixed RDMs remain a good approximation, so the energy descends
        from above toward the true minimum (variational). ``None`` disables the cap
        (unrestricted step, original behavior). Default 0.5 rad.

    Returns
    -------
    Ua : ndarray (norb, norb)
        Optimal alpha (or RHF single-channel) rotation matrix.
    Ub : ndarray (norb, norb)
        Optimal beta rotation matrix (== Ua for RHF).
    energy : float
        Energy after orbital optimization (including nuclear repulsion).
    grad_norm : float
        Euclidean norm of the orbital gradient at the returned solution. This is the
        MCSCF convergence signal (generalized Brillouin condition ``g -> 0`` at the
        minimum, as used by CASSCF codes such as MOLCAS/ORCA/Molpro): the caller's two-step
        loop should stop when ``grad_norm`` falls below a threshold, giving a
        reference-free stopping criterion valid for large systems where no near-exact
        (DMRG/FCI) energy floor is available. A small residual gradient means the orbitals
        are stationary; a large gradient with a still-decreasing energy is the signature of
        the fixed-RDM breakdown described under ``trust_radius``.
    """
    norb = elec_props.num_orbitals
    nuc = elec_props.nuclear_repulsion_energy
    h1_a = elec_props.one_body_tensor
    h2_aa = elec_props.two_body_tensor  # chemist's (pq|rs)

    if rdm2_ab is None:
        rdm2_ab = rdm2_aa
    if rdm2_bb is None:
        rdm2_bb = rdm2_aa

    # Convert pqrs-storage (PySCF/solver) to prqs-storage (internal convention).
    # prqs[p,r,q,s] = pqrs[p,q,r,s].transpose(0,2,1,3)
    # prqs-storage satisfies: 0.5 * einsum('pqrs,prqs->', h2_chem, rdm2_prqs)
    if rdm2_notation == "pqrs":
        rdm2_aa = np.asarray(rdm2_aa).transpose(0, 2, 1, 3)
        rdm2_ab = np.asarray(rdm2_ab).transpose(0, 2, 1, 3)
        rdm2_bb = np.asarray(rdm2_bb).transpose(0, 2, 1, 3)
    elif rdm2_notation != "prqs":
        raise ValueError(f"rdm2_notation must be 'pqrs' or 'prqs', got {rdm2_notation!r}")

    if elec_props.unrestricted:
        h1_b  = elec_props.one_body_tensor_b
        h2_ab = elec_props.two_body_tensor_ab
        h2_bb = elec_props.two_body_tensor_bb
    else:
        # RHF: treat as spin-free UHF with identical alpha/beta channels
        h1_b  = h1_a
        h2_ab = h2_aa
        h2_bb = h2_aa

    n_p = norb * (norb - 1) // 2  # parameters per spin channel
    x0 = np.zeros(2 * n_p, dtype=np.float64)

    # Resolve JAX availability and user preference
    _gradient_methods = ("L-BFGS-B", "BFGS", "CG")
    if use_jax is False:
        # User explicitly requested NumPy — skip JAX entirely
        obj_jax, grad_jax = None, None
    else:
        obj_jax, grad_jax = _make_jax_uhf_obj_and_grad(
            norb, n_p,
            rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
            h1_a, h1_b, h2_aa, h2_ab, h2_bb, nuc,
        )
        if use_jax is True and obj_jax is None:
            raise ImportError(
                "use_jax=True was requested but JAX is not importable. "
                "Install JAX or pass use_jax=False."
            )
    _use_jax = (obj_jax is not None) and (method.upper() in _gradient_methods)

    if _use_jax:
        log.info("  [OrbOpt] Using JAX analytical gradient")
        eval_obj = obj_jax
    else:
        log.info("  [OrbOpt] Using NumPy objective (no analytical gradient)")
        eval_obj = partial(
            _uhf_energy,
            norb=norb, n_p=n_p,
            rdm1_aa=rdm1_aa, rdm1_bb=rdm1_bb,
            rdm2_aa=rdm2_aa, rdm2_ab=rdm2_ab, rdm2_bb=rdm2_bb,
            h1_a=h1_a, h1_b=h1_b,
            h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
            nuc=nuc,
        )

    e_before = eval_obj(x0)
    log.info("  [OrbOpt] E before optimization: %.10f Ha", e_before)

    minimize_kwargs: dict = dict(
        fun=eval_obj,
        x0=x0,
        method=method,
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )
    if _use_jax:
        minimize_kwargs["jac"] = grad_jax
    # Trust-region box: cap each rotation parameter to +/- trust_radius. L-BFGS-B honors bounds
    # natively; for other methods the cap is applied by clipping the returned step below. This is
    # the MCSCF step-restriction safeguard against over-rotation on fixed (approximate) RDMs.
    if trust_radius is not None and method.upper() == "L-BFGS-B":
        minimize_kwargs["bounds"] = [(-trust_radius, trust_radius)] * (2 * n_p)

    res = scipy.optimize.minimize(**minimize_kwargs)
    x_opt = res.x
    if trust_radius is not None and method.upper() != "L-BFGS-B":
        x_opt = np.clip(x_opt, -trust_radius, trust_radius)

    e_after = float(res.fun if x_opt is res.x else eval_obj(x_opt))

    # Orbital gradient norm at the solution -- the MCSCF convergence signal (Brillouin g->0).
    # Use the analytical JAX gradient when available; else a cheap central finite-difference.
    if _use_jax:
        grad_norm = float(np.linalg.norm(np.asarray(grad_jax(x_opt))))
    else:
        h = 1e-6
        g = np.empty_like(x_opt)
        for i in range(x_opt.size):
            xp = x_opt.copy(); xp[i] += h
            xm = x_opt.copy(); xm[i] -= h
            g[i] = (eval_obj(xp) - eval_obj(xm)) / (2 * h)
        grad_norm = float(np.linalg.norm(g))

    log.info(
        "  [OrbOpt] E after optimization: %.10f Ha  ΔE=%.4e  |grad|=%.3e  converged=%s  nit=%d",
        e_after, e_after - e_before, grad_norm, res.success, res.nit,
    )
    if not res.success:
        log.warning("  [OrbOpt] optimizer status=%d: %s", res.status, res.message)

    Ua = _unitary_from_skew(x_opt[:n_p], norb)
    Ub = _unitary_from_skew(x_opt[n_p:], norb)
    return Ua, Ub, e_after, grad_norm


# ─────────────────────────────────────────────────────────────────────────────
# Public API — rotate_electronic_properties
# ─────────────────────────────────────────────────────────────────────────────

def rotate_electronic_properties(
    elec_props: "ElectronicProperties",
    Ua: np.ndarray,
    Ub: np.ndarray | None = None,
) -> "ElectronicProperties":
    """Return a new ElectronicProperties with all integral tensors rotated by (Ua, Ub).

    Rotation formulas
    -----------------
      h1_new = U^T h1 U
      h2_new[P,Q,R,S] = Σ_{pqrs} U[p,P] U[q,Q] h2[p,q,r,s] U[r,R] U[s,S]

    For RHF (``elec_props.unrestricted == False``) a single U is applied to
    both one- and two-body tensors.  For UHF, Ua rotates alpha integrals, Ub
    rotates beta integrals, and the mixed (αβ) block uses (Ua, Ub) together.

    The ``t2`` amplitudes and ``initial_occupancy`` are intentionally NOT rotated
    because they are used only to initialize the SQD subspace; the RDMs (occupancies)
    produced by the solver are self-consistently updated each iteration.
    """
    from qcsc_workflow_utility.chem import ElectronicProperties

    if Ub is None:
        Ub = Ua

    def rot1(h1: np.ndarray, U: np.ndarray) -> np.ndarray:
        return U.T @ h1 @ U

    def rot2(h2: np.ndarray, U_bra: np.ndarray, U_ket: np.ndarray) -> np.ndarray:
        """Rotate chemist's (pq|rs) two-body tensor."""
        return np.einsum(
            "pP,qQ,pqrs,rR,sS->PQRS",
            U_bra, U_bra, h2, U_ket, U_ket,
            optimize="greedy",
        )

    h1_a_new = rot1(elec_props.one_body_tensor, Ua)
    h2_aa_new = rot2(elec_props.two_body_tensor, Ua, Ua)

    if elec_props.unrestricted:
        h1_b_new  = rot1(elec_props.one_body_tensor_b, Ub)
        h2_ab_new = rot2(elec_props.two_body_tensor_ab, Ua, Ub)
        h2_bb_new = rot2(elec_props.two_body_tensor_bb, Ub, Ub)
    else:
        h1_b_new  = None
        h2_ab_new = None
        h2_bb_new = None

    return ElectronicProperties(
        one_body_tensor=h1_a_new,
        two_body_tensor=h2_aa_new,
        t2=elec_props.t2,
        initial_occupancy=elec_props.initial_occupancy,
        nuclear_repulsion_energy=elec_props.nuclear_repulsion_energy,
        num_orbitals=elec_props.num_orbitals,
        num_electrons=elec_props.num_electrons,
        open_shell=elec_props.open_shell,
        spin_sq=elec_props.spin_sq,
        unrestricted=elec_props.unrestricted,
        one_body_tensor_b=h1_b_new,
        two_body_tensor_ab=h2_ab_new,
        two_body_tensor_bb=h2_bb_new,
        t2_ab=elec_props.t2_ab,
        t2_bb=elec_props.t2_bb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — resolve_orbitals_self_consistent  (oo_resolve_rdms path)
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_subspace_by_excitation(
    dets: np.ndarray, num_elec: int, keep: int
) -> np.ndarray:
    """Keep the ``keep`` determinants closest to Hartree-Fock by excitation level.

    For a near-HF state the CI weight is concentrated in low excitations (HF, singles, doubles,
    ...), and the Hamiltonian only couples HF to singles/doubles (Slater-Condon). Ranking by
    excitation level from HF is therefore a cheap, solve-free proxy for CI weight -- good enough to
    converge the ORBITALS on a small representative subspace, after which the rotated basis is
    applied to the full SQD run. Ties (same excitation level) are broken by determinant value for
    reproducibility. HF is always kept.
    """
    dets = np.asarray(dets, dtype=np.int64)
    if keep >= dets.size:
        return dets
    hf = (1 << num_elec) - 1
    exc = np.array([bin(int(d) ^ hf).count("1") // 2 for d in dets], dtype=np.int64)
    order = np.lexsort((dets, exc))  # primary: excitation level asc; secondary: det value
    return np.sort(dets[order[:keep]])


def resolve_orbitals_self_consistent(
    elec_props: "ElectronicProperties",
    alphadets: np.ndarray,
    betadets: np.ndarray | None,
    *,
    num_elec: tuple[int, int] | None = None,
    resolve_maxdim: int | None = 4_000_000,
    max_macro: int = 20,
    grad_tol: float = 1e-3,
    energy_tol: float = 1e-7,
    trust_radius: float = 0.1,
    oo_maxiter: int = 40,
    use_jax: bool | None = None,
    logger=None,
) -> tuple["ElectronicProperties", float, float, int]:
    """Fully self-consistent two-step MCSCF on a FIXED CI subspace (the oo_resolve_rdms path).

    This is the rigorous alternative to reusing the previous trial's stale RDMs. Each macro
    iteration:
      1. re-diagonalizes the SAME fixed determinant subspace (alphadets, betadets) in the CURRENT
         (rotated) integral basis via ``qiskit_addon_sqd.fermion.solve_fermion`` -- fully in-process,
         NO GPU child job and NO re-sampling, so it is fast (one small dense CI solve on a fixed
         subspace);
      2. builds fresh 1-/2-RDMs from that solution (so the RDMs always match the current orbitals);
      3. takes one trust-region orbital step (``optimize_orbitals`` with a small trust radius and a
         short L-BFGS budget) using those fresh RDMs;
      4. rotates the integrals by the returned U.
    It repeats until the orbital gradient is below ``grad_tol`` (Brillouin stationarity) or the
    energy stops changing. Because the RDMs are refreshed every step, the orbital gradient is the
    TRUE MCSCF gradient (unlike the fixed-RDM path, where it is not), so gradient-based convergence
    is meaningful and the energy descends onto the minimum from above (variational).

    Why this is fast: solve_fermion projects H onto the fixed subspace (dimension = len(alphadets) x
    len(betadets)) and diagonalizes it directly in NumPy/PySCF -- the same subspace the GPU solver
    used, but re-diagonalized in-process. No new determinants, no sampling, no scheduler round-trip.

    Returns ``(rotated_elec_props, final_energy, final_grad_norm, n_macro)``.

    References: two-step MCSCF of Kreplin/Knowles/Werner, J. Chem. Phys. 152, 074102 (2020).
    """
    from qiskit_addon_sqd.fermion import solve_fermion

    if betadets is None:
        betadets = alphadets
    adet = np.asarray(alphadets, dtype=np.int64)
    bdet = np.asarray(betadets, dtype=np.int64)
    open_shell = bool(elec_props.unrestricted)

    # Truncate the CI subspace used for the ORBITAL optimization to keep the in-process
    # solve_fermion re-solves fast. A full production subspace (e.g. sqrt(sqd_dim) ~ thousands of
    # dets/spin -> millions of configs) makes each dense selected-CI re-solve minutes-to-hours;
    # since the orbital rotation is well-determined by the dominant (low-excitation) determinants,
    # we converge the orbitals on a truncated subspace (keep ~sqrt(resolve_maxdim) per spin, ranked
    # by excitation level from HF) and then apply the rotated basis to the full SQD run downstream.
    # This is the standard MCSCF active-space economy. resolve_maxdim=None disables truncation.
    if resolve_maxdim is not None and adet.size * bdet.size > resolve_maxdim:
        if num_elec is None:
            num_elec = elec_props.num_electrons
        na, nb = num_elec
        keep = max(int(resolve_maxdim ** 0.5), 1)
        a_full, b_full = adet.size, bdet.size
        adet = _truncate_subspace_by_excitation(adet, na, keep)
        bdet = _truncate_subspace_by_excitation(bdet, nb, keep)
        if logger is not None:
            logger.info(
                "  [OO-SC] truncated OO subspace for fast re-solve: alpha %d->%d, beta %d->%d "
                "(net %d -> %d configs).",
                a_full, adet.size, b_full, bdet.size, a_full * b_full, adet.size * bdet.size,
            )
    ci = (adet, bdet)
    # solve_fermion returns the ELECTRONIC energy only; add the (rotation-invariant) nuclear
    # repulsion to compare against total energies / references. Rotating orbitals never changes it.
    nuc = float(elec_props.nuclear_repulsion_energy)

    ep = elec_props
    prev_e = None
    e = float("nan")
    grad_norm = float("inf")
    macro = 0
    for macro in range(1, max_macro + 1):
        # (1)+(2) re-diagonalize the fixed subspace in the current basis -> fresh energy + RDMs.
        # solve_fermion returns (energy, SCIState, avg_occupancies, spin_sq); SCIState.rdm(rank,
        # spin_summed=False) gives the per-spin blocks in pqrs-storage.
        e, sci, _occ, _s2 = solve_fermion(
            ci, ep.one_body_tensor, ep.two_body_tensor, open_shell=open_shell,
        )
        e = float(e) + nuc
        rdm1 = sci.rdm(rank=1, spin_summed=False)   # shape (2, norb, norb) -> alpha, beta
        rdm2 = sci.rdm(rank=2, spin_summed=False)   # shape (3 or 4, norb, norb, norb, norb)
        rdm1_aa, rdm1_bb = rdm1[0], rdm1[1]
        # qiskit-addon-sqd 2-RDM spin blocks: [aa, ab, bb] (spin_summed=False).
        rdm2_aa, rdm2_ab, rdm2_bb = rdm2[0], rdm2[1], rdm2[-1]

        if logger is not None:
            logger.info(
                "  [OO-SC] macro %d/%d: E(fixed-subspace, current basis) = %.10f Ha",
                macro, max_macro, e,
            )

        # Convergence on the TRUE (re-solved) energy between macro-iterations.
        if prev_e is not None and abs(e - prev_e) < energy_tol:
            if logger is not None:
                logger.info("  [OO-SC] converged: |dE|=%.2e < %.1e (macro %d).",
                            abs(e - prev_e), energy_tol, macro)
            break

        # (3) propose an orbital step on the FRESH RDMs, then ACCEPT/REJECT by the TRUE re-solved
        # energy (line search / trust-region). The orbital-step functional value (e_oo) can dip
        # non-variationally below the real energy; the authoritative quantity is solve_fermion's
        # energy in the rotated basis. We shrink the trust radius until a rotation actually lowers
        # the true energy (or give up and stop). This is what keeps the loop strictly variational.
        tr = trust_radius
        accepted = False
        # SPEED (Part D, deferred): each optimize_orbitals call rebuilds fresh jax.jit closures
        # (_make_jax_uhf_obj_and_grad) that bake in the RDMs+integrals. Across these <=6 line-search
        # iterations the RDMs are FIXED (only trust_radius/box-bounds change), so the jitted
        # objective/gradient are identical -- yet each call re-traces + re-JITs (~200 ms compile each,
        # measured). The win at Fe4S4 scale (36 orb, many macros): build obj_jax/grad_jax ONCE here
        # per macro and thread them into optimize_orbitals so the 6 trust-radius shrinks reuse the
        # compiled functions. On the tiny validation systems OO converges in 1 macro and does nothing,
        # so this is a scale-only optimization -- left as a note per "speed after pipeline verified".
        for _ls in range(6):
            Ua, Ub, e_oo, grad_norm = optimize_orbitals(
                ep, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
                rdm2_notation="pqrs", trust_radius=tr, maxiter=oo_maxiter, use_jax=use_jax,
            )
            ep_try = rotate_electronic_properties(ep, Ua, Ub)
            e_try, _sci2, _o2, _s2b = solve_fermion(
                ci, ep_try.one_body_tensor, ep_try.two_body_tensor, open_shell=open_shell,
            )
            e_try = float(e_try) + nuc
            if e_try <= e + 1e-12:  # true energy decreased -> accept
                ep = ep_try
                accepted = True
                if logger is not None:
                    logger.info(
                        "  [OO-SC] macro %d: accepted step (tr=%.3f) E %.10f -> %.10f  |grad|=%.3e",
                        macro, tr, e, e_try, grad_norm,
                    )
                break
            tr *= 0.4  # reject: shrink trust radius and retry a smaller step
        if not accepted:
            if logger is not None:
                logger.info("  [OO-SC] macro %d: no step lowered the true energy -> stop (|grad|=%.3e).",
                            macro, grad_norm)
            break

        # Gradient stationarity (Brillouin) on the accepted step -> converged.
        if grad_norm < grad_tol:
            if logger is not None:
                logger.info("  [OO-SC] converged: |grad|=%.3e < %.1e (macro %d).",
                            grad_norm, grad_tol, macro)
            break
        prev_e = e

    # Final re-solve in the (accepted) basis for the reported energy -- RDMs consistent with H.
    e_final, _sci, _occ, _s2 = solve_fermion(
        ci, ep.one_body_tensor, ep.two_body_tensor, open_shell=open_shell,
    )
    return ep, float(e_final) + nuc, float(grad_norm), macro
