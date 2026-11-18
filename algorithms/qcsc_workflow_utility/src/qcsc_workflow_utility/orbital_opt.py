"""Orbital optimization utilities for RHF and UHF SQD closed loops.

This implements the orbital-rotation step of a two-step (alternating CI / orbital)
MCSCF optimization on top of SQD: the selected-CI diagonalization supplies the
reduced density matrices, and here we minimize the energy over unitary orbital
rotations U = exp(skew(x)) using the RDMs from that SQD/Davidson trial as a
fixed objective during this one orbital-rotation call, then rotate the
integrals for the next SQD trial. The outer SQD loop supplies fresh RDMs after
each new Davidson trial. The orbital step is solved with L-BFGS (SciPy) using
an analytical gradient via JAX when available (NumPy fallback otherwise; see
the ``use_jax`` parameter of :func:`optimize_orbitals`).

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

  RHF (one rotation U for both spins; spin-summed RDMs γ1 = γα + γβ,
       g2 = aa + ab + ba + bb, see spin_summed_rdms):
    E(U) = h0 + Tr[h1 · (U γ1 U^T)] + 0.5 · Σ h2[p,q,r,s] · g2_rot[p,r,q,s]

  UHF:
    E(Uα, Uβ) = E_nuc
      + Tr[h1_α · (Uα γ1_αα Uα^T)]
      + Tr[h1_β · (Uβ γ1_ββ Uβ^T)]
      + 0.5 · einsum('pqrs,prqs->', h2_αα, g2_αα_rot(Uα))
      +       einsum('pqrs,prqs->', h2_αβ, g2_αβ_rot(Uα,Uβ))
      + 0.5 · einsum('pqrs,prqs->', h2_ββ, g2_ββ_rot(Uβ))
    (restricted=True: Uα = Uβ = U on the UHF integrals)

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
  # RDMs are frozen only during this call; pass fresh RDMs on the next trial.
  Ua, Ub, e_opt, grad = optimize_orbitals(elec_props, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb)
  elec_props = rotate_electronic_properties(elec_props, Ua, Ub)

  # RHF, spin-summed RDMs (e.g. the native RHF solver's rdm2.txt, prqs storage):
  U, e_opt, grad = optimize_orbitals_rhf(elec_props, rdm1, rdm2, rdm2_notation="prqs")
  elec_props = rotate_electronic_properties(elec_props, U)
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
# Spin summation (RHF orbital optimization input)
# ─────────────────────────────────────────────────────────────────────────────

def spin_summed_rdms(
    rdm1_a: np.ndarray,
    rdm1_b: np.ndarray,
    rdm2_aa: np.ndarray,
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    *,
    rdm2_notation: str = "prqs",
) -> tuple[np.ndarray, np.ndarray]:
    """Spin-summed 1-/2-RDM ``(gamma, Gamma)`` from the per-spin blocks, in the input storage.

      gamma = gamma_a + gamma_b
      Gamma = aa + ab + ba + bb

    The ba block (beta index first) is the ab block with the two electron pairs swapped:
    ``ab.transpose(1, 0, 3, 2)`` in prqs storage, ``ab.transpose(2, 3, 0, 1)`` in pqrs storage.
    The native RHF solver build writes Gamma directly (rdm2.txt, prqs); this is the fallback for
    binaries that write only the per-spin blocks, and for PySCF / solve_fermion per-spin RDMs.
    """
    if rdm2_notation == "prqs":
        swap = (1, 0, 3, 2)
    elif rdm2_notation == "pqrs":
        swap = (2, 3, 0, 1)
    else:
        raise ValueError(f"rdm2_notation must be 'pqrs' or 'prqs', got {rdm2_notation!r}")
    rdm2_ab = np.asarray(rdm2_ab)
    gamma = np.asarray(rdm1_a) + np.asarray(rdm1_b)
    big_gamma = np.asarray(rdm2_aa) + np.asarray(rdm2_bb) + rdm2_ab + rdm2_ab.transpose(swap)
    return gamma, big_gamma


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
    restricted: bool = False,
) -> float:
    """UHF energy E(Uα, Uβ) as a function of the skew-symmetric parameters x.

    x[:n_p] → Aα upper-triangle → Uα = expm(Aα)
    x[n_p:] → Aβ upper-triangle → Uβ = expm(Aβ)
    restricted=True: x has n_p entries and Uβ = Uα (one rotation for both spins).
    """
    Ua = _unitary_from_skew(x[:n_p], norb)
    Ub = Ua if restricted else _unitary_from_skew(x[n_p:], norb)

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

    h2 in chemist's (pq|rs); rdm1 / rdm2 spin-summed, rdm2 in physicist's (prqs).
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
# JAX analytical-gradient path (optional fast path)
# ─────────────────────────────────────────────────────────────────────────────

def _import_jax():
    """Return (jax, jax.numpy) with float64 enabled, or (None, None) when JAX is unavailable."""
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
    except ImportError:
        return None, None
    return jax, jnp


def _jax_skew_to_U(jax, jnp, x_half, norb: int):
    idx = jnp.triu_indices(norb, k=1)
    A = jnp.zeros((norb, norb), dtype=x_half.dtype)
    A = A.at[idx].set(x_half)
    A = A - A.T
    return jax.scipy.linalg.expm(A)


def _jax_rot_rdm2(jnp, rdm2_prqs, Ubra, Uket):
    tmp = jnp.einsum("pP,rR,PRQS->prQS", Ubra, Uket, rdm2_prqs, optimize=True)
    return jnp.einsum("prQS,qQ,sS->prqs", tmp, Ubra, Uket, optimize=True)


def _jit_to_numpy(jax, jnp, objective):
    """Wrap a JAX objective as NumPy-in/NumPy-out (objective, gradient) for scipy."""
    obj_jit  = jax.jit(objective)
    grad_jit = jax.jit(jax.grad(objective))

    def obj_np(x):
        return float(obj_jit(jnp.array(x, dtype=jnp.float64)))

    def grad_np(x):
        return np.array(grad_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

    return obj_np, grad_np


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
    restricted: bool = False,
):
    """Build JAX-jitted (objective, gradient) for UHF orbital optimization.

    restricted=True: x has n_p entries and Uβ = Uα. Returns (None, None) when JAX is
    unavailable; caller falls back to NumPy.
    """
    jax, jnp = _import_jax()
    if jax is None:
        return None, None

    j = {k: jnp.array(v) for k, v in dict(
        rdm1_aa=rdm1_aa, rdm1_bb=rdm1_bb,
        rdm2_aa=rdm2_aa, rdm2_ab=rdm2_ab, rdm2_bb=rdm2_bb,
        h1_a=h1_a, h1_b=h1_b,
        h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
    ).items()}

    def objective(x: "jnp.ndarray") -> "jnp.ndarray":
        Ua = _jax_skew_to_U(jax, jnp, x[:n_p], norb)
        Ub = Ua if restricted else _jax_skew_to_U(jax, jnp, x[n_p:], norb)
        g1a = Ua @ j["rdm1_aa"] @ Ua.T
        g1b = Ub @ j["rdm1_bb"] @ Ub.T
        g2aa = _jax_rot_rdm2(jnp, j["rdm2_aa"], Ua, Ua)
        g2ab = _jax_rot_rdm2(jnp, j["rdm2_ab"], Ua, Ub)
        g2bb = _jax_rot_rdm2(jnp, j["rdm2_bb"], Ub, Ub)
        return (
            nuc
            + jnp.einsum("pq,pq->", j["h1_a"], g1a)
            + jnp.einsum("pq,pq->", j["h1_b"], g1b)
            + 0.5 * jnp.einsum("pqrs,prqs->", j["h2_aa"], g2aa)
            +       jnp.einsum("pqrs,prqs->", j["h2_ab"], g2ab)
            + 0.5 * jnp.einsum("pqrs,prqs->", j["h2_bb"], g2bb)
        )

    return _jit_to_numpy(jax, jnp, objective)


def _make_jax_rhf_obj_and_grad(
    norb: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    nuc: float,
):
    """Build JAX-jitted (objective, gradient) for RHF orbital optimization (one U, spin-summed
    RDMs, rdm2 in prqs storage). Returns (None, None) when JAX is unavailable."""
    jax, jnp = _import_jax()
    if jax is None:
        return None, None

    j_rdm1, j_rdm2, j_h1, j_h2 = (jnp.array(a) for a in (rdm1, rdm2, h1, h2))

    def objective(x: "jnp.ndarray") -> "jnp.ndarray":
        U = _jax_skew_to_U(jax, jnp, x, norb)
        g1 = U @ j_rdm1 @ U.T
        g2 = _jax_rot_rdm2(jnp, j_rdm2, U, U)
        return (
            nuc
            + jnp.einsum("pq,pq->", j_h1, g1)
            + 0.5 * jnp.einsum("pqrs,prqs->", j_h2, g2)
        )

    return _jit_to_numpy(jax, jnp, objective)


# ─────────────────────────────────────────────────────────────────────────────
# Shared optimizer driver
# ─────────────────────────────────────────────────────────────────────────────

_GRADIENT_METHODS = ("L-BFGS-B", "BFGS", "CG")


def _resolve_jax(use_jax: bool | None, method: str, build):
    """Build the JAX (objective, gradient) per ``use_jax`` and report whether to use it.

    Returns ``(obj_jax, grad_jax, use)``; ``build`` is a zero-argument builder.
    """
    if use_jax is False:
        # User explicitly requested NumPy — skip JAX entirely
        return None, None, False
    obj_jax, grad_jax = build()
    if use_jax is True and obj_jax is None:
        raise ImportError(
            "use_jax=True was requested but JAX is not importable. "
            "Install JAX or pass use_jax=False."
        )
    use = (obj_jax is not None) and (method.upper() in _GRADIENT_METHODS)
    if use:
        log.info("  [OrbOpt] Using JAX analytical gradient")
    else:
        log.info("  [OrbOpt] Using NumPy objective (no analytical gradient)")
    return obj_jax, grad_jax, use


def _to_prqs(rdm2: np.ndarray, rdm2_notation: str) -> np.ndarray:
    """Convert pqrs-storage (PySCF/solver) to prqs-storage (internal convention).

    prqs[p,r,q,s] = pqrs[p,q,r,s].transpose(0,2,1,3); prqs-storage satisfies
    0.5 * einsum('pqrs,prqs->', h2_chem, rdm2_prqs).
    """
    if rdm2_notation == "pqrs":
        return np.asarray(rdm2).transpose(0, 2, 1, 3)
    if rdm2_notation != "prqs":
        raise ValueError(f"rdm2_notation must be 'pqrs' or 'prqs', got {rdm2_notation!r}")
    return np.asarray(rdm2)


def _log_self_consistency(e_before: float, nuc: float, davidson_ref_energy: float | None) -> None:
    # Self-consistency: E reconstructed from these RDMs at U=I (electronic, nuc removed)
    # must equal the Davidson eigenvalue of the SAME state. A nonzero Delta on the SAME
    # best_sbd_result rules provenance out and points at the RDM build / spin fallback.
    if davidson_ref_energy is None:
        return
    _elec_recon = float(e_before) - float(nuc)
    _dsc = _elec_recon - float(davidson_ref_energy)
    log.info(
        "  [OrbOpt] SELF-CONSISTENCY: E_recon(elec,U=I)=%.10f  E_davidson(elec)=%.10f  "
        "Delta=%+.3e Ha (%.3f mHa)  [same state => should be ~0]",
        _elec_recon, float(davidson_ref_energy), _dsc, _dsc * 1000.0,
    )


def _maybe_dump_rdms(**arrays) -> None:
    # Optional raw dump (OO_DUMP_RDM=1): persist integrals + RDMs + reference energy so the
    # RDM->energy contraction can be brute-forced offline against the native Davidson value.
    import os as _os
    if _os.environ.get("OO_DUMP_RDM") != "1":
        return
    _dd = _os.environ.get("OO_DUMP_DIR", "/home/nfs1/q0000230/work/rdm_dumps")
    _os.makedirs(_dd, exist_ok=True)
    _tag = _os.environ.get("SLURM_JOB_ID", "local")
    _idx = 0
    while _os.path.exists(_os.path.join(_dd, f"rdmdump_{_tag}_{_idx}.npz")):
        _idx += 1
    _fp = _os.path.join(_dd, f"rdmdump_{_tag}_{_idx}.npz")
    np.savez_compressed(_fp, **{k: np.asarray(v) for k, v in arrays.items()})
    log.info("  [OrbOpt] OO_DUMP_RDM: wrote %s", _fp)


def _minimize_orbital_objective(
    eval_obj,
    grad,
    n_params: int,
    e_before: float,
    *,
    method: str,
    maxiter: int,
    ftol: float,
    gtol: float,
    trust_radius: float | None,
) -> tuple[np.ndarray, float, float]:
    """Minimize ``eval_obj`` over the skew parameters from x = 0; shared by the RHF and UHF paths.

    ``grad`` is the analytical gradient (JAX) or None (scipy's finite differences). Returns
    ``(x_opt, e_after, grad_norm)``.
    """
    x0 = np.zeros(n_params, dtype=np.float64)
    minimize_kwargs: dict = dict(
        fun=eval_obj,
        x0=x0,
        method=method,
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )
    if grad is not None:
        minimize_kwargs["jac"] = grad
    # Trust-region box: cap each rotation parameter to +/- trust_radius. L-BFGS-B honors bounds
    # natively; for other methods the cap is applied by clipping the returned step below. This is
    # the MCSCF step-restriction safeguard against over-rotation on fixed (approximate) RDMs.
    if trust_radius is not None and method.upper() == "L-BFGS-B":
        minimize_kwargs["bounds"] = [(-trust_radius, trust_radius)] * n_params

    # --- iteration trajectory logging: follow the L-BFGS-B objective E_oo per step ---
    # scipy calls this after every accepted iteration. Reuse the existing objective
    # (and JAX gradient when available); all values are TOTAL energies (incl. nuc), so
    # they compare directly against the Davidson total energy. Logging only -- does not
    # affect the optimizer or the returned result.
    _oo_trace = {"it": 0, "prev": e_before}

    def _oo_callback(xk, *args):  # L-BFGS-B passes only xk
        _oo_trace["it"] += 1
        fk = float(eval_obj(xk))
        if grad is not None:
            _gk_vec = np.asarray(grad(xk))
            gk = float(np.linalg.norm(_gk_vec))              # L2 norm
            gk_max = float(np.max(np.abs(_gk_vec)))          # max-norm == scipy L-BFGS-B pgtol test
            log.info(
                "  [OrbOpt] L-BFGS-B iter %3d: E_oo=%.10f Ha  dE=%+.3e (vs prev)  "
                "dE=%+.3e (vs U=I)  |grad|_2=%.3e  |grad|_inf=%.3e (pgtol)",
                _oo_trace["it"], fk, fk - _oo_trace["prev"], fk - e_before, gk, gk_max,
            )
        else:
            log.info(
                "  [OrbOpt] L-BFGS-B iter %3d: E_oo=%.10f Ha  dE=%+.3e (vs prev)  "
                "dE=%+.3e (vs U=I)",
                _oo_trace["it"], fk, fk - _oo_trace["prev"], fk - e_before,
            )
        _oo_trace["prev"] = fk

    minimize_kwargs["callback"] = _oo_callback

    res = scipy.optimize.minimize(**minimize_kwargs)
    x_opt = res.x
    if trust_radius is not None and method.upper() != "L-BFGS-B":
        x_opt = np.clip(x_opt, -trust_radius, trust_radius)

    e_after = float(res.fun if x_opt is res.x else eval_obj(x_opt))

    # Orbital gradient norm at the solution -- the MCSCF convergence signal (Brillouin g->0).
    # Use the analytical JAX gradient when available; else a cheap central finite-difference.
    if grad is not None:
        _g_final = np.asarray(grad(x_opt))
    else:
        h = 1e-6
        _g_final = np.empty_like(x_opt)
        for i in range(x_opt.size):
            xp = x_opt.copy(); xp[i] += h
            xm = x_opt.copy(); xm[i] -= h
            _g_final[i] = (eval_obj(xp) - eval_obj(xm)) / (2 * h)
    grad_norm = float(np.linalg.norm(_g_final))          # L2 norm (reported historically)
    grad_norm_max = float(np.max(np.abs(_g_final)))      # max-norm == scipy L-BFGS-B pgtol test

    log.info(
        "  [OrbOpt] E after optimization (jax calculation): %.10f Ha  ΔE=%.4e  "
        "|grad|_2=%.3e  |grad|_inf=%.3e (pgtol)  converged=%s  nit=%d",
        e_after, e_after - e_before, grad_norm, grad_norm_max, res.success, res.nit,
    )
    if not res.success:
        log.warning("  [OrbOpt] optimizer status=%d: %s", res.status, res.message)

    return x_opt, e_after, grad_norm


# ─────────────────────────────────────────────────────────────────────────────
# Public API — optimize_orbitals_rhf  (one rotation, spin-summed RDMs)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_orbitals_rhf(
    elec_props: "ElectronicProperties",
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    *,
    method: str = "L-BFGS-B",
    maxiter: int = 300,
    ftol: float = 0,
    gtol: float = 1e-5,
    rdm2_notation: str = "pqrs",
    use_jax: bool | None = None,
    trust_radius: float | None = 0.5,
    davidson_ref_energy: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """RHF orbital optimization: one rotation U shared by alpha and beta.

    Minimizes E(U) = E_nuc + Tr[h1 · U γ U^T] + 0.5 · Σ h2[p,q,r,s] · Γ_rot[p,r,q,s] over the
    norb(norb-1)/2 skew parameters of U = exp(skew(x)), with the spin-summed RDMs (γ = γα + γβ,
    Γ = aa + ab + ba + bb; see :func:`spin_summed_rdms`) of one SQD/Davidson trial held fixed. The
    integrals are the restricted ones of ``elec_props`` (spatial FCIDUMP). Rotate them for the
    next trial with ``rotate_electronic_properties(elec_props, U)``.

    Parameters are those of :func:`optimize_orbitals`; ``trust_radius`` caps each component of
    the single parameter vector. ``rdm2_notation`` is "prqs" for the native solver's rdm2.txt and
    "pqrs" for PySCF ``make_rdm2`` / solve_fermion.

    Returns ``(U, energy, grad_norm)``: the rotation, the objective after optimization (total,
    including nuclear repulsion) and the L2 norm of the orbital gradient there.
    """
    if elec_props.unrestricted:
        raise ValueError(
            "optimize_orbitals_rhf needs restricted integrals (elec_props.unrestricted=False); "
            "use optimize_orbitals(..., restricted=True) for one rotation on UHF orbitals."
        )
    norb = elec_props.num_orbitals
    nuc = elec_props.nuclear_repulsion_energy
    h1 = elec_props.one_body_tensor
    h2 = elec_props.two_body_tensor  # chemist's (pq|rs)
    rdm1 = np.asarray(rdm1, dtype=np.float64)
    rdm2 = _to_prqs(rdm2, rdm2_notation)
    n_p = norb * (norb - 1) // 2

    obj_jax, grad_jax, _use_jax = _resolve_jax(
        use_jax, method, lambda: _make_jax_rhf_obj_and_grad(norb, rdm1, rdm2, h1, h2, nuc),
    )
    if _use_jax:
        eval_obj = obj_jax
    else:
        eval_obj = partial(_rhf_energy, norb=norb, rdm1=rdm1, rdm2=rdm2, h1=h1, h2=h2, nuc=nuc)

    e_before = eval_obj(np.zeros(n_p, dtype=np.float64))
    log.info(
        "  [OrbOpt] RHF (one rotation, spin-summed RDMs): E(trial-RDM; U=I) objective before "
        "optimization: %.10f Ha",
        e_before,
    )

    # Diagnostics at U=I, the RHF counterparts of the UHF battery in optimize_orbitals.
    _e1 = float(np.einsum("pq,pq->", h1, rdm1))
    _e2 = float(0.5 * np.einsum("pqrs,prqs->", h2, rdm2))
    log.info(
        "  [OrbOpt] E(trial-RDM; U=I) decomposition: nuc=%.6f e1=%.6f e2=%.6f total=%.10f",
        nuc, _e1, _e2, nuc + _e1 + _e2,
    )
    _N = int(elec_props.num_electrons[0]) + int(elec_props.num_electrons[1])
    _dev = float(np.max(np.abs(np.einsum("prqr->pq", rdm2) - (_N - 1) * rdm1)))
    log.info(
        "  [OrbOpt] RDM trace: Tr(rdm1)=%.6f (N=%d)  Tr(rdm2)=%.6f (want N(N-1)=%d)  "
        "partial-trace->1RDM max|dev|=%.3e  [0 => 2RDM consistent with 1RDM]",
        float(np.trace(rdm1)), _N, float(np.einsum("prpr->", rdm2)), _N * (_N - 1), _dev,
    )
    _log_self_consistency(e_before, nuc, davidson_ref_energy)
    _maybe_dump_rdms(
        h1=h1, h2=h2, rdm1=rdm1, rdm2=rdm2, nuc=float(nuc),
        e_davidson=float(davidson_ref_energy) if davidson_ref_energy is not None else np.nan,
        e_recon=float(e_before) - float(nuc), n=_N,
    )

    x_opt, e_after, grad_norm = _minimize_orbital_objective(
        eval_obj, grad_jax if _use_jax else None, n_p, e_before,
        method=method, maxiter=maxiter, ftol=ftol, gtol=gtol, trust_radius=trust_radius,
    )
    return _unitary_from_skew(x_opt, norb), e_after, grad_norm


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
    ftol: float = 0,
    gtol: float = 1e-5,
    rdm2_notation: str = "pqrs",
    use_jax: bool | None = None,
    trust_radius: float | None = 0.5,
    davidson_ref_energy: float | None = None,
    restricted: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Find the orbital rotation that minimizes the SCI energy expectation value.

    Works for both RHF (``elec_props.unrestricted == False``) and UHF
    (``elec_props.unrestricted == True``).  RHF integrals hold one orbital set, so the RHF path
    sums the per-spin inputs (:func:`spin_summed_rdms`) and optimizes a single rotation
    (:func:`optimize_orbitals_rhf`). The UHF path optimizes Uα and Uβ independently, or one
    rotation for both spins with ``restricted=True``.

    Parameters
    ----------
    elec_props:
        ElectronicProperties from the current SQD iteration (carries the
        Hamiltonian integrals in chemist's notation for h2 tensors).
    rdm1_aa, rdm1_bb:
        Alpha and beta 1-RDMs in physicist's convention: γ[p,q] = <q†p>.
        These are the RDMs from one SQD/Davidson trial. They remain fixed
        throughout this call; the outer loop should pass fresh RDMs after the
        next Davidson/SQD trial.
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
        ``"prqs"``: native SBD solver / internal convention —
            rdm2[p,r,q,s] = <p†r†sq>, correct contraction is
            ``einsum('pqrs,prqs->', h2_chem, rdm2)``.
        ``solve_fermion`` and PySCF ``make_rdm2s`` return pqrs-storage;
        the native SBD solver RDM files and ``solver_job.py`` use prqs-storage.
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
    restricted : bool
        UHF only: optimize one rotation U applied to both the alpha and beta orbitals
        (Uα = Uβ = U, norb(norb-1)/2 parameters) with the spin-resolved RDMs and integrals. RHF
        integrals are always optimized this way.

    Returns
    -------
    Ua : ndarray (norb, norb)
        Optimal alpha (or RHF single-channel) rotation matrix.
    Ub : ndarray (norb, norb)
        Optimal beta rotation matrix (== Ua for RHF and for ``restricted=True``).
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
    if rdm2_ab is None:
        rdm2_ab = rdm2_aa
    if rdm2_bb is None:
        rdm2_bb = rdm2_aa

    if not elec_props.unrestricted:
        rdm1, rdm2 = spin_summed_rdms(
            rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb, rdm2_notation=rdm2_notation,
        )
        U, e_after, grad_norm = optimize_orbitals_rhf(
            elec_props, rdm1, rdm2,
            method=method, maxiter=maxiter, ftol=ftol, gtol=gtol, rdm2_notation=rdm2_notation,
            use_jax=use_jax, trust_radius=trust_radius, davidson_ref_energy=davidson_ref_energy,
        )
        return U, U, e_after, grad_norm

    norb = elec_props.num_orbitals
    nuc = elec_props.nuclear_repulsion_energy
    h1_a = elec_props.one_body_tensor
    h2_aa = elec_props.two_body_tensor  # chemist's (pq|rs)
    h1_b  = elec_props.one_body_tensor_b
    h2_ab = elec_props.two_body_tensor_ab
    h2_bb = elec_props.two_body_tensor_bb

    rdm2_aa = _to_prqs(rdm2_aa, rdm2_notation)
    rdm2_ab = _to_prqs(rdm2_ab, rdm2_notation)
    rdm2_bb = _to_prqs(rdm2_bb, rdm2_notation)

    n_p = norb * (norb - 1) // 2  # parameters per spin channel
    n_params = n_p if restricted else 2 * n_p
    if restricted:
        log.info("  [OrbOpt] UHF integrals, restricted rotation (Ua = Ub = U)")

    obj_jax, grad_jax, _use_jax = _resolve_jax(
        use_jax, method,
        lambda: _make_jax_uhf_obj_and_grad(
            norb, n_p,
            rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
            h1_a, h1_b, h2_aa, h2_ab, h2_bb, nuc, restricted,
        ),
    )
    if _use_jax:
        eval_obj = obj_jax
    else:
        eval_obj = partial(
            _uhf_energy,
            norb=norb, n_p=n_p,
            rdm1_aa=rdm1_aa, rdm1_bb=rdm1_bb,
            rdm2_aa=rdm2_aa, rdm2_ab=rdm2_ab, rdm2_bb=rdm2_bb,
            h1_a=h1_a, h1_b=h1_b,
            h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
            nuc=nuc, restricted=restricted,
        )

    e_before = eval_obj(np.zeros(n_params, dtype=np.float64))
    log.info(
        "  [OrbOpt] E(trial-RDM; U=I) objective before optimization: %.10f Ha",
        e_before,
    )

    # Diagnostic: decompose E(U=I) into components so the caller can compare with the solver.
    _e1a = float(np.einsum("pq,pq->", h1_a, rdm1_aa))
    _e1b = float(np.einsum("pq,pq->", h1_b, rdm1_bb))
    _e2aa = float(0.5 * np.einsum("pqrs,prqs->", h2_aa, rdm2_aa))
    _e2ab = float(np.einsum("pqrs,prqs->", h2_ab, rdm2_ab))
    _e2bb = float(0.5 * np.einsum("pqrs,prqs->", h2_bb, rdm2_bb))
    _e_check = nuc + _e1a + _e1b + _e2aa + _e2ab + _e2bb
    log.info(
        "  [OrbOpt] E(trial-RDM; U=I) decomposition: nuc=%.6f e1a=%.6f e1b=%.6f "
        "e2aa=%.6f e2ab=%.6f e2bb=%.6f total=%.10f",
        nuc, _e1a, _e1b, _e2aa, _e2ab, _e2bb, _e_check,
    )
    # Trace check: Tr(rdm1_a) should equal N_alpha, Tr(rdm1_b) should equal N_beta
    _tra = float(np.trace(rdm1_aa))
    _trb = float(np.trace(rdm1_bb))
    log.info(
        "  [OrbOpt] RDM1 trace: Tr(rdm1_a)=%.6f (N_a=%d) Tr(rdm1_b)=%.6f (N_b=%d)",
        _tra, elec_props.num_electrons[0], _trb, elec_props.num_electrons[1],
    )
    # --- 2-RDM self-consistency battery (prqs storage: G[p,r,q,s]=<p^dag r^dag s q>) ---
    # Pure array checks (no re-diagonalization). If the energy recon is off but these
    # invariants hold -> defect is in energy/integral contraction; if an invariant is
    # violated -> the 2-RDM block itself is mis-built/mis-normalized (localizes the bug).
    _Na, _Nb = int(elec_props.num_electrons[0]), int(elec_props.num_electrons[1])
    # full double traces: sum_{p,r} G[p,r,p,r]
    _t2aa = float(np.einsum("prpr->", rdm2_aa))
    _t2ab = float(np.einsum("prpr->", rdm2_ab))
    _t2bb = float(np.einsum("prpr->", rdm2_bb))
    log.info(
        "  [OrbOpt] 2RDM trace: aa=%.6f (want Na(Na-1)=%d) ab=%.6f (want Na*Nb=%d) "
        "bb=%.6f (want Nb(Nb-1)=%d)",
        _t2aa, _Na * (_Na - 1), _t2ab, _Na * _Nb, _t2bb, _Nb * (_Nb - 1),
    )
    # partial trace to 1-RDM: sum_r G_aa[p,r,q,r] = (Na-1) gamma_a ; sum_r G_bb[p,r,q,r]=(Nb-1) gamma_b
    # ab over beta pair: sum_r G_ab[p,r,q,r]=Nb gamma_a ; over alpha pair: sum_p G_ab[p,r,p,s]=Na gamma_b
    _pt_aa = np.einsum("prqr->pq", rdm2_aa)
    _pt_bb = np.einsum("prqr->pq", rdm2_bb)
    _pt_ab_b = np.einsum("prqr->pq", rdm2_ab)   # -> Nb * gamma_a
    _pt_ab_a = np.einsum("prps->rs", rdm2_ab)   # -> Na * gamma_b
    _d_aa = float(np.max(np.abs(_pt_aa - (_Na - 1) * rdm1_aa)))
    _d_bb = float(np.max(np.abs(_pt_bb - (_Nb - 1) * rdm1_bb)))
    _d_ab_b = float(np.max(np.abs(_pt_ab_b - _Nb * rdm1_aa)))
    _d_ab_a = float(np.max(np.abs(_pt_ab_a - _Na * rdm1_bb)))
    log.info(
        "  [OrbOpt] 2RDM partial-trace->1RDM max|dev|: aa=%.3e bb=%.3e ab(->Nb.ga)=%.3e "
        "ab(->Na.gb)=%.3e  [0 => 2RDM consistent with 1RDM]",
        _d_aa, _d_bb, _d_ab_b, _d_ab_a,
    )
    _log_self_consistency(e_before, nuc, davidson_ref_energy)
    _maybe_dump_rdms(
        h1_a=h1_a, h1_b=h1_b, h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
        rdm1_aa=rdm1_aa, rdm1_bb=rdm1_bb, rdm2_aa=rdm2_aa, rdm2_ab=rdm2_ab, rdm2_bb=rdm2_bb,
        nuc=float(nuc),
        e_davidson=float(davidson_ref_energy) if davidson_ref_energy is not None else np.nan,
        e_recon=float(e_before) - float(nuc), na=_Na, nb=_Nb,
    )

    x_opt, e_after, grad_norm = _minimize_orbital_objective(
        eval_obj, grad_jax if _use_jax else None, n_params, e_before,
        method=method, maxiter=maxiter, ftol=ftol, gtol=gtol, trust_radius=trust_radius,
    )

    Ua = _unitary_from_skew(x_opt[:n_p], norb)
    Ub = Ua if restricted else _unitary_from_skew(x_opt[n_p:], norb)
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

    The ``t2`` amplitudes are intentionally NOT rotated because they are used only to
    initialize the SQD subspace. ``initial_occupancy`` follows the orbitals when it was taken
    from a 1-RDM (``occupancy_rdm1`` set): the 1-RDMs are rotated as U^T gamma U and the new
    occupancies are their diagonal. The HF 1/0 seed (``occupancy_rdm1`` None) is kept as is.
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

    if elec_props.occupancy_rdm1 is not None:
        g_a, g_b = (np.asarray(g) for g in elec_props.occupancy_rdm1)
        occupancy_rdm1 = (rot1(g_a, Ua), rot1(g_b, Ub if elec_props.unrestricted else Ua))
        initial_occupancy = (np.diag(occupancy_rdm1[0]).copy(), np.diag(occupancy_rdm1[1]).copy())
    else:
        occupancy_rdm1 = None
        initial_occupancy = elec_props.initial_occupancy

    return ElectronicProperties(
        one_body_tensor=h1_a_new,
        two_body_tensor=h2_aa_new,
        t2=elec_props.t2,
        initial_occupancy=initial_occupancy,
        occupancy_rdm1=occupancy_rdm1,
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
        # <phi^alpha_new | phi^beta_new> = Ua^T <phi^alpha | phi^beta> Ub (identity if unset)
        ab_overlap=(
            Ua.T @ (np.eye(Ua.shape[0]) if elec_props.ab_overlap is None
                    else np.asarray(elec_props.ab_overlap)) @ Ub
            if elec_props.unrestricted else None
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — spin_square_from_rdms
# ─────────────────────────────────────────────────────────────────────────────

def spin_square_from_rdms(
    rdm1_a: np.ndarray,
    rdm1_b: np.ndarray,
    rdm2_aa: np.ndarray,
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    ab_overlap: np.ndarray | None = None,
) -> float:
    """<S^2> of the state whose RDMs the native SBD solver wrote (``do_rdm != 0``).

    Inputs are in the solver's file layout as read by ``sbd.solver_job._read_rdm_file``: rdm1[p, q]
    = <q^dag p> and every 2-RDM block (aa, ab, bb) in "prqs" storage, rdm2[p, r, q, s] =
    <p^dag r^dag s q> (alpha index first in the ab block), the same storage ``optimize_orbitals``
    consumes with ``rdm2_notation="prqs"``. One transpose (0,2,1,3) maps each block to PySCF's
    ``make_rdm12s`` layout, which goes to PySCF's ``spin_square_general``. (``main.cc`` writes the
    native flat r[io + L jo + L^2 ia + L^3 ja] in row-major (io, jo, ia, ja) order, so the file
    element [i, j, k, l] is that native value; the test pins this against PySCF through a replica
    of the patched native kernel.) ``ab_overlap`` = <phi_p^alpha | phi_q^beta> of the orbitals the RDMs
    are expressed in (ElectronicProperties.ab_overlap); None = identity (alpha and beta orbitals
    coincide). For a UHF orbital basis the identity is WRONG, because S_+ S_- couples alpha and
    beta orbitals through their overlap.
    """
    from pyscf.fci.spin_op import spin_square_general

    n = np.asarray(rdm1_a).shape[0]
    s_ab = np.eye(n) if ab_overlap is None else np.asarray(ab_overlap, dtype=np.float64)
    # spin_square_general builds every overlap block from (mo_a, mo_b) and an AO overlap. Use an
    # orthonormal "AO" basis = the alpha orbitals, so mo_a = I and mo_b = S_ab. The beta-beta block
    # mo_b^T mo_b must then be the identity, i.e. S_ab must be orthogonal (true for a rotation of
    # orthonormal alpha/beta sets spanning the same space).
    dev = float(np.max(np.abs(s_ab.T @ s_ab - np.eye(n))))
    if dev > 1e-6:
        raise ValueError(f"ab_overlap is not orthogonal (max |S^T S - 1| = {dev:.2e})")
    ss, _ = spin_square_general(
        np.asarray(rdm1_a), np.asarray(rdm1_b),
        np.asarray(rdm2_aa).transpose(0, 2, 1, 3),
        np.asarray(rdm2_ab).transpose(0, 2, 1, 3),
        np.asarray(rdm2_bb).transpose(0, 2, 1, 3),
        (np.eye(n), s_ab), 1,
    )
    return float(ss)


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
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-7,
    trust_radius: float = 0.1,
    oo_maxiter: int = 40,
    resolve_backend: str = "solve_fermion",
    davidson_solver=None,
    use_jax: bool | None = None,
    restricted: bool = False,
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

    RHF integrals take the RHF orbital step (one rotation, spin-summed RDMs; see
    :func:`optimize_orbitals_rhf`); for UHF integrals ``restricted=True`` uses one rotation for
    both spins (see :func:`optimize_orbitals`).

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

    # ── Re-diagonalization backend selection ────────────────────────────────────────────────
    # The fixed subspace is re-solved in the current (rotated) basis by either the in-process CPU
    # solve_fermion (default) or the native GPU Davidson (SBDSolverJob) -- the SAME solver the main
    # SQD loop uses. The GPU path uses the FULL UHF integral blocks (h1_a/h1_b/aa/ab/bb) and returns
    # RDMs in prqs-storage (matching solver_job.py); solve_fermion uses alpha integrals only and
    # returns pqrs-storage. Hence rdm2_notation differs per backend.
    _resolve_notation = "prqs" if resolve_backend == "davidson_gpu" else "pqrs"
    if resolve_backend == "davidson_gpu":
        if davidson_solver is None:
            raise ValueError(
                "resolve_backend='davidson_gpu' requires davidson_solver=<SBDSolverJob>."
            )
        if not getattr(davidson_solver, "do_rdm", 0):
            raise ValueError(
                "resolve_backend='davidson_gpu' requires the solver block do_rdm>=1 "
                "(the re-solve needs full RDMs); got do_rdm=0."
            )
        import asyncio
        _na, _nb = (num_elec if num_elec is not None else elec_props.num_electrons)
        _norb = int(np.asarray(elec_props.one_body_tensor).shape[0])

    def _resolve_diagonalize(ci_pair, ep_cur, *, want_rdms):
        """Re-diagonalize the fixed subspace `ci_pair` in basis `ep_cur`.

        Returns (E_total, rdms) where rdms is (rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb) for UHF
        integrals and the spin-summed (rdm1, rdm2) for RHF integrals, in `_resolve_notation`
        storage, when want_rdms, else None.
        """
        if resolve_backend == "davidson_gpu":
            # Native GPU Davidson on the fixed subspace (no sampling), full UHF integral blocks.
            # NOTE: even want_rdms=False (line-search energy) evals compute RDMs here because the
            # block's do_rdm>=1; the RDM cost is negligible at validation norb. A do_rdm=0 solver
            # copy for the energy-only evals is a possible future micro-opt.
            r = asyncio.run(davidson_solver.run(
                ci_strings=(ci_pair[0], ci_pair[1]),
                one_body_tensor=ep_cur.one_body_tensor,
                two_body_tensor=ep_cur.two_body_tensor,
                norb=_norb, nelec=(_na, _nb),
                one_body_tensor_b=ep_cur.one_body_tensor_b,
                two_body_tensor_ab=ep_cur.two_body_tensor_ab,
                two_body_tensor_bb=ep_cur.two_body_tensor_bb,
            ))
            E = float(r.energy) + nuc
            if not want_rdms:
                return E, None
            if not open_shell:
                return E, (r.rdm1 + r.rdm1_b, r.rdm2_spin_summed)
            return E, (r.rdm1, r.rdm1_b, r.rdm2, r.rdm2_ab, r.rdm2_bb)
        # default: in-process CPU solve_fermion (alpha integrals; pqrs-storage RDMs).
        e_el, sci, _occ, _s2 = solve_fermion(
            ci_pair, ep_cur.one_body_tensor, ep_cur.two_body_tensor, open_shell=open_shell,
        )
        E = float(e_el) + nuc
        if not want_rdms:
            return E, None
        if not open_shell:
            return E, (sci.rdm(rank=1, spin_summed=True), sci.rdm(rank=2, spin_summed=True))
        r1 = sci.rdm(rank=1, spin_summed=False)
        r2 = sci.rdm(rank=2, spin_summed=False)
        return E, (r1[0], r1[1], r2[0], r2[1], r2[-1])

    ep = elec_props
    prev_e = None
    e = float("nan")
    grad_norm = float("inf")
    macro = 0
    for macro in range(1, max_macro + 1):
        # (1)+(2) re-diagonalize the fixed subspace in the current basis -> fresh energy + RDMs.
        # solve_fermion returns (energy, SCIState, avg_occupancies, spin_sq); SCIState.rdm(rank,
        # spin_summed=False) gives the per-spin blocks in pqrs-storage.
        e, _rdms = _resolve_diagonalize(ci, ep, want_rdms=True)

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
            _step_kwargs = dict(
                rdm2_notation=_resolve_notation, trust_radius=tr, maxiter=oo_maxiter, use_jax=use_jax,
                gtol=grad_tol,  # inner L-BFGS-B pgtol unified with the SC macro tol (OO_GRAD_TOL)
            )
            if open_shell:
                Ua, Ub, e_oo, grad_norm = optimize_orbitals(
                    ep, *_rdms, restricted=restricted, **_step_kwargs,
                )
            else:
                Ua, e_oo, grad_norm = optimize_orbitals_rhf(ep, *_rdms, **_step_kwargs)
                Ub = Ua
            ep_try = rotate_electronic_properties(ep, Ua, Ub)
            e_try, _ = _resolve_diagonalize(ci, ep_try, want_rdms=False)
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
    e_final, _ = _resolve_diagonalize(ci, ep, want_rdms=True)
    return ep, float(e_final), float(grad_norm), macro
