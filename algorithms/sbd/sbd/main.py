# Workflow for observability demo on Miyabi

import os
import pathlib
from typing import Self

import numpy as np
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import RUN_ID, Inputs
from prefect.futures import PrefectFutureList
from prefect.task_runners import ConcurrentTaskRunner
from prefect_ray import RayTaskRunner
from pydantic import BaseModel, Field
from qcsc_workflow_utility.chem import (
    ElectronicProperties,
    NpStrict1DArrayF64,
    NpStrict2DArrayF64,
    compute_molecular_integrals_from_fcidump,
    refresh_cc_seed,
)
from qcsc_workflow_utility.orbital_opt import optimize_orbitals, rotate_electronic_properties

from .data_io import extend_table_artifact
from .flow_params import FlowParameters
from .lucj import initialize_ucj_parameters
from .np_type_extension import NpStrict2DArrayBool
from .solver_job import SBDSolverJob
from .sqd import walker_sqd


def _apply_cc_refresh_and_reseed(elec_props, state, logger, trial_index):
    """After OO rotation, refresh CC seed and reset DE population/carryover.

    Called only when OO saturation is detected (determinant-space limited).
    """
    logger.info("Trial %d: OO saturated -> running CC refresh on rotated Hamiltonian...", trial_index)
    refreshed = refresh_cc_seed(elec_props)
    if refreshed is None:
        logger.warning("Trial %d: CC refresh failed. Keeping old t2/occupancy.", trial_index)
        return elec_props, state

    old_t2_norm = np.linalg.norm(np.asarray(elec_props.t2))
    new_t2_norm = np.linalg.norm(np.asarray(refreshed.t2))
    logger.info(
        "Trial %d: CC refresh done. ||t2_old||=%.4e -> ||t2_new||=%.4e",
        trial_index, old_t2_norm, new_t2_norm,
    )

    state.best_index = None
    state.energies[:] = 0.0
    state.carryover = np.full((0, refreshed.num_orbitals), False, dtype=bool)
    logger.info("Trial %d: DE population reset (best_index=None, carryover cleared) for re-seed.", trial_index)

    return refreshed, state


MODULE_RNG = np.random.default_rng(seed=4574)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _build_task_runner():
    mode = os.getenv("SBD_TASK_RUNNER", "ray").strip().lower()
    if mode == "concurrent":
        return ConcurrentTaskRunner()

    # Keep legacy behavior when PREFECT_RAY_NUM_CPUS is not explicitly set.
    ray_cpus_raw = os.getenv("PREFECT_RAY_NUM_CPUS", "").strip()
    if not ray_cpus_raw:
        return RayTaskRunner

    for key, value in THREAD_ENV.items():
        os.environ.setdefault(key, value)

    ray_cpus = int(ray_cpus_raw)
    return RayTaskRunner(
        init_kwargs={
            "num_cpus": ray_cpus,
            "runtime_env": {"env_vars": THREAD_ENV},
        }
    )


def _build_ab_indices(norb: int, stride: int) -> list[tuple[int, int]]:
    """Alpha-beta LUCJ coupling pairs (p, p), ordered by priority (most important first).

    On heavy-hex hardware the ab couplings are ancilla-mediated and can't all be realized, so
    ffsim's generate_lucj_pass_manager drops pairs from the END of the list until the layout
    fits. We therefore emit the stock stride-4 anchors FIRST (they always fit on heavy-hex),
    then the densification pairs (the extra p's a smaller stride adds) AFTER. This way a denser
    request degrades gracefully back toward the stock coupling instead of losing anchor pairs.
    stride=4 reproduces the historical [(p, p) for p in range(0, norb, 4)] exactly.
    """
    stride = max(1, int(stride))
    anchors = [(p, p) for p in range(0, norb, 4)]
    seen = {p for p, _ in anchors}
    extra = [(p, p) for p in range(0, norb, stride) if p not in seen]
    return anchors + extra


class OptimizerState(BaseModel):
    """Intermediate data for optimization."""

    energies: NpStrict1DArrayF64
    populations: NpStrict2DArrayF64
    carryover: NpStrict2DArrayBool
    best_index: int | None = Field(
        default=None,
        ge=0,
    )

    def best_energy(self) -> float | None:
        if self.best_index is None:
            return None
        return float(self.energies[self.best_index])

    def copy(self) -> Self:
        return OptimizerState(
            energies=self.energies.copy(),
            populations=self.populations.copy(),
            carryover=self.carryover.copy(),
            best_index=self.best_index,
        )

    @classmethod
    def from_parameters(
        cls,
        num_walkers: int,
        norb: int,
        n_aa_params: int,
        n_ab_params: int,
        n_reps: int,
    ) -> "OptimizerState":
        num_lucj_params = n_reps * (n_aa_params + n_ab_params + norb**2) + norb**2
        return OptimizerState(
            energies=np.zeros(num_walkers, dtype=np.float64),
            populations=np.full((num_walkers, num_lucj_params), np.nan, dtype=np.float64),
            carryover=np.full((0, norb), np.nan, dtype=bool),
        )


@flow(
    task_runner=_build_task_runner(),
)
def riken_sqd_de(
    parameters: FlowParameters,
):
    logger = get_run_logger()
    logger.info("Task runner mode: %s", os.getenv("SBD_TASK_RUNNER", "ray").strip().lower())

    # ★ fail-fast: solver block existence & sanity check
    slug, name = parse_block_ref(parameters.solver_block_ref)
    if slug != "sbd_solver_job":
        raise ValueError(
            f"solver_block_ref must be 'sbd_solver_job/<name>'. got: {parameters.solver_block_ref}"
        )

    try:
        solver = SBDSolverJob.load(name)
    except Exception:
        logger.exception("Failed to load solver block: %s", parameters.solver_block_ref)
        raise

    logger.info(
        "Solver OK: ref=%s mode=%s",
        parameters.solver_block_ref,
        getattr(solver, "solver_mode", "unknown"),
    )

    telemetry_data = []
    create_table_artifact(
        table=telemetry_data,
        key="sqd-telemetry",
        description="SQD intermediate data.",
    )

    # The solver block is the single source of truth for RHF vs UHF; drive the classical
    # integral computation (and the rest of the open-shell pipeline) from solver.method.
    unrestricted = getattr(solver, "method", "rhf") == "uhf"
    logger.info("Electronic-structure method: %s", "uhf" if unrestricted else "rhf")

    # Orbital optimization runs between DE trials when the solver writes full RDMs (do_rdm != 0):
    # the best-walker RDMs rotate the Hamiltonian integrals so the next trial starts from an
    # improved orbital basis (MCSCF-style two-step optimization). Needs iterations >= 2 to have any
    # effect. See qcsc_workflow_utility.orbital_opt (Kreplin/Knowles/Werner, JCP 152, 074102 (2020)).
    do_orbital_opt = getattr(solver, "do_rdm", 0) != 0
    logger.info(
        "Orbital optimization: %s (do_rdm=%d, iterations=%d)",
        "enabled" if do_orbital_opt else "disabled",
        getattr(solver, "do_rdm", 0), parameters.de_params.iterations,
    )

    elec_props = compute_molecular_integrals_from_fcidump(
        fcidump_file=parameters.fcidump,
        unrestricted=unrestricted,
    )

    # We assume heavy-hex topology
    # Orbitals for different spins have connections between every 4th orbital.
    aa_indices = [(p, p + 1) for p in range(elec_props.num_orbitals - 1)]
    ab_indices = _build_ab_indices(
        elec_props.num_orbitals, parameters.circ_params.ab_stride
    )
    logger.info(
        "LUCJ alpha-beta coupling: stride=%s -> %s pairs (stock stride-4 would give %s)",
        parameters.circ_params.ab_stride,
        len(ab_indices),
        len(range(0, elec_props.num_orbitals, 4)),
    )

    state = OptimizerState.from_parameters(
        num_walkers=parameters.de_params.num_walkers,
        norb=elec_props.num_orbitals,
        n_aa_params=len(aa_indices),
        n_ab_params=len(ab_indices),
        n_reps=parameters.circ_params.n_lucj_layers,
    )

    # OO effect tracking: store the prediction from the previous trial's OO
    # so the next trial can compare prediction vs actual Davidson energy. Store
    # all values here as total energies (including nuclear repulsion).
    _oo_prediction = None  # dict with trial, e_davidson_source, e_oo_trial_rdm or e_oo_sc

    # CC refresh stagnation trigger: count consecutive trials where OO was skipped
    # (best not beaten). When the count reaches OO_SAT_STAGNATION (default 2),
    # fire CC refresh to break out of the determinant-space bottleneck.
    _consecutive_oo_skips = 0

    # Start differential evoluation
    for i in range(parameters.de_params.iterations):
        logger.info(f"Running differential evolution trial {i}")

        state, best_sbd_result = differential_evolution_trial(
            trial_index=i,
            parameters=parameters,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            state=state,
        )

        logger.info(f"Current best energy = {state.best_energy()} (walker {state.best_index})")

        # ── OO effect tracking: compare previous OO prediction with this trial's actual energy ──
        # Also detect OO saturation: when Davidson improvement is small but the prediction gap
        # (= determinant-space limitation) is large, the circuit needs updating via CC refresh.
        if _oo_prediction is not None:
            _prev = _oo_prediction
            _e_actual = (
                float(best_sbd_result.energy) + float(elec_props.nuclear_repulsion_energy)
                if best_sbd_result is not None else None
            )
            if _e_actual is not None:
                _delta_actual = (_e_actual - _prev["e_davidson_source"]) * 1000  # mHa
                _delta_pred = (_prev["e_oo_estimate"] - _prev["e_davidson_source"]) * 1000
                _gap = (_e_actual - _prev["e_oo_estimate"]) * 1000  # positive = underperformance
                logger.info(
                    "Trial %d: OO effect (from trial %d):\n"
                    "  E_davidson(source, trial %d) [Davidson-GPU, truncated CI, total]:  %.10f\n"
                    "  E_oo(%s prediction)          [%s; total]:  %.10f  (predicted dE=%.1f mHa)\n"
                    "  E_davidson(actual, trial %d)  [Davidson-GPU, truncated CI, total]:  %.10f  (actual dE=%.1f mHa)\n"
                    "  Prediction gap: %.1f mHa (%s)",
                    i, _prev["trial"],
                    _prev["trial"], _prev["e_davidson_source"],
                    _prev["method"], _prev["method_detail"], _prev["e_oo_estimate"], _delta_pred,
                    i, _e_actual, _delta_actual,
                    _gap, "overestimate" if _gap < 0 else "underestimate" if _gap > 0 else "exact",
                )


            _oo_prediction = None

        # ── Orbital optimization (between DE trials) ────────────────────────────
        # Rotate the Hamiltonian integrals using the best walker's RDMs so the next trial starts
        # from an improved orbital basis. Guarded by do_rdm; a hard self-consistency gate checks
        # that the energy rebuilt from the read RDMs (at U=I) matches the solver's Davidson energy
        # before trusting the rotation.
        if do_orbital_opt and best_sbd_result is not None:
            # Only run OO when this trial produced a new all-time best Davidson energy.
            # A worse RDM drives the orbital gradient in a non-improving direction; skipping
            # OO preserves the current basis until a better state arrives.
            # Disable with OO_SKIP_NONBEST=0 to recover the old (always-fire) behavior.
            _skip_nonbest = os.environ.get("OO_SKIP_NONBEST", "1") != "0"
            _rdm_e = best_sbd_result.energy
            _best_e = state.best_energy()
            _is_new_best = (
                _rdm_e is not None and _best_e is not None
                and abs(_rdm_e - _best_e) < 1e-12
            )
            if _skip_nonbest and not _is_new_best:
                _consecutive_oo_skips += 1
                _oo_to_lucj = int(os.environ.get("OO_TO_LUCJ", "0"))
                _stagnation_n = int(os.environ.get("OO_SAT_STAGNATION", "2"))
                if _oo_to_lucj == 1 and _consecutive_oo_skips >= _stagnation_n:
                    logger.info(
                        "Trial %d: Davidson %.6f did not beat best %.6f "
                        "(%d consecutive skips >= %d) -> firing CC refresh to break stagnation.",
                        i, _rdm_e if _rdm_e is not None else float("nan"),
                        _best_e if _best_e is not None else float("nan"),
                        _consecutive_oo_skips, _stagnation_n,
                    )
                    elec_props, state = _apply_cc_refresh_and_reseed(elec_props, state, logger, i)
                    _consecutive_oo_skips = 0
                else:
                    logger.info(
                        "Trial %d: Davidson %.6f did not beat best %.6f; skipping OO "
                        "(consecutive skips: %d/%d).",
                        i, _rdm_e if _rdm_e is not None else float("nan"),
                        _best_e if _best_e is not None else float("nan"),
                        _consecutive_oo_skips, _stagnation_n,
                    )
                continue  # skip to next trial

            rdm1_aa = best_sbd_result.rdm1
            rdm2_aa = best_sbd_result.rdm2
            if rdm1_aa is not None and rdm2_aa is not None:
                rdm1_bb = best_sbd_result.rdm1_b if best_sbd_result.rdm1_b is not None else rdm1_aa
                rdm2_ab = best_sbd_result.rdm2_ab if best_sbd_result.rdm2_ab is not None else rdm2_aa
                rdm2_bb = best_sbd_result.rdm2_bb if best_sbd_result.rdm2_bb is not None else rdm2_aa
                _consecutive_oo_skips = 0  # OO fires -> reset stagnation counter
                _nuc = float(elec_props.nuclear_repulsion_energy)
                logger.info(
                    "Trial %d: running orbital optimization (norb=%d, unrestricted=%s, "
                    "E_davidson source: elec=%.10f  total(+nuc)=%.10f  (nuc=%.6f); "
                    "RDM frozen only within this OO step) ...",
                    i, elec_props.num_orbitals, unrestricted,
                    float(_rdm_e), float(_rdm_e) + _nuc, _nuc,
                )

                # ── Self-consistent path (oo_resolve_rdms): re-diagonalize the fixed CI subspace
                # in the rotated basis each orbital step (fresh RDMs) so the gradient is the TRUE
                # MCSCF gradient and convergence is meaningful/variational. Requires the subspace
                # (alpha/beta determinant lists) carried on the SBDResult. Falls back to the
                # fixed-RDM path if unavailable.
                if getattr(parameters, "oo_resolve_rdms", False) and (
                    getattr(best_sbd_result, "alphadets", None) is not None
                ):
                    try:
                        from qcsc_workflow_utility.orbital_opt import (
                            resolve_orbitals_self_consistent,
                        )
                        _elec_props_before_oo = elec_props
                        elec_props, e_sc, grad_sc, n_macro = resolve_orbitals_self_consistent(
                            elec_props,
                            best_sbd_result.alphadets,
                            best_sbd_result.betadets,
                            num_elec=elec_props.num_electrons,
                            resolve_maxdim=getattr(parameters, "oo_resolve_maxdim", 4_000_000),
                            grad_tol=(
                                float(os.environ["OO_GRAD_TOL"]) if "OO_GRAD_TOL" in os.environ
                                else getattr(parameters, "oo_grad_tol", 1e-3)
                            ),
                            trust_radius=(
                                float(os.environ["OO_TRUST"]) if "OO_TRUST" in os.environ
                                else getattr(parameters, "oo_trust_radius", 0.1)
                            ),
                            # NOTE: oo_maxiter here is MACRO iterations (self-consistent re-diag),
                            # a different quantity from the L-BFGS inner maxiter unified via
                            # OO_MAXITER; left on its own param/default deliberately.
                            oo_maxiter=getattr(parameters, "oo_maxiter", 40),
                            resolve_backend=getattr(parameters, "oo_resolve_backend", "solve_fermion"),
                            davidson_solver=solver,
                            logger=logger,
                        )
                        logger.info(
                            "Trial %d: self-consistent OO converged E=%.10f Ha |grad|=%.3e "
                            "(%d macro-iters). Hamiltonian rotated for next trial.",
                            i, e_sc, grad_sc, n_macro,
                        )
                        # CC refresh (if needed) is now triggered by stagnation detection above
                        _oo_prediction = {
                            "trial": i,
                            "e_davidson_source": float(_rdm_e) + float(elec_props.nuclear_repulsion_energy),
                            "e_oo_estimate": float(e_sc),
                            "method": "OO-SC",
                            "method_detail": "JAX orbital step + fresh RDM each macro-step",
                        }

                        # OO_CHECK: the OO-SC orbital step may have converged on a truncated
                        # subspace. Re-diagonalize the full original determinant subspace with
                        # the rotated H before leaving this branch, so OO_RESOLVE does not hide
                        # the full-space energy after truncation.
                        if int(os.environ.get("OO_CHECK", "0")) == 1:
                            _check_adets = getattr(best_sbd_result, "alphadets", None)
                            _check_bdets = getattr(best_sbd_result, "betadets", None)
                            if _check_adets is not None:
                                try:
                                    import asyncio as _asyncio
                                    _check_r = _asyncio.run(solver.run(
                                        ci_strings=(
                                            _check_adets,
                                            _check_bdets
                                            if _check_bdets is not None
                                            else _check_adets,
                                        ),
                                        one_body_tensor=elec_props.one_body_tensor,
                                        two_body_tensor=elec_props.two_body_tensor,
                                        norb=elec_props.num_orbitals,
                                        nelec=elec_props.num_electrons,
                                        one_body_tensor_b=elec_props.one_body_tensor_b,
                                        two_body_tensor_ab=elec_props.two_body_tensor_ab,
                                        two_body_tensor_bb=elec_props.two_body_tensor_bb,
                                    ))
                                    _e_check = float(_check_r.energy)
                                    logger.info(
                                        "Trial %d: OO_CHECK [Davidson-GPU, full original subspace, "
                                        "rotated H] (all energies total, incl. nuclear repulsion):\n"
                                        "  E_before = E_davidson before OO                           : %.10f\n"
                                        "  E_sc     = truncated OO-SC energy (fresh RDM per macro-step): %.10f  "
                                        "(different subspace/definition -- not directly comparable)\n"
                                        "  E_after  = E_davidson after OO (full subspace, rotated H)  : %.10f\n"
                                        "  true OO effect = E_after - E_before = %+.4f mHa",
                                        i,
                                        float(_rdm_e) + float(_elec_props_before_oo.nuclear_repulsion_energy),
                                        e_sc,
                                        _e_check + float(elec_props.nuclear_repulsion_energy),
                                        (_e_check - _rdm_e) * 1000,
                                    )
                                    if _e_check > _rdm_e:
                                        elec_props = _elec_props_before_oo
                                        _oo_prediction = None
                                        logger.warning(
                                            "Trial %d: OO_CHECK energy increased by %.3e Ha "
                                            "-> rolling back OO-SC rotation; keeping pre-OO integrals.",
                                            i, _e_check - _rdm_e,
                                        )
                                        continue
                                except Exception:
                                    logger.exception("Trial %d: OO_CHECK failed.", i)

                        _grad_tol_sc = (
                            float(os.environ["OO_GRAD_TOL"]) if "OO_GRAD_TOL" in os.environ
                            else getattr(parameters, "oo_grad_tol", 1e-3)
                        )
                        if grad_sc < _grad_tol_sc and not getattr(parameters, "oo_refire_every_trial", False):
                            logger.info(
                                "Trial %d: orbitals stationary (|grad| < tol) -> freezing basis.", i
                            )
                            do_orbital_opt = False
                    except Exception:
                        logger.exception(
                            "Trial %d: self-consistent OO failed; keeping current integrals.", i
                        )
                    continue  # skip the fixed-RDM path below

                try:
                    Ua, Ub, e_opt, grad_norm = optimize_orbitals(
                        elec_props=elec_props,
                        rdm1_aa=rdm1_aa,
                        rdm1_bb=rdm1_bb,
                        rdm2_aa=rdm2_aa,
                        rdm2_ab=rdm2_ab,
                        rdm2_bb=rdm2_bb,
                        # The native solver (main.cc) and solver_job.py both write the 2-RDM in
                        # prqs-storage (rdm2[p,r,q,s]=<p^dag r^dag s q>). optimize_orbitals defaults
                        # to "pqrs"; passing "pqrs"-stored data as prqs (or vice versa) applies a
                        # wrong transpose -> unphysical energy (~-159 Ha for OH). Must be "prqs".
                        rdm2_notation="prqs",
                        # Shared OO thresholds: OO_TRUST / OO_MAXITER env override the flow
                        # params so the initial OO (chem._apply_initial_oo) and this DE-loop OO
                        # use identical criteria. Env absent -> unchanged (backward compatible).
                        trust_radius=(
                            float(os.environ["OO_TRUST"]) if "OO_TRUST" in os.environ
                            else getattr(parameters, "oo_trust_radius", 0.5)
                        ),
                        maxiter=(
                            int(os.environ["OO_MAXITER"]) if "OO_MAXITER" in os.environ
                            else getattr(parameters, "oo_maxiter", 300)
                        ),
                        davidson_ref_energy=float(_rdm_e) if _rdm_e is not None else None,
                    )
                    e_solver = float(state.best_energy()) if state.best_energy() is not None else None
                    _nuc = float(elec_props.nuclear_repulsion_energy)
                    logger.info(
                        "Trial %d: E_oo(trial-RDM; frozen within OO step) "
                        "[L-BFGS-B objective on rotated H] = %.10f Ha (total)  "
                        "|grad|=%.3e  (E_davidson [Davidson-GPU]: elec=%.10f "
                        "total(+nuc)=%.10f  best-so-far(total) = %s)",
                        i, e_opt, grad_norm,
                        float(_rdm_e), float(_rdm_e) + _nuc,
                        f"{e_solver:.10f}" if e_solver is not None else "n/a",
                    )

                    # ── Two-step MCSCF stopping logic (reference-free, CASSCF-style) ──────────
                    # (1) Macro-convergence: orbital gradient ~0 => orbitals are stationary
                    #     (generalized Brillouin condition, the criterion CASSCF codes use). No
                    #     external DMRG/FCI floor needed -> valid for large systems.
                    # (2) Self-consistency guard: the orbital-optimization energy is computed on
                    #     the current trial's RDMs, held fixed only during this OO call. The next
                    #     Davidson trial refreshes the RDMs. If it runs far below the solver
                    #     energy of the SAME state, the trial-RDM objective has decoupled from the
                    #     rotated Hamiltonian (non-variational artifact). Detect that divergence
                    #     and stop rotating rather than propagate a spurious basis.
                    oo_gtol = (
                        float(os.environ["OO_GRAD_TOL"]) if "OO_GRAD_TOL" in os.environ
                        else getattr(parameters, "oo_grad_tol", 1e-3)
                    )
                    oo_sc_tol = getattr(parameters, "oo_selfconsistency_tol", 0.05)  # 50 mHa
                    diverged = (e_solver is not None) and (e_opt < e_solver - oo_sc_tol)
                    if diverged:
                        logger.warning(
                            "Trial %d: OO energy %.6f is %.1f mHa below the solver energy %.6f "
                            "-> trial-RDM objective decoupling (non-variational). NOT rotating; stopping OO.",
                            i, e_opt, (e_solver - e_opt) * 1000.0, e_solver,
                        )
                        do_orbital_opt = False  # freeze the basis; keep running DE without OO
                    else:
                        _elec_props_before_oo = elec_props
                        elec_props = rotate_electronic_properties(elec_props, Ua, Ub)
                        logger.info("Trial %d: Hamiltonian rotated for next trial.", i)

                        # OO_CHECK: re-diagonalize the SAME determinant subspace with the
                        # ROTATED H to get the true energy after OO (without re-sampling).
                        # This isolates the OO effect from the re-sampling effect.
                        if int(os.environ.get("OO_CHECK", "0")) == 1:
                            _check_adets = getattr(best_sbd_result, "alphadets", None)
                            _check_bdets = getattr(best_sbd_result, "betadets", None)
                            if _check_adets is not None:
                                try:
                                    import asyncio as _asyncio
                                    _check_r = _asyncio.run(solver.run(
                                        ci_strings=(_check_adets, _check_bdets if _check_bdets is not None else _check_adets),
                                        one_body_tensor=elec_props.one_body_tensor,
                                        two_body_tensor=elec_props.two_body_tensor,
                                        norb=elec_props.num_orbitals,
                                        nelec=elec_props.num_electrons,
                                        one_body_tensor_b=elec_props.one_body_tensor_b,
                                        two_body_tensor_ab=elec_props.two_body_tensor_ab,
                                        two_body_tensor_bb=elec_props.two_body_tensor_bb,
                                    ))
                                    _e_check = float(_check_r.energy)
                                    logger.info(
                                        "Trial %d: OO_CHECK [Davidson-GPU, same subspace, rotated H] "
                                        "(all energies total, incl. nuclear repulsion):\n"
                                        "  E_before = E_davidson before OO                          : %.10f\n"
                                        "  E_oo     = fixed trial-RDM L-BFGS objective on rotated H  : %.10f  "
                                        "(different definition; may dip below E_before via RDM/H decoupling -- not directly comparable)\n"
                                        "  E_after  = E_davidson after OO (same subspace, rotated H) : %.10f\n"
                                        "  true OO effect = E_after - E_before = %+.4f mHa",
                                        i,
                                        float(_rdm_e) + float(_elec_props_before_oo.nuclear_repulsion_energy),
                                        e_opt,
                                        _e_check + float(elec_props.nuclear_repulsion_energy),
                                        (_e_check - _rdm_e) * 1000,
                                    )
                                    if _e_check > _rdm_e:
                                        elec_props = _elec_props_before_oo
                                        _oo_prediction = None
                                        logger.warning(
                                            "Trial %d: OO_CHECK energy increased by %.3e Ha "
                                            "-> rolling back OO rotation; keeping pre-OO integrals.",
                                            i, _e_check - _rdm_e,
                                        )
                                        continue
                                except Exception:
                                    logger.exception("Trial %d: OO_CHECK failed.", i)

                        # CC refresh (if needed) is now triggered by stagnation detection above
                        _oo_prediction = {
                            "trial": i,
                            "e_davidson_source": float(_rdm_e) + float(elec_props.nuclear_repulsion_energy),
                            "e_oo_estimate": float(e_opt),
                            "method": "trial-RDM",
                            "method_detail": "L-BFGS-B objective; trial RDM frozen within OO step",
                        }
                        # NOT a true OO energy gain. e_solver is the best-so-far Davidson
                        # eigenvalue (variational, on the truncated CI subspace); e_opt is the
                        # fixed-trial-RDM L-BFGS objective on the rotated H. Their difference is the
                        # fixed-RDM / rotated-H DECOUPLING GAP -- how far the objective sits below
                        # the variational energy -- not an energy change. The true post-OO energy on
                        # the same subspace is the OO_CHECK line above (E_after - E_before). The
                        # basis is frozen on the orbital-gradient (Brillouin) criterion, not this gap.
                        objective_gap = (e_opt - e_solver) if e_solver is not None else None
                        logger.info(
                            "Trial %d: OO objective vs solver energy (decoupling diagnostic, NOT a "
                            "true gain): E_oo(fixed trial-RDM, L-BFGS, total)=%.10f minus "
                            "E_solver(best-so-far, Davidson, total)=%s = %s Ha "
                            "(negative => objective dipped below the variational energy; see OO_CHECK "
                            "for the real post-OO energy). |grad|=%.3e (oo_gtol=%.1e).",
                            i, e_opt,
                            f"{e_solver:.10f}" if e_solver is not None else "n/a",
                            f"{objective_gap:+.3e}" if objective_gap is not None else "n/a",
                            grad_norm, oo_gtol,
                        )
                        if grad_norm < oo_gtol and not getattr(parameters, "oo_refire_every_trial", False):
                            logger.info(
                                "Trial %d: |grad|=%.3e < oo_gtol=%.1e -> orbitals stationary "
                                "(generalized Brillouin condition). Freezing basis.",
                                i, grad_norm, oo_gtol,
                            )
                            do_orbital_opt = False
                except Exception:
                    logger.exception(
                        "Trial %d: orbital optimization failed; keeping current integrals.", i
                    )
            else:
                logger.info(
                    "Trial %d: RDMs not available in SBDResult (rdm1=%s); skipping orbital opt. "
                    "Is the solver writing rdm*.txt (do_rdm != 0 + a binary that emits them)?",
                    i, "None" if rdm1_aa is None else "present",
                )

    return state.best_energy()


@task(
    task_run_name="de_trial#{trial_index:02d}",
    # Cache on the flow run ID and trial_index.
    # This is roughly identical with the conventional checkpoint mechanism.
    cache_policy=Inputs(
        exclude=[
            "parameters",
            "elec_props",
            "aa_indices",
            "ab_indices",
            "state",
        ]
    )
    + RUN_ID,
)
def differential_evolution_trial(
    trial_index: int,
    parameters: FlowParameters,
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    state: OptimizerState,
) -> tuple[OptimizerState, "SBDResult | None"]:
    """Run one DE trial. Returns (new_state, best_sbd_result), where best_sbd_result is the
    SBDResult (carrying RDMs, for orbital optimization) of the lowest-energy walker (or None)."""
    from .solver_job import SBDResult  # noqa: F401  (type only; avoids import cycle at module load)

    logger = get_run_logger()

    if state.best_index is not None:
        if parameters.de_params.num_walkers < 4:
            # No differential evolution: DE mutation (a - b + c - d) needs >= 4 distinct
            # walkers. With fewer, re-evaluate the SAME population each trial; the closed
            # loop is then driven PURELY by the OO feed-forward (the Hamiltonian rotated
            # between trials). This is the clean single-ansatz OO-effect measurement, with
            # no DE exploration confounding whether re-diagonalizing in the rotated basis
            # lowers the energy.
            trial_populations = state.populations
        else:
            # Create next generation
            trial_populations = mutation_and_crossover(
                current_populations=state.populations,
                best_index=state.best_index,
                scaling_factor=parameters.de_params.fxc,
                crossover_rate=parameters.de_params.cr_prob,
            )
    else:
        # Initialize populations
        trial_populations = initialize_ucj_parameters(
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            num_walkers=parameters.de_params.num_walkers,
            randomization_factor=parameters.de_params.randomization_factor,
            n_lucj_layers=parameters.circ_params.n_lucj_layers,
            ucj_optimize=parameters.circ_params.ucj_optimize,
        )

    _, solver_block_name = parse_block_ref(parameters.solver_block_ref)

    futs = PrefectFutureList()
    for walker_index, ucj_parameter in enumerate(trial_populations):
        prefect_fut = walker_sqd.submit(
            trial_index=trial_index,
            walker_index=walker_index,
            ucj_parameter=ucj_parameter,
            circuit_params=parameters.circ_params,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            carryover=state.carryover,
            sqd_dim=parameters.sqd_dim,
            solver_block_name=solver_block_name,
            quantum_source=parameters.quantum_source,
            random_seed=parameters.random_seed,
            n_recovery_steps=parameters.n_recovery_steps,
            n_batches=parameters.n_batches,
            seed_cisd=parameters.seed_cisd,
            seed_budget_frac=parameters.seed_budget_frac,
            hci_boost=parameters.hci_boost,
            hci_boost_max=parameters.hci_boost_max,
            hci_boost_ncore=parameters.hci_boost_ncore,
        )
        futs.append(prefect_fut)

    # Collect results
    result_energies = np.full(parameters.de_params.num_walkers, np.nan, dtype=np.float64)
    result_carryovers: list[NpStrict2DArrayBool] = [None] * parameters.de_params.num_walkers
    result_sbd_results: list = [None] * parameters.de_params.num_walkers
    records: list[dict] = [None] * parameters.de_params.num_walkers
    for walker_index, ((energy, carryover, sbd_result), telemery) in enumerate(futs.result()):
        result_energies[walker_index] = energy
        result_carryovers[walker_index] = carryover
        result_sbd_results[walker_index] = sbd_result
        records[walker_index] = telemery

    # Update artifact
    artifact_id = extend_table_artifact(
        artifact_key="sqd-telemetry",
        new_table=records,
    )
    logger.debug(f"Updated sqd-telemetry artifact {str(artifact_id)}")

    new_state = selection(
        trial_populations=trial_populations,
        trial_energies=result_energies,
        trial_carryovers=result_carryovers,
        current_state=state,
    )

    # Best-energy walker's SBDResult (RDMs) for orbital optimization between trials.
    best_index = (
        int(np.nanargmin(result_energies)) if not np.all(np.isnan(result_energies)) else None
    )
    best_sbd_result = result_sbd_results[best_index] if best_index is not None else None

    return new_state, best_sbd_result


@task
def mutation_and_crossover(
    current_populations: NpStrict2DArrayF64,
    best_index: int,
    scaling_factor: float,
    crossover_rate: float,
) -> NpStrict2DArrayF64:
    global MODULE_RNG
    num_walkers, num_params = current_populations.shape

    if num_walkers < 4:
        # Each mutant draws 4 distinct other walkers (a - b + c - d); this is undefined below 4.
        # num_walkers < 4 is only allowed for a single evaluation pass (iterations = 1), which
        # never reaches this function, so reaching here with < 4 is a misconfiguration.
        raise ValueError(
            "Differential-evolution mutation requires num_walkers >= 4 "
            f"(got {num_walkers}); use iterations = 1 for a single-walker evaluation pass."
        )

    mutant = np.zeros_like(current_populations, dtype=np.float64)
    for i in range(num_walkers):
        r1, r2, r3, r4 = MODULE_RNG.choice(
            num_walkers,
            size=4,
            replace=False,
            shuffle=True,
        )
        drift_vec = (
            current_populations[r1]
            - current_populations[r2]
            + current_populations[r3]
            - current_populations[r4]
        )
        mutant[i] = current_populations[best_index] + scaling_factor * drift_vec

    for i in range(num_walkers):
        crossover_weights = MODULE_RNG.random(num_params)
        mask = crossover_weights > crossover_rate
        # Mutate at least one dimension
        index_to_keep = MODULE_RNG.choice(num_params, size=1)
        mask[index_to_keep] = False
        mutant[i, mask] = current_populations[i, mask]

    return mutant


@task
def selection(
    trial_populations: list[NpStrict2DArrayF64],
    trial_energies: NpStrict1DArrayF64,
    trial_carryovers: list[NpStrict2DArrayBool],
    current_state: OptimizerState,
) -> OptimizerState:
    logger = get_run_logger()

    new_state = current_state.copy()

    # The population array is pre-allocated from the spin-balanced (RHF) parameter count in
    # OptimizerState.from_parameters. The spin-unbalanced (UHF) UCJ operator has more parameters
    # (separate alpha/beta orbital rotations plus a beta-beta block), so the actual per-walker
    # vector is longer. Re-size the (still-uninitialized, all-NaN) population array to match the
    # real trial-parameter width the first time we see it; for RHF the widths already agree so
    # this is a no-op, and on later iterations the array holds real data and is left untouched.
    trial_width = int(np.asarray(trial_populations[0]).shape[0])
    if (
        new_state.populations.shape[1] != trial_width
        and np.all(np.isnan(new_state.populations))
    ):
        new_state.populations = np.full(
            (new_state.populations.shape[0], trial_width), np.nan, dtype=np.float64
        )

    best_index = int(np.nanargmin(trial_energies))
    if (
        current_state.best_energy() is None
        or trial_energies[best_index] < current_state.best_energy()
    ):
        # Update carryover when the best energy is updated
        logger.info(f"walker {best_index}: Update the best energy and carryover")
        new_state.best_index = best_index
        new_state.carryover = trial_carryovers[best_index]
    for walker_idx in range(len(trial_energies)):
        if np.isnan(trial_energies[walker_idx]):
            continue
        delta_e = trial_energies[walker_idx] - current_state.energies[walker_idx]
        logger.info(
            f"walker {walker_idx}: Davidson final energy = "
            f"{trial_energies[walker_idx]} (ΔE = {delta_e})"
        )
        if delta_e < 0:
            # Update reference energy and population when the trial gets lower energy
            new_state.energies[walker_idx] = trial_energies[walker_idx]
            new_state.populations[walker_idx] = trial_populations[walker_idx]
    return new_state


def parse_block_ref(ref: str) -> tuple[str, str]:
    parts = ref.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid solver_block_ref: {ref}")
    return parts[0], parts[1]


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    riken_sqd_de.serve(
        name="riken_sqd_de",
        description="SQD with LUCJ parameter optimization with differential evoluation.",
    )
