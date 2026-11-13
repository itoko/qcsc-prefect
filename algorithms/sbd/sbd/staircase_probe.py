"""Opt-in probes for the RHF-input "staircase" study (PLAN_staircase_mechanism.md, C/D).

(The branch save/restore at the end of this file is documented there.) The first three are controlled by environment variables and are OFF unless set:

* ``SBD_DUMP_RDM_DIR`` -- per-trial dump of the best walker's RDMs, determinant lists,
  occupancies, this trial's OO rotation (Ua, Ub) and the CUMULATIVE rotation from the input
  FCIDUMP orbitals (``C_a``, ``C_b``; h_t = C^T h_0 C). ``SBD_DUMP_RDM_TRIALS`` limits which
  trials are written ("all", "30", "25-60", "10,20,30-40"; default "all"). One trial is
  ~40 MB uncompressed for norb=36.
* ``SBD_DUMP_RECOVERY_STEPS`` -- (needs ``SBD_DUMP_RDM_DIR``) per-RECOVERY-STEP dump inside one
  trial, for MODE=cronly where the whole run is trial 0 with many recovery steps. 1-based step
  numbers in the same spec syntax ("1,10,50-60"); writes ``rdmdump_t<trial>_r<step>.npz`` with
  that step's best-batch RDMs, determinant lists, carryover and the occupancies that steer the
  next step. No rotation fields (the orbitals do not change within a trial). Unset = off.
* ``SBD_FREEZE_SUBSPACE_AT=T`` (D4) -- from trial T on, every Davidson solve reuses the alpha/beta
  determinant lists drawn at trial T; only the orbitals (via OO) change. The lists are kept in
  ``SBD_FREEZE_SUBSPACE_FILE`` (default ``$FE4S4_CKPT_DIR/frozen_subspace.npz``).
* ``SBD_FREEZE_ORBITALS_AT=T`` (mirror of D4) -- from trial T on, the OO rotation is not applied,
  so the orbitals stay at those of trial T while sampling / recovery / carryover continue. OO is
  still solved on each trial's RDMs and its predicted gain (E_oo - E_davidson) is logged, which
  measures how much room for OO each new subspace creates at fixed orbitals.

The cumulative rotation is tracked for the fixed-RDM OO path only; with OO_RESOLVE=1 the
self-consistent OO does not return (Ua, Ub), so ``c_a``/``c_b`` in the dump would be wrong.
"""

from __future__ import annotations

import os

import numpy as np


def parse_trials(spec: str | None) -> set[int] | None:
    """Parse "all" / "30" / "25-60" / "10,20,30-40". None means every trial."""
    spec = (spec or "all").strip().lower()
    if spec in ("", "all"):
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return out


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


# ── RDM / rotation dump ──────────────────────────────────────────────────────


class RotationTracker:
    """Cumulative orbital rotation relative to the input FCIDUMP orbitals.

    ``rotate_electronic_properties`` applies h_new = U^T h_old U, so after rotations
    U_0, U_1, ... the integrals are C^T h_0 C with C = U_0 U_1 ...
    """

    def __init__(self, norb: int):
        self.c_a = np.eye(norb)
        self.c_b = np.eye(norb)

    def apply(self, ua: np.ndarray, ub: np.ndarray | None) -> None:
        self.c_a = self.c_a @ np.asarray(ua)
        self.c_b = self.c_b @ np.asarray(ub if ub is not None else ua)


def dump_enabled(trial: int) -> bool:
    if not os.environ.get("SBD_DUMP_RDM_DIR", "").strip():
        return False
    trials = parse_trials(os.environ.get("SBD_DUMP_RDM_TRIALS"))
    return trials is None or trial in trials


def dump_trial(
    trial: int,
    sbd_result,
    tracker: RotationTracker,
    *,
    e_davidson_total: float | None,
    ua: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    e_oo: float | None = None,
    ab_overlap: np.ndarray | None = None,
    logger=None,
) -> str | None:
    """Write ``rdmdump_t<trial>.npz``. ``tracker`` must hold the rotation BEFORE this trial's OO
    (the basis the RDMs are expressed in). Returns the path, or None when dumping is off."""
    if not dump_enabled(trial):
        return None
    out_dir = os.environ["SBD_DUMP_RDM_DIR"].strip()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rdmdump_t{trial:03d}.npz")

    def arr(x):
        return np.asarray(x) if x is not None else np.empty(0)

    occ = getattr(sbd_result, "orbital_occupancies", None)
    np.savez_compressed(
        path,
        trial=np.asarray(trial),
        e_davidson_total=np.asarray(np.nan if e_davidson_total is None else e_davidson_total),
        e_oo_total=np.asarray(np.nan if e_oo is None else e_oo),
        rdm1_a=arr(sbd_result.rdm1),
        rdm1_b=arr(sbd_result.rdm1_b),
        rdm2_aa=arr(sbd_result.rdm2),
        rdm2_ab=arr(sbd_result.rdm2_ab),
        rdm2_bb=arr(sbd_result.rdm2_bb),
        rdm2_notation=np.asarray("prqs"),
        alphadets=arr(getattr(sbd_result, "alphadets", None)),
        betadets=arr(getattr(sbd_result, "betadets", None)),
        # carryover = the solver's highest-weight determinants (per spin); their turnover between
        # trials is the H2 signal "the important determinants changed"
        carryover_a=arr(getattr(sbd_result, "carryover_bitstrings", None)),
        carryover_b=arr(getattr(sbd_result, "carryover_bitstrings_b", None)),
        occ_a=arr(occ[0] if occ is not None else None),
        occ_b=arr(occ[1] if occ is not None else None),
        c_a=tracker.c_a,
        c_b=tracker.c_b,
        ua=arr(ua),
        ub=arr(ub),
        ab_overlap=arr(ab_overlap),
    )
    if logger is not None:
        logger.info("[dump] trial %d: wrote %s (%.1f MB)", trial, path, os.path.getsize(path) / 1e6)
    return path


# ── per-recovery-step dump (Plan D, PD5: CR-only) ────────────────────────────


def recovery_dump_enabled(step: int) -> bool:
    """``step`` is 1-based (the number printed in ``[diag] recovery <step>/<n>``)."""
    if not os.environ.get("SBD_DUMP_RDM_DIR", "").strip():
        return False
    raw = os.environ.get("SBD_DUMP_RECOVERY_STEPS", "").strip()
    if not raw:
        return False
    steps = parse_trials(raw)
    return steps is None or step in steps


def dump_recovery_step(
    trial: int,
    step: int,
    sbd_result,
    *,
    energy_total: float | None,
    occ_a: np.ndarray,
    occ_b: np.ndarray,
    ab_overlap: np.ndarray | None = None,
    logger=None,
) -> str | None:
    """Write ``rdmdump_t<trial>_r<step>.npz`` (``step`` 1-based). ``occ_a/b`` are this step's
    batch-averaged occupancies (the ``avg_occ`` of the next step). Returns the path, or None
    when off."""
    if not recovery_dump_enabled(step) or sbd_result is None:
        return None
    out_dir = os.environ["SBD_DUMP_RDM_DIR"].strip()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rdmdump_t{trial:03d}_r{step:03d}.npz")

    def arr(x):
        return np.asarray(x) if x is not None else np.empty(0)

    np.savez_compressed(
        path,
        trial=np.asarray(trial),
        recovery_step=np.asarray(step),
        e_davidson_total=np.asarray(np.nan if energy_total is None else energy_total),
        rdm1_a=arr(sbd_result.rdm1),
        rdm1_b=arr(sbd_result.rdm1_b),
        rdm2_aa=arr(sbd_result.rdm2),
        rdm2_ab=arr(sbd_result.rdm2_ab),
        rdm2_bb=arr(sbd_result.rdm2_bb),
        rdm2_notation=np.asarray("prqs"),
        alphadets=arr(getattr(sbd_result, "alphadets", None)),
        betadets=arr(getattr(sbd_result, "betadets", None)),
        carryover_a=arr(getattr(sbd_result, "carryover_bitstrings", None)),
        carryover_b=arr(getattr(sbd_result, "carryover_bitstrings_b", None)),
        occ_a=np.asarray(occ_a),
        occ_b=np.asarray(occ_b),
        ab_overlap=arr(ab_overlap),
    )
    if logger is not None:
        logger.info("[dump] trial %d recovery %d: wrote %s (%.1f MB)",
                    trial, step, path, os.path.getsize(path) / 1e6)
    return path


# ── D4: freeze the determinant subspace ──────────────────────────────────────


def _subspace_file() -> str:
    explicit = os.environ.get("SBD_FREEZE_SUBSPACE_FILE", "").strip()
    if explicit:
        return explicit
    base = os.environ.get("FE4S4_CKPT_DIR", "").strip() or "."
    return os.path.join(base, "frozen_subspace.npz")


def freeze_subspace(trial: int, key: str, ci_a: np.ndarray, ci_b: np.ndarray, logger=None):
    """Return the determinant lists to diagonalize at ``trial``.

    Off (env unset) or trial < T: the drawn lists unchanged. Trial == T: store them and return
    them. Trial > T: return the lists stored at T (``key`` distinguishes recovery step / batch).
    """
    t_freeze = _env_int("SBD_FREEZE_SUBSPACE_AT")
    if t_freeze is None or trial < t_freeze:
        return ci_a, ci_b
    path = _subspace_file()
    if trial == t_freeze:
        stored = dict(np.load(path)) if os.path.exists(path) else {}
        stored[f"{key}_a"] = np.asarray(ci_a)
        stored[f"{key}_b"] = np.asarray(ci_b)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **stored)
        if logger is not None:
            logger.info("[freeze-subspace] trial %d: stored %s (n_a=%d n_b=%d) in %s",
                        trial, key, len(ci_a), len(ci_b), path)
        return ci_a, ci_b
    with np.load(path) as z:
        fa, fb = z[f"{key}_a"], z[f"{key}_b"]
    if logger is not None:
        logger.info("[freeze-subspace] trial %d: reusing %s from trial %d (n_a=%d n_b=%d; "
                    "drawn lists discarded)", trial, key, t_freeze, len(fa), len(fb))
    return fa, fb


# ── mirror of D4: freeze the orbitals ────────────────────────────────────────


def orbitals_frozen(trial: int) -> bool:
    t_freeze = _env_int("SBD_FREEZE_ORBITALS_AT")
    return t_freeze is not None and trial >= t_freeze


# ── branch: save the full between-trial state and restart from it ────────────
#
# Between two DE trials the loop carries only: the (rotated) integrals ``elec_props``, the
# optimizer ``state`` (energies, LUCJ populations, carryover, best index) and the cumulative
# rotation. Recovery occupancies are NOT carried (each trial starts from
# ``elec_props.initial_occupancy``), and the subsampling RNG is seeded per (random_seed, trial),
# so a run restored at trial T with the same seed continues exactly like the original except
# for solver-level numerical noise. That makes "branch at T, then intervene" experiments possible.
#
#   SBD_BRANCH_SAVE_AT="25,30"  write branch_t<T>.npz (state BEFORE trial T runs) in
#                                SBD_BRANCH_DIR (default $FE4S4_CKPT_DIR)
#   SBD_BRANCH_FROM=<file>      restore that state and start the loop at its trial T


def branch_save_trials() -> set[int]:
    raw = os.environ.get("SBD_BRANCH_SAVE_AT", "").strip()
    return (parse_trials(raw) or set()) if raw and raw.lower() != "all" else set()


def _branch_dir() -> str:
    return (os.environ.get("SBD_BRANCH_DIR", "").strip()
            or os.environ.get("FE4S4_CKPT_DIR", "").strip() or ".")


_EP_ARRAYS = ("one_body_tensor", "two_body_tensor", "t2", "one_body_tensor_b", "two_body_tensor_ab",
              "two_body_tensor_bb", "t2_ab", "t2_bb", "ab_overlap")


def save_branch(trial: int, elec_props, state, tracker: RotationTracker, logger=None) -> str:
    """Write the state the loop holds just BEFORE running ``trial``."""
    out = {f"ep_{k}": np.asarray(getattr(elec_props, k)) for k in _EP_ARRAYS
           if getattr(elec_props, k) is not None}
    out.update(
        trial=np.int64(trial),
        ep_initial_occupancy_a=np.asarray(elec_props.initial_occupancy[0]),
        ep_initial_occupancy_b=np.asarray(elec_props.initial_occupancy[1]),
        ep_nuclear_repulsion_energy=np.float64(elec_props.nuclear_repulsion_energy),
        ep_num_orbitals=np.int64(elec_props.num_orbitals),
        ep_num_electrons=np.asarray(elec_props.num_electrons, dtype=np.int64),
        ep_open_shell=np.bool_(elec_props.open_shell),
        ep_spin_sq=np.float64(elec_props.spin_sq),
        ep_unrestricted=np.bool_(elec_props.unrestricted),
        st_energies=np.asarray(state.energies),
        st_populations=np.asarray(state.populations),
        st_carryover=np.asarray(state.carryover, dtype=bool),
        st_best_index=np.int64(-1 if state.best_index is None else state.best_index),
        c_a=tracker.c_a, c_b=tracker.c_b,
    )
    d = _branch_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"branch_t{trial:03d}.npz")
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **out)
    os.replace(tmp, path)
    if logger is not None:
        logger.info("[branch] saved state before trial %d -> %s (%.1f MB)",
                    trial, path, os.path.getsize(path) / 1e6)
    return path


def load_branch(path: str, elec_props_template, state_cls, norb: int):
    """Return (start_trial, elec_props, state, tracker) restored from ``save_branch`` output."""
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    upd = {k: d[f"ep_{k}"] for k in _EP_ARRAYS if f"ep_{k}" in d}
    upd.update(
        initial_occupancy=(d["ep_initial_occupancy_a"], d["ep_initial_occupancy_b"]),
        nuclear_repulsion_energy=float(d["ep_nuclear_repulsion_energy"]),
        num_orbitals=int(d["ep_num_orbitals"]),
        num_electrons=tuple(int(x) for x in d["ep_num_electrons"]),
        open_shell=bool(d["ep_open_shell"]),
        spin_sq=float(d["ep_spin_sq"]),
        unrestricted=bool(d["ep_unrestricted"]),
    )
    if upd["num_orbitals"] != norb:
        raise ValueError(f"branch file {path}: norb {upd['num_orbitals']} != {norb}")
    elec_props = elec_props_template.model_copy(update=upd)
    bi = int(d["st_best_index"])
    state = state_cls(
        energies=d["st_energies"], populations=d["st_populations"],
        carryover=d["st_carryover"].astype(bool), best_index=None if bi < 0 else bi,
    )
    tracker = RotationTracker(norb)
    tracker.c_a, tracker.c_b = d["c_a"], d["c_b"]
    return int(d["trial"]), elec_props, state, tracker


# ── D6: different OO settings for the first trials ───────────────────────────
#
# The first fixed-RDM OO (trial 0 from RHF orbitals) does not converge with the production
# settings (trust 0.5, maxiter 300): it stops at the iteration cap with ~40 parameters on the box,
# and two runs whose RDMs differ by 1e-13 end 0.25 apart (|x| ~ 8 rad). Which spin-coupling basin
# a run lands in appears to be decided there. These env vars override the OO settings for trials
# 0..N-1 only; later trials use the normal OO_TRUST / OO_MAXITER / OO_GRAD_TOL.
#
#   SBD_EARLY_OO_TRIALS=N        number of early trials (unset/0 = off)
#   SBD_EARLY_OO_TRUST=<float>   trust radius (box bound) for those trials
#   SBD_EARLY_OO_MAXITER=<int>   L-BFGS-B maxiter for those trials
#   SBD_EARLY_OO_GTOL=<float>    L-BFGS-B pgtol for those trials


def early_oo_overrides(trial: int) -> dict:
    """Overrides for optimize_orbitals at ``trial`` ({} when off or past the early window)."""
    n = _env_int("SBD_EARLY_OO_TRIALS") or 0
    if trial >= n:
        return {}
    out: dict = {}
    for env, key, cast in (("SBD_EARLY_OO_TRUST", "trust_radius", float),
                           ("SBD_EARLY_OO_MAXITER", "maxiter", int),
                           ("SBD_EARLY_OO_GTOL", "gtol", float)):
        raw = os.environ.get(env, "").strip()
        if raw:
            out[key] = cast(raw)
    return out


# ── H2 attribution: re-diagonalize the PREVIOUS subspace in the NEW orbitals ──
#
# Each trial's energy change splits exactly into an orbital part and a subspace part:
#   E(S_t, h_t) - E(S_{t-1}, h_{t-1})
#     = [E(S_{t-1}, h_t) - E(S_{t-1}, h_{t-1})]      orbital part (OO_CHECK's "true OO effect")
#     + [E(S_t,     h_t) - E(S_{t-1}, h_t)]          subspace part (new determinants, same orbitals)
# SBD_H2_PROBE=1 runs one extra Davidson per trial on the previous trial's determinant lists with
# this trial's (rotated) integrals and logs both parts. The probe solve's RDMs are written to the
# dump (prefix "prev_") when the RDM dump is on, so offline OO on S_{t-1} vs S_t at the same
# orbitals measures how much room for OO the new determinants create. The probe never feeds
# back into the run (the trajectory is the same as without it, up to solver noise).


def h2_probe_enabled() -> bool:
    return os.environ.get("SBD_H2_PROBE", "").strip() == "1"


def dump_prev_subspace(trial: int, probe_result, logger=None) -> str | None:
    """Write ``rdmdump_t<trial>_prev.npz``: RDMs of S_{t-1} diagonalized with h_t."""
    if not dump_enabled(trial) or probe_result is None:
        return None
    out_dir = os.environ["SBD_DUMP_RDM_DIR"].strip()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rdmdump_t{trial:03d}_prev.npz")

    def arr(x):
        return np.asarray(x) if x is not None else np.empty(0)

    np.savez_compressed(
        path, trial=np.asarray(trial), e_davidson_total=np.asarray(float(probe_result.energy)),
        rdm1_a=arr(probe_result.rdm1), rdm1_b=arr(probe_result.rdm1_b), rdm2_aa=arr(probe_result.rdm2),
        rdm2_ab=arr(probe_result.rdm2_ab), rdm2_bb=arr(probe_result.rdm2_bb),
        alphadets=arr(getattr(probe_result, "alphadets", None)),
        betadets=arr(getattr(probe_result, "betadets", None)),
    )
    if logger is not None:
        logger.info("[h2-probe] trial %d: wrote %s", trial, path)
    return path


# ── trial-0 sample only: draw trial 0 with a different seed ──────────────────
#
# The saved pool is fixed, so trial 0's determinants are set by the initial occupancies (t2) and
# the subsampling RNG seed. SBD_TRIAL0_SEED=<int> replaces random_seed for trial 0 only; every
# later trial keeps the run's random_seed, so two runs differ only in which determinants trial 0
# drew. Off by default.


def walker_seed(random_seed: int, trial_index: int, walker_index: int) -> int:
    """Per-(trial, walker) subsampling seed (Cantor pairing, as before), with the trial-0 override."""
    base = random_seed
    if trial_index == 0:
        override = _env_int("SBD_TRIAL0_SEED")
        if override is not None:
            base = override
    k = trial_index + walker_index
    return int(base + (k * (k + 1) // 2 + walker_index))


# ── Plan E: change the determinant subspace (PE2 injection, PE3 exclusion) ───
#
# Both act on the per-spin lists [HF, carryover, seed, sampled] right after subsampling (UHF path).
#   SBD_INJECT_FILE=<npz with int arrays "alpha", "beta", in priority order>; "{SEED}" in the path is
#     replaced by $SEED. SBD_INJECT_TRIALS (required, same syntax as SBD_DUMP_RDM_TRIALS) and
#     SBD_INJECT_N (default: all, capped at half the list). The first N file strings not already in
#     the list replace the same number of strings at the END of the list (the sampled part), so the
#     dimension is unchanged and HF / carryover are kept.
#   SBD_EXCLUDE_EXC=1 drops every string that is a single excitation of the current reference
#     (orbitals 0..n-1 of the current basis), for both spins and every trial. HF stays.
# SBD_DUMP_ROT_DIR: per-trial rotation dump rot_tNNN.npz (ua, ub, the cumulative rotation before
#   this trial's OO, Davidson and OO energies), ~20 KB, for the rotation trajectory without RDMs.


def resolve_seed_path(path: str) -> str:
    return path.replace("{SEED}", os.environ.get("SEED", "").strip())


def _single_of_ref(strings: np.ndarray, n_elec: int) -> np.ndarray:
    ref = (1 << n_elec) - 1
    inside = np.array([bin(int(s) & ref).count("1") for s in strings], dtype=np.int64)
    return inside == n_elec - 1


_INJECT_CACHE: dict = {}


def _inject_strings(path: str):
    if path not in _INJECT_CACHE:
        with np.load(path) as z:
            _INJECT_CACHE[path] = (np.asarray(z["alpha"], dtype=np.int64), np.asarray(z["beta"], dtype=np.int64))
    return _INJECT_CACHE[path]


def modify_subspace(trial: int, ci_a, ci_b, num_elec_a: int, num_elec_b: int, logger=None):
    """Return the (alpha, beta) lists after the opt-in exclusion / injection (unchanged when off)."""
    excl = os.environ.get("SBD_EXCLUDE_EXC", "").strip() == "1"
    inj_path = os.environ.get("SBD_INJECT_FILE", "").strip()
    inj_trials = parse_trials(os.environ.get("SBD_INJECT_TRIALS")) if inj_path else set()
    if inj_path and inj_trials is None:
        raise ValueError("SBD_INJECT_FILE needs an explicit SBD_INJECT_TRIALS (not 'all')")
    inj_on = bool(inj_path) and trial in inj_trials
    if not excl and not inj_on:
        return ci_a, ci_b
    lists = [np.asarray(ci_a, dtype=np.int64), np.asarray(ci_b, dtype=np.int64)]
    nel = (num_elec_a, num_elec_b)
    if excl:
        for k in range(2):
            drop = _single_of_ref(lists[k], nel[k])
            lists[k] = lists[k][~drop]
            if logger is not None:
                logger.info("[exclude] trial %d spin %s: dropped %d single excitations of the reference "
                            "(%d strings left)", trial, "ab"[k], int(drop.sum()), len(lists[k]))
    if inj_on:
        path = resolve_seed_path(inj_path)
        files = _inject_strings(path)
        n_req = _env_int("SBD_INJECT_N")
        for k in range(2):
            cur = lists[k]
            present = set(cur.tolist())
            pop_ok = [int(s) for s in files[k].tolist() if bin(int(s)).count("1") == nel[k]]
            add = [s for s in pop_ok if s not in present]
            n_add = len(add) if n_req is None else min(n_req, len(add))
            n_add = min(n_add, len(cur) // 2 - 1)
            if n_add > 0:
                lists[k] = np.concatenate([cur[:len(cur) - n_add], np.asarray(add[:n_add], dtype=np.int64)])
            if logger is not None:
                logger.info("[inject] trial %d spin %s: +%d strings from %s (requested %s, %d already present, "
                            "%d with wrong electron count; dimension %d)", trial, "ab"[k], n_add, path,
                            "all" if n_req is None else n_req, len(pop_ok) - len(add),
                            len(files[k]) - len(pop_ok), len(lists[k]))
    return lists[0].astype(np.longlong), lists[1].astype(np.longlong)


def dump_rotation(trial: int, ua, ub, tracker: RotationTracker, *, e_davidson_total=None, e_oo=None,
                  logger=None) -> str | None:
    """Write rot_tNNN.npz when SBD_DUMP_ROT_DIR is set. ``tracker`` = rotation BEFORE this OO."""
    out_dir = os.environ.get("SBD_DUMP_ROT_DIR", "").strip()
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rot_t{trial:03d}.npz")
    np.savez(path, trial=np.int64(trial), ua=np.asarray(ua), ub=np.asarray(ub if ub is not None else ua),
             c_a=tracker.c_a, c_b=tracker.c_b,
             e_davidson_total=np.float64(np.nan if e_davidson_total is None else e_davidson_total),
             e_oo_total=np.float64(np.nan if e_oo is None else e_oo))
    if logger is not None:
        logger.info("[dump-rot] trial %d: wrote %s", trial, path)
    return path
