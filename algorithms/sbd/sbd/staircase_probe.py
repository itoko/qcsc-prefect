"""Opt-in probes for the RHF-input "staircase" study (PLAN_staircase_mechanism.md, C/D).

(The branch save/restore at the end of this file is documented there.) The first three are controlled by environment variables and are OFF unless set:

* ``SBD_DUMP_RDM_DIR`` -- per-trial dump of the best walker's RDMs, determinant lists,
  occupancies, this trial's OO rotation (Ua, Ub) and the CUMULATIVE rotation from the input
  FCIDUMP orbitals (``C_a``, ``C_b``; h_t = C^T h_0 C). ``SBD_DUMP_RDM_TRIALS`` limits which
  trials are written ("all", "30", "25-60", "10,20,30-40"; default "all"). One trial is
  ~40 MB uncompressed for norb=36.
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
