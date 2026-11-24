"""Save / restore the DE-loop state between jobs (closed-loop SQD-OO; opt-in, default off).

Closed-loop SQD-OO runs one DE trial per job. The GPU job restores the state held just BEFORE
trial k, runs trial k on the pool drawn from the circuit of the current (rotated) Hamiltonian
(QSOURCE=saved), and writes the state before trial k+1. A sampling job in between re-solves CC on
the rotated integrals, writes the new t2 into that file and draws the pool for trial k+1. Batch
SQD-OO uses the same files to resume a run cut by the walltime.

    SBD_STATE_FROM=<state_tNNN.npz>   restore that state and start the loop at its trial
    SBD_STATE_SAVE_DIR=<dir>          write state_t<i>.npz before every trial i after the first
                                      one run here, and after the last trial (atomic rename)
    SBD_STATE_KEEP_EVERY=N            after writing state_t<i>, delete state_t<i-2> there unless
                                      i-2 is a multiple of N (a UHF Fe4S4 file is ~40 MB; unset =
                                      keep all)

The file holds everything the loop carries from one trial to the next: the integrals, t2, the
initial occupancies and the 1-RDMs they came from (rotated with the orbitals), the DE population,
energies, best index and carryover, the cumulative orbital rotation, whether OO is still enabled
and the previous OO prediction (for the "OO effect" log line). ElectronicProperties is rebuilt
from the file alone, so the integrals the job loads from the FCIDUMP are only checked, not used.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

STATE_VERSION = 1

# Optional ElectronicProperties arrays (None for RHF / before the first SQD result).
_EP_ARRAYS = ("one_body_tensor", "two_body_tensor", "t2", "one_body_tensor_b", "two_body_tensor_ab",
              "two_body_tensor_bb", "t2_ab", "t2_bb", "ab_overlap")
# t2 fields the sampling job replaces after a CC refresh.
T2_FIELDS = ("t2", "t2_ab", "t2_bb")


class CumulativeRotation:
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

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.c_a.copy(), self.c_b.copy()

    def restore(self, snap: tuple[np.ndarray, np.ndarray]) -> None:
        self.c_a, self.c_b = snap[0].copy(), snap[1].copy()


def _log_norm(u: np.ndarray) -> float:
    """Frobenius norm of the real generator K of an orthogonal U = exp(K) (sqrt(2) x the
    rotation angles); nan if U has det -1 (no real logarithm)."""
    from scipy.linalg import logm

    u = np.asarray(u, dtype=np.float64)
    if np.linalg.det(u) < 0:
        return float("nan")
    return float(np.linalg.norm(np.real(logm(u))))


def rotation_summary(ua: np.ndarray, ub: np.ndarray | None, rotation: CumulativeRotation) -> str:
    """``|log U|`` of this trial's rotation and of the cumulative one, per spin (the ``[cl-rot]``
    log line). Only the fixed-RDM OO path is tracked; the self-consistent path (oo_resolve_rdms)
    does not expose its rotation."""
    ub = ua if ub is None else ub
    return (
        f"step_logU_a={_log_norm(ua):.6e} step_logU_b={_log_norm(ub):.6e} "
        f"cum_logU_a={_log_norm(rotation.c_a):.6e} cum_logU_b={_log_norm(rotation.c_b):.6e}"
    )


def state_from_env() -> str:
    return os.environ.get("SBD_STATE_FROM", "").strip()


def state_save_dir() -> str:
    return os.environ.get("SBD_STATE_SAVE_DIR", "").strip()


def loader_skip_enabled() -> bool:
    """``SBD_STATE_SKIP_LOADER=0`` keeps loading the FCIDUMP when a state is restored (the old behaviour)."""
    return os.environ.get("SBD_STATE_SKIP_LOADER", "1").strip() != "0"


def file_sha1(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def state_fcidump_check(path: str, fcidump: str) -> tuple[bool, str]:
    """Whether the state at ``path`` was built from ``fcidump`` (the ``fcidump_sha1`` in its meta,
    written by cl_sample.py init and carried along by every saved state).

    (True, reason): the sha1 matches, so the job may skip the FCIDUMP loader (the restore replaces
    every field the loader returns). (False, reason): the state records no sha1; load the FCIDUMP as
    before. Raises ValueError when the sha1 differs (the state belongs to another system).
    """
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    want = meta.get("fcidump_sha1")
    if not want:
        return False, f"{os.path.basename(path)} records no fcidump_sha1"
    got = file_sha1(fcidump)
    if got != want:
        raise ValueError(
            f"state file {path} was built from an FCIDUMP with sha1 {want[:12]}, "
            f"but {fcidump} has sha1 {got[:12]}"
        )
    return True, f"fcidump_sha1 {got[:12]} matches {os.path.basename(path)}"


def state_path(directory: str, trial: int) -> str:
    return os.path.join(directory, f"state_t{trial:03d}.npz")


def read_raw(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_raw(path: str, data: dict) -> str:
    """Write ``data`` as an .npz at ``path`` through a temporary file and an atomic rename."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **data)
    os.replace(tmp, path)
    return path


def state_to_raw(
    trial: int,
    elec_props,
    state,
    rotation: CumulativeRotation,
    *,
    do_orbital_opt: bool = True,
    oo_prediction: dict | None = None,
    meta: dict | None = None,
) -> dict:
    """The arrays ``save_state`` writes (also used by the sampling job to build state_t000)."""
    out = {f"ep_{k}": np.asarray(getattr(elec_props, k), dtype=np.float64) for k in _EP_ARRAYS
           if getattr(elec_props, k) is not None}
    if elec_props.occupancy_rdm1 is not None:
        out["ep_occupancy_rdm1_a"] = np.asarray(elec_props.occupancy_rdm1[0], dtype=np.float64)
        out["ep_occupancy_rdm1_b"] = np.asarray(elec_props.occupancy_rdm1[1], dtype=np.float64)
    out.update(
        version=np.int64(STATE_VERSION),
        trial=np.int64(trial),
        ep_initial_occupancy_a=np.asarray(elec_props.initial_occupancy[0], dtype=np.float64),
        ep_initial_occupancy_b=np.asarray(elec_props.initial_occupancy[1], dtype=np.float64),
        ep_nuclear_repulsion_energy=np.float64(elec_props.nuclear_repulsion_energy),
        ep_num_orbitals=np.int64(elec_props.num_orbitals),
        ep_num_electrons=np.asarray(elec_props.num_electrons, dtype=np.int64),
        ep_open_shell=np.bool_(elec_props.open_shell),
        ep_spin_sq=np.float64(elec_props.spin_sq),
        ep_unrestricted=np.bool_(elec_props.unrestricted),
        st_energies=np.asarray(state.energies, dtype=np.float64),
        st_populations=np.asarray(state.populations, dtype=np.float64),
        st_carryover=np.asarray(state.carryover, dtype=bool),
        st_best_index=np.int64(-1 if state.best_index is None else state.best_index),
        rot_c_a=np.asarray(rotation.c_a),
        rot_c_b=np.asarray(rotation.c_b),
        loop_do_orbital_opt=np.bool_(do_orbital_opt),
        loop_oo_prediction=np.asarray(json.dumps(oo_prediction)),
        meta=np.asarray(json.dumps(meta or {})),
    )
    return out


def carry_entries(path: str) -> dict:
    """The ``cc_*`` / ``meta`` entries of a state file (the CC convergence record the sampling job
    wrote), to be copied into every state saved after restoring from it (``save_state(carry=)``).
    Read once at restore: the file itself may be pruned later."""
    return {k: v for k, v in read_raw(path).items() if k.startswith("cc_") or k == "meta"}


def save_state(path: str, trial: int, elec_props, state, rotation: CumulativeRotation, *,
               do_orbital_opt: bool = True, oo_prediction: dict | None = None,
               carry: dict | None = None, logger=None) -> str:
    """Write the state the loop holds just BEFORE running ``trial``.

    ``carry``: entries from :func:`carry_entries` copied along, so each file keeps the record of
    the t2 it holds.
    """
    data = state_to_raw(trial, elec_props, state, rotation,
                        do_orbital_opt=do_orbital_opt, oo_prediction=oo_prediction)
    if carry:
        data.update(carry)
    write_raw(path, data)
    if logger is not None:
        logger.info("[state] saved state before trial %d -> %s (%.1f MB)",
                    trial, path, os.path.getsize(path) / 1e6)
    pruned = prune_states(os.path.dirname(os.path.abspath(path)), trial)
    if pruned and logger is not None:
        logger.info("[state] pruned %s", ", ".join(os.path.basename(p) for p in pruned))
    return path


def prune_states(directory: str, newest: int, keep_every: int | None = None) -> list[str]:
    """Delete state_t<newest-2>.npz (and its .precc.npz) in ``directory`` unless it is a multiple of ``keep_every``
    (default: SBD_STATE_KEEP_EVERY; unset or 0 keeps everything). The two newest files and
    state_t000 are always kept."""
    if keep_every is None:
        raw = os.environ.get("SBD_STATE_KEEP_EVERY", "").strip()
        keep_every = int(raw) if raw else 0
    old = newest - 2
    if keep_every <= 0 or old <= 0 or old % keep_every == 0:
        return []
    path = state_path(directory, old)
    removed = []
    # cl_sample.py step keeps the pre-CC copy next to the file it refreshes
    for p in (path, path[: -len(".npz")] + ".precc.npz"):
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)
    return removed


def elec_props_from_raw(d: dict):
    from qcsc_workflow_utility.chem import ElectronicProperties

    fields = {k: d[f"ep_{k}"] for k in _EP_ARRAYS if f"ep_{k}" in d}
    occ_rdm1 = None
    if "ep_occupancy_rdm1_a" in d:
        occ_rdm1 = (d["ep_occupancy_rdm1_a"], d["ep_occupancy_rdm1_b"])
    return ElectronicProperties(
        **fields,
        occupancy_rdm1=occ_rdm1,
        initial_occupancy=(d["ep_initial_occupancy_a"], d["ep_initial_occupancy_b"]),
        nuclear_repulsion_energy=float(d["ep_nuclear_repulsion_energy"]),
        num_orbitals=int(d["ep_num_orbitals"]),
        num_electrons=tuple(int(x) for x in d["ep_num_electrons"]),
        open_shell=bool(d["ep_open_shell"]),
        spin_sq=float(d["ep_spin_sq"]),
        unrestricted=bool(d["ep_unrestricted"]),
    )


def load_state(path: str, state_cls, *, norb: int | None = None, unrestricted: bool | None = None):
    """Return ``(start_trial, elec_props, state, rotation, loop)`` restored from ``path``.

    ``loop`` = {"do_orbital_opt": bool, "oo_prediction": dict | None, "meta": dict, "cc": dict}.
    ``norb`` / ``unrestricted``: what the job loaded from its FCIDUMP; a mismatch raises.
    """
    d = read_raw(path)
    if int(d.get("version", -1)) != STATE_VERSION:
        raise ValueError(f"state file {path}: version {d.get('version')} != {STATE_VERSION}")
    elec_props = elec_props_from_raw(d)
    if norb is not None and elec_props.num_orbitals != norb:
        raise ValueError(f"state file {path}: norb {elec_props.num_orbitals} != {norb}")
    if unrestricted is not None and elec_props.unrestricted != unrestricted:
        raise ValueError(
            f"state file {path}: unrestricted={elec_props.unrestricted} but the solver is "
            f"{'uhf' if unrestricted else 'rhf'}"
        )
    bi = int(d["st_best_index"])
    state = state_cls(
        energies=d["st_energies"], populations=d["st_populations"],
        carryover=d["st_carryover"].astype(bool), best_index=None if bi < 0 else bi,
    )
    rotation = CumulativeRotation(elec_props.num_orbitals)
    rotation.c_a, rotation.c_b = d["rot_c_a"], d["rot_c_b"]
    loop = {
        "do_orbital_opt": bool(d["loop_do_orbital_opt"]),
        "oo_prediction": json.loads(str(d["loop_oo_prediction"])),
        "meta": json.loads(str(d["meta"])) if "meta" in d else {},
        "cc": {k[3:]: d[k].item() if d[k].ndim == 0 else d[k] for k in d if k.startswith("cc_")},
    }
    return int(d["trial"]), elec_props, state, rotation, loop


def cc_record_to_raw(info: dict, iteration: int) -> dict:
    """``cc_*`` entries of a state file from a ``refresh_cc_seed(return_info=True)`` record."""
    hist = np.asarray(info.get("history") or np.zeros((0, 4)), dtype=np.float64).reshape(-1, 4)
    return {
        "cc_iteration": np.int64(iteration),
        "cc_method": np.asarray(info["method"]),
        "cc_converged": np.bool_(info["converged"]),
        "cc_cycles": np.int64(info["cycles"]),
        "cc_max_cycle": np.int64(info["max_cycle"]),
        "cc_e_tot": np.float64(info["e_tot"]),
        "cc_e_corr": np.float64(info["e_corr"]),
        "cc_de_last": np.float64(info["de_last"]),
        "cc_norm_t_last": np.float64(info["norm_t_last"]),
        "cc_t1_norm": np.float64(info["t1_norm"]),
        "cc_t2_norm": np.float64(info["t2_norm"]),
        "cc_warm_start": np.bool_(info["warm_start"]),
        "cc_wall_s": np.float64(info["wall_s"]),
        "cc_history": hist,
    }


def cc_summary_line(info: dict, iteration: int) -> str:
    """The ``[cl-cc]`` line the sampling job logs (parsed by cl_collect.py)."""
    return (
        f"[cl-cc] iter={iteration} method={info['method']} converged={info['converged']} "
        f"cycles={info['cycles']} max_cycle={info['max_cycle']} E_CCSD={info['e_tot']:.10f} "
        f"E_corr={info['e_corr']:.10f} dE_last={info['de_last']:.3e} "
        f"norm_t_last={info['norm_t_last']:.3e} ||t1||={info['t1_norm']:.6e} "
        f"||t2||={info['t2_norm']:.6e} warm_start={'yes' if info['warm_start'] else 'no'} "
        f"wall={info['wall_s']:.1f}s"
    )
