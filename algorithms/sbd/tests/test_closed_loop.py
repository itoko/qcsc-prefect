"""sbd.closed_loop: the state file closed-loop SQD-OO hands from one job to the next.

  * save -> load round-trips everything the DE loop carries between trials, for RHF (occupancy
    1-RDMs present, best walker set, OO frozen, an OO prediction pending) and UHF (beta / mixed
    blocks and ab_overlap, no 1-RDMs yet, no best walker).
  * A file from another system (norb, RHF vs UHF) or format version is refused.
  * carry_entries / save_state(carry=) copy the CC convergence record (cc_*) and meta of the
    sampling job's file, even after that file is gone.
  * The [cl-cc] and [cl-rot] lines carry the values they claim.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from scipy.linalg import expm

from qcsc_workflow_utility.chem import ElectronicProperties
from sbd import closed_loop
from sbd.main import OptimizerState

N = 5
NE = (2, 2)


def _rng():
    return np.random.default_rng(7)


def _sym4(rng, n):
    a = rng.normal(size=(n, n, n, n))
    return a + a.transpose(1, 0, 3, 2)


def _orth(rng, n, scale=0.3):
    k = rng.normal(size=(n, n)) * scale
    return expm(k - k.T), k - k.T


def _ep_rhf(rng):
    g = np.diag(np.linspace(1.9, 0.1, N))
    return ElectronicProperties(
        one_body_tensor=rng.normal(size=(N, N)), two_body_tensor=_sym4(rng, N),
        t2=rng.normal(size=(2, 2, 3, 3)), initial_occupancy=(np.diag(g) / 2, np.diag(g) / 2),
        occupancy_rdm1=(g / 2, g / 2 + 0.01), nuclear_repulsion_energy=12.5, num_orbitals=N,
        num_electrons=NE, open_shell=False, spin_sq=0.0,
    )


def _ep_uhf(rng):
    return ElectronicProperties(
        one_body_tensor=rng.normal(size=(N, N)), one_body_tensor_b=rng.normal(size=(N, N)),
        two_body_tensor=_sym4(rng, N), two_body_tensor_ab=_sym4(rng, N),
        two_body_tensor_bb=_sym4(rng, N),
        t2=rng.normal(size=(2, 2, 3, 3)), t2_ab=rng.normal(size=(2, 2, 3, 3)),
        t2_bb=rng.normal(size=(2, 2, 3, 3)),
        initial_occupancy=(np.array([1.0, 1, 0, 0, 0]), np.array([1.0, 1, 0, 0, 0])),
        nuclear_repulsion_energy=-3.25, num_orbitals=N, num_electrons=NE, open_shell=False,
        spin_sq=0.0, unrestricted=True, ab_overlap=_orth(rng, N)[0],
    )


def _state(rng, best):
    return OptimizerState(
        energies=rng.normal(size=1), populations=rng.normal(size=(1, 17)),
        carryover=rng.random((9, N)) < 0.5, best_index=best,
    )


def _assert_ep_equal(a: ElectronicProperties, b: ElectronicProperties):
    for k in closed_loop._EP_ARRAYS:
        va, vb = getattr(a, k), getattr(b, k)
        assert (va is None) == (vb is None), k
        if va is not None:
            np.testing.assert_array_equal(va, vb, err_msg=k)
    for s in (0, 1):
        np.testing.assert_array_equal(a.initial_occupancy[s], b.initial_occupancy[s])
    assert (a.occupancy_rdm1 is None) == (b.occupancy_rdm1 is None)
    if a.occupancy_rdm1 is not None:
        for s in (0, 1):
            np.testing.assert_array_equal(a.occupancy_rdm1[s], b.occupancy_rdm1[s])
    for k in ("nuclear_repulsion_energy", "num_orbitals", "num_electrons", "open_shell",
              "spin_sq", "unrestricted"):
        assert getattr(a, k) == getattr(b, k), k


@pytest.mark.parametrize("kind", ["rhf", "uhf"])
def test_round_trip(tmp_path, kind):
    rng = _rng()
    ep = _ep_rhf(rng) if kind == "rhf" else _ep_uhf(rng)
    best = 0 if kind == "rhf" else None
    st = _state(rng, best)
    rot = closed_loop.CumulativeRotation(N)
    ua, ub = _orth(rng, N)[0], _orth(rng, N)[0]
    rot.apply(ua, ub if kind == "uhf" else None)
    pred = {"trial": 6, "e_davidson_source": -1.25, "e_oo_estimate": -1.5, "method": "trial-RDM",
            "method_detail": "x"} if kind == "rhf" else None
    path = closed_loop.state_path(str(tmp_path), 7)
    closed_loop.save_state(path, 7, ep, st, rot, do_orbital_opt=(kind == "uhf"),
                           oo_prediction=pred)
    assert path.endswith("state_t007.npz")
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())

    trial, ep2, st2, rot2, loop = closed_loop.load_state(
        path, OptimizerState, norb=N, unrestricted=(kind == "uhf")
    )
    assert trial == 7
    _assert_ep_equal(ep, ep2)
    np.testing.assert_array_equal(st.energies, st2.energies)
    np.testing.assert_array_equal(st.populations, st2.populations)
    np.testing.assert_array_equal(st.carryover, st2.carryover)
    assert st2.carryover.dtype == bool
    assert st2.best_index == best
    np.testing.assert_array_equal(rot.c_a, rot2.c_a)
    np.testing.assert_array_equal(rot.c_b, rot2.c_b)
    if kind == "rhf":
        np.testing.assert_array_equal(rot2.c_a, rot2.c_b)
    assert loop["do_orbital_opt"] is (kind == "uhf")
    assert loop["oo_prediction"] == pred
    assert loop["cc"] == {}


def test_refuses_other_system(tmp_path):
    rng = _rng()
    path = str(tmp_path / "s.npz")
    closed_loop.save_state(path, 0, _ep_rhf(rng), _state(rng, None), closed_loop.CumulativeRotation(N))
    with pytest.raises(ValueError, match="norb"):
        closed_loop.load_state(path, OptimizerState, norb=N + 1)
    with pytest.raises(ValueError, match="unrestricted"):
        closed_loop.load_state(path, OptimizerState, unrestricted=True)
    d = closed_loop.read_raw(path)
    d["version"] = np.int64(99)
    closed_loop.write_raw(path, d)
    with pytest.raises(ValueError, match="version"):
        closed_loop.load_state(path, OptimizerState)


def _cc_info(converged=True):
    hist = [(1, -0.1, -0.1, 0.5), (2, -0.12, -0.02, 0.05), (3, -0.1201, -1e-4, 1e-3)]
    return {"method": "UCCSD", "converged": converged, "cycles": 3, "max_cycle": 200,
            "e_tot": -327.1, "e_corr": -0.1201, "de_last": -1e-4, "norm_t_last": 1e-3,
            "t1_norm": 0.01, "t2_norm": 0.7, "warm_start": True, "wall_s": 12.0, "history": hist}


def test_carry_copies_cc_record(tmp_path):
    rng = _rng()
    ep, st, rot = _ep_uhf(rng), _state(rng, 0), closed_loop.CumulativeRotation(N)
    src = str(tmp_path / "state_t003.npz")
    data = closed_loop.state_to_raw(3, ep, st, rot, meta={"variant": "bsuhf"})
    data.update(closed_loop.cc_record_to_raw(_cc_info(converged=False), 3))
    closed_loop.write_raw(src, data)

    out = str(tmp_path / "state_t004.npz")
    carry = closed_loop.carry_entries(src)
    import os
    os.remove(src)  # the source may be pruned after restore; the carried entries are in memory
    closed_loop.save_state(out, 4, ep, st, rot, carry=carry)
    trial, *_, loop = closed_loop.load_state(out, OptimizerState)
    assert trial == 4
    assert loop["meta"] == {"variant": "bsuhf"}
    cc = loop["cc"]
    assert cc["iteration"] == 3 and cc["converged"] is False and cc["cycles"] == 3
    assert cc["method"] == "UCCSD"
    np.testing.assert_array_equal(cc["history"], np.asarray(_cc_info()["history"]))


def test_cc_summary_line():
    line = closed_loop.cc_summary_line(_cc_info(converged=False), 12)
    m = re.match(
        r"\[cl-cc\] iter=12 method=UCCSD converged=False cycles=3 max_cycle=200 "
        r"E_CCSD=(\S+) E_corr=(\S+) dE_last=(\S+) norm_t_last=(\S+) \|\|t1\|\|=(\S+) "
        r"\|\|t2\|\|=(\S+) warm_start=yes wall=12\.0s$", line)
    assert m, line
    assert float(m.group(1)) == pytest.approx(-327.1)
    assert float(m.group(3)) == pytest.approx(-1e-4)


def test_rotation_summary_reports_generator_norm():
    rng = _rng()
    ua, ka = _orth(rng, N, 0.1)
    ub, kb = _orth(rng, N, 0.1)
    rot = closed_loop.CumulativeRotation(N)
    rot.apply(ua, ub)
    snap = rot.snapshot()
    rot.apply(ua, ub)
    s = closed_loop.rotation_summary(ua, ub, rot)
    vals = dict(kv.split("=") for kv in s.split())
    assert float(vals["step_logU_a"]) == pytest.approx(np.linalg.norm(ka), rel=1e-6)
    assert float(vals["step_logU_b"]) == pytest.approx(np.linalg.norm(kb), rel=1e-6)
    assert float(vals["cum_logU_a"]) == pytest.approx(2 * np.linalg.norm(ka), rel=1e-6)
    rot.restore(snap)
    np.testing.assert_allclose(rot.c_a, ua)


def test_prune_keeps_every_nth_and_two_newest(tmp_path, monkeypatch):
    rng = _rng()
    ep, st, rot = _ep_rhf(rng), _state(rng, 0), closed_loop.CumulativeRotation(N)
    monkeypatch.setenv("SBD_STATE_KEEP_EVERY", "5")
    for t in range(0, 13):
        closed_loop.save_state(closed_loop.state_path(str(tmp_path), t), t, ep, st, rot)
    kept = sorted(int(p.name[7:10]) for p in tmp_path.glob("state_t*.npz"))
    assert kept == [0, 5, 10, 11, 12]
    monkeypatch.delenv("SBD_STATE_KEEP_EVERY")
    assert closed_loop.prune_states(str(tmp_path), 13) == []


# ── Plan QR C3: per-trial noiseless resampling command ─────────────────────────

def test_sim_resample_argv(monkeypatch):
    import sys

    from sbd import closed_loop as cl

    monkeypatch.delenv("SBD_SIM_RESAMPLE_CMD", raising=False)
    assert not cl.sim_resample_enabled()
    monkeypatch.setenv("SBD_SIM_RESAMPLE_CMD", "{python} /kit/sim_sample.py --state {state} --seed {seed} --out {out}")
    monkeypatch.setenv("SBD_SAVED_PER_TRIAL", "file:///p/pool_t{trial:03d}.npz")
    monkeypatch.setenv("SEED", "43")
    assert cl.sim_resample_enabled()
    argv, out = cl.sim_resample_argv(7, "/s/state_t007.npz")
    assert out == "/p/pool_t007.npz"
    assert argv == [sys.executable, "/kit/sim_sample.py", "--state", "/s/state_t007.npz", "--seed", "43007",
                    "--out", "/p/pool_t007.npz"]
    monkeypatch.setenv("SBD_SAVED_PER_TRIAL", "file:///p/pool.npz")
    import pytest

    with pytest.raises(ValueError):
        cl.sim_resample_argv(7, "/s/state_t007.npz")
