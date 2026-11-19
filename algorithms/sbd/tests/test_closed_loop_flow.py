"""riken_sqd_de with SBD_STATE_FROM / SBD_STATE_SAVE_DIR: one trial per job == one run of N trials.

The solver, the SQD trial and the orbital optimizer are replaced by deterministic stand-ins (a
function of the integrals, the carryover and the trial index), so the DE loop's own bookkeeping is
what is tested:

  * A run of 4 trials started from state_t000 and 4 chained runs of one trial each (each restored
    from the previous run's file, the closed-loop GPU job) end in byte-identical states, and so do
    the intermediate files.
  * The restored run continues the "OO effect" bookkeeping (the previous OO prediction) and logs
    [cl-trial] / [cl-rot] / [state] lines with the right trial numbers.
  * Without the two variables nothing is written.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from qcsc_workflow_utility.chem import ElectronicProperties
from sbd import closed_loop, main
from sbd.flow_params import DEParameters, FlowParameters

N = 4
NE = (2, 2)


def _ep() -> ElectronicProperties:
    rng = np.random.default_rng(3)
    h1 = rng.normal(size=(N, N))
    h2 = rng.normal(size=(N,) * 4)
    occ = (np.array([1.0, 1, 0, 0]), np.array([1.0, 1, 0, 0]))
    return ElectronicProperties(
        one_body_tensor=h1 + h1.T, two_body_tensor=h2 + h2.transpose(1, 0, 3, 2),
        t2=rng.normal(size=(2, 2, 2, 2)), initial_occupancy=occ, nuclear_repulsion_energy=1.5,
        num_orbitals=N, num_electrons=NE, open_shell=False, spin_sq=0.0,
    )


def _fake_trial(trial_index, parameters, elec_props, aa_indices, ab_indices, state):
    h1 = np.asarray(elec_props.one_body_tensor)
    e = float(np.trace(h1)) + 0.01 * len(state.carryover) - 0.001 * trial_index
    g = np.diag(np.linspace(0.9, 0.1, N)) + 0.01 * (h1 + h1.T) / 2
    res = SimpleNamespace(
        energy=e, rdm1=g, rdm1_b=g, rdm2=None, rdm2_ab=None, rdm2_bb=None,
        rdm2_spin_summed=np.zeros((N,) * 4), orbital_occupancies=(np.diag(g), np.diag(g)),
        alphadets=None, betadets=None,
    )
    new = state.copy()
    best = state.best_energy()
    new.energies = np.array([e])
    if best is None or e < best:
        new.best_index = 0
        new.carryover = np.abs(h1[: trial_index % N + 1]) > 0.5
    return new, res


def _fake_oo(elec_props, rdm1, rdm2, **kw):
    a = np.asarray(elec_props.one_body_tensor) @ rdm1
    u = expm(0.05 * (a - a.T))
    return u, float(np.trace(elec_props.one_body_tensor)) - 0.1, 1e-3


@pytest.fixture
def flow(monkeypatch):
    log = logging.getLogger("test_closed_loop_flow")
    monkeypatch.setattr(main, "get_run_logger", lambda: log)
    monkeypatch.setattr(main, "create_table_artifact", lambda *a, **k: None)
    monkeypatch.setattr(main.SBDSolverJob, "load",
                        classmethod(lambda cls, name: SimpleNamespace(method="rhf", do_rdm=1,
                                                                       solver_mode="gpu")))
    monkeypatch.setattr(main, "compute_molecular_integrals_from_fcidump",
                        lambda fcidump_file: _ep())
    monkeypatch.setattr(main, "differential_evolution_trial", _fake_trial)
    monkeypatch.setattr(main, "optimize_orbitals_rhf", _fake_oo)
    for k in ("OO_CHECK", "OO_TRUST", "OO_MAXITER", "OO_GRAD_TOL", "SBD_STATE_KEEP_EVERY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OO_SKIP_NONBEST", "0")

    def run(iterations, state_from=None, save_dir=None):
        if state_from:
            monkeypatch.setenv("SBD_STATE_FROM", str(state_from))
        else:
            monkeypatch.delenv("SBD_STATE_FROM", raising=False)
        if save_dir:
            monkeypatch.setenv("SBD_STATE_SAVE_DIR", str(save_dir))
        else:
            monkeypatch.delenv("SBD_STATE_SAVE_DIR", raising=False)
        p = FlowParameters(fcidump="unused", solver_block_ref="sbd_solver_job/x", oo_method="rhf",
                           oo_refire_every_trial=True,
                           de_params=DEParameters(num_walkers=1, iterations=iterations))
        return main.riken_sqd_de.fn(p)

    return run


def _state0(path):
    ep = _ep()
    st = main.OptimizerState.from_parameters(num_walkers=1, norb=N, n_aa_params=N - 1,
                                             n_ab_params=1, n_reps=2)
    closed_loop.write_raw(str(path), closed_loop.state_to_raw(0, ep, st, closed_loop.CumulativeRotation(N)))
    return path


def _same(a, b):
    da, db = closed_loop.read_raw(str(a)), closed_loop.read_raw(str(b))
    assert sorted(da) == sorted(db)
    for k in da:
        np.testing.assert_array_equal(da[k], db[k], err_msg=k)


def test_chained_single_trials_equal_one_run(tmp_path, flow, caplog):
    s0 = _state0(tmp_path / "state_t000.npz")
    a, b = tmp_path / "a", tmp_path / "b"
    caplog.set_level(logging.INFO, logger="test_closed_loop_flow")
    e_a = flow(4, s0, a)
    assert sorted(p.name for p in a.glob("state_t*.npz")) == [f"state_t00{i}.npz" for i in (1, 2, 3, 4)]
    caplog.clear()
    for k in range(4):
        e_b = flow(k + 1, s0 if k == 0 else b / f"state_t{k:03d}.npz", b)
    assert e_a == e_b
    for i in (1, 2, 3, 4):
        _same(a / f"state_t{i:03d}.npz", b / f"state_t{i:03d}.npz")
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("[cl-trial] trial=3 ") for m in msgs)
    assert any(m.startswith("[cl-rot] trial=3 ") for m in msgs)
    assert any("OO effect (from trial 2)" in m for m in msgs), "OO prediction lost across jobs"
    assert any(m.startswith("[state] restored state before trial 3") for m in msgs)
    _, _, st, rot, loop = closed_loop.load_state(str(b / "state_t004.npz"), main.OptimizerState)
    assert loop["oo_prediction"]["trial"] == 3
    assert not np.allclose(rot.c_a, np.eye(N))


def test_off_without_variables(tmp_path, flow, monkeypatch):
    monkeypatch.chdir(tmp_path)
    flow(2)
    assert not list(tmp_path.rglob("state_t*.npz"))
