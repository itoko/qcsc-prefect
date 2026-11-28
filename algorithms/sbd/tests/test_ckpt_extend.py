"""walker_sqd end to end on a tiny closed-shell system (PLAN_trypsin_rhf_fock.md §4-2, §4-3).

The SBD solver is replaced by PySCF selected CI (qiskit_addon_sqd.fermion.solve_fermion) and the
Prefect tasks are called through ``.fn``; everything else is the production walker_sqd. Checks:

* FE4S4_CKPT_DIR extension: a CR-only run stopped after 2 recovery steps and resumed to 4 gives
  bit-identical step 3 / 4 energies to an uninterrupted 4-step run (walker_rng state and the next
  step's carryover come from the ckpt), also with the closed-shell Fock injection on;
* an old ckpt without rng_state still resumes (rng from the seed);
* the closed-shell Fock injection runs from recovery step 2 ("spin r") and keeps the dimension.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from qcsc_workflow_utility import chem
from qiskit_addon_sqd.fermion import solve_fermion

from sbd import solver_job, sqd

NORB, NEL = 6, 3


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args, **kw):
        self.lines.append(msg % args if args else msg)

    warning = debug = error = info

    def exception(self, msg, *args, **kw):
        raise AssertionError(msg % args if args else msg)


class _FakeSolver:
    """async run(...) -> SBDResult from PySCF selected CI on the given (alpha, beta) lists."""

    def __init__(self, root_dir, n_carry=4):
        self.root_dir = str(root_dir)                       # where walker_sqd puts its summary plot
        self.n_carry = n_carry

    async def run(self, ci_strings, one_body_tensor, two_body_tensor, norb, nelec, **kw):
        ca, cb = (np.asarray(x, dtype=np.int64) for x in ci_strings)
        e, state, occs, _ = solve_fermion((ca, cb), one_body_tensor, two_body_tensor)
        amp = np.abs(np.asarray(state.amplitudes)).max(axis=1)
        top = ca[np.argsort(-amp, kind="stable")[: self.n_carry]]
        rows = np.array([[(int(s) >> (norb - 1 - j)) & 1 for j in range(norb)] for s in top], dtype=bool)
        return solver_job.SBDResult(energy=float(e), orbital_occupancies=(np.asarray(occs[0]), np.asarray(occs[1])),
                                    carryover_bitstrings=rows)


def _props():
    rng = np.random.default_rng(5)
    h1 = rng.standard_normal((NORB, NORB)) * 0.3
    h1 = h1 + h1.T - np.diag(np.arange(NORB, 0, -1) * 0.8)
    g = rng.standard_normal((NORB,) * 4) * 0.05
    g = g + g.transpose(1, 0, 2, 3); g = g + g.transpose(0, 1, 3, 2); g = g + g.transpose(2, 3, 0, 1)
    g += np.einsum("pq,rs->pqrs", np.eye(NORB), np.eye(NORB)) * 0.5
    return chem.ElectronicProperties(
        one_body_tensor=h1, two_body_tensor=g, t2=np.zeros((NEL, NEL, NORB - NEL, NORB - NEL)),
        initial_occupancy=chem.hf_occupancy(NORB, (NEL, NEL)), nuclear_repulsion_energy=1.25,
        num_orbitals=NORB, num_electrons=(NEL, NEL), open_shell=False, spin_sq=0.0)


@pytest.fixture
def run(monkeypatch, tmp_path):
    for k in list(os.environ):
        if k.startswith(("SBD_", "FE4S4_")):
            monkeypatch.delenv(k)
    log = _Log()
    monkeypatch.setattr(sqd, "get_run_logger", lambda: log)
    monkeypatch.setattr(sqd.SBDSolverJob, "load", staticmethod(lambda name: _FakeSolver(tmp_path)))
    monkeypatch.setattr(sqd.Variable, "get", staticmethod(lambda *a, **k: {"params": {"shots": 3000}}))
    monkeypatch.setattr(sqd, "save_ndarray", lambda **kw: "mem://unused")
    for name in ("recover_configurations", "postselect_bitstrings", "subsample_close_shell"):
        monkeypatch.setattr(sqd, name, getattr(sqd, name).fn)

    def _run(n_steps, ckpt=None, seed=11):
        if ckpt is not None:
            monkeypatch.setenv("FE4S4_CKPT_DIR", str(ckpt))
        log.lines.clear()
        (best, _, _), tel = sqd.walker_sqd.fn(
            trial_index=0, walker_index=0, ucj_parameter=np.zeros(1), circuit_params=None,
            elec_props=_props(), aa_indices=[], ab_indices=[], carryover=np.zeros((0, NORB), bool),
            sqd_dim=64, solver_block_name="fake", quantum_source="random", random_seed=seed,
            n_recovery_steps=n_steps)
        return [t["energy"] for t in tel["recovery_trace"]], list(log.lines)
    return _run


@pytest.mark.parametrize("fock", [False, True])
def test_cronly_extension_is_bit_identical(run, tmp_path, monkeypatch, fock):
    if fock:
        monkeypatch.setenv("SBD_FOCK_INJECT_EPS", "0.1")
    full, _ = run(4, tmp_path / "a")
    assert len(full) == 4 and len(set(full)) > 1            # the steps differ (the test has teeth)
    first, _ = run(2, tmp_path / "b")
    assert first == full[:2]
    rest, lines = run(4, tmp_path / "b")
    assert any("[ckpt] RESUME" in x and "(rng restored)" in x for x in lines)
    assert rest == full[2:]
    if fock:
        fi = [x for x in lines if x.startswith("[fock-inject]")]
        assert fi and all(" spin r: " in x for x in fi)


def test_old_ckpt_without_rng_still_resumes(run, tmp_path):
    run(2, tmp_path)
    p = tmp_path / "ckpt_t00_w0.npz"
    with np.load(p) as z:
        old = {k: z[k] for k in ("occ_a", "occ_b", "next_step", "best_energy", "carryover")}
    np.savez(p, **old)
    rest, lines = run(4, tmp_path)
    assert len(rest) == 2 and any("(rng from seed)" in x for x in lines)


def test_closed_shell_fock_inject_in_walker(run, monkeypatch):
    base, lines0 = run(3)
    assert not any(x.startswith("[fock-inject]") for x in lines0)
    monkeypatch.setenv("SBD_FOCK_INJECT_EPS", "0.1")
    fock, lines = run(3)
    fi = [x for x in lines if x.startswith("[fock-inject]")]
    assert len(fi) == 3 and all(" spin r: " in x for x in fi)               # once per step
    assert "step 1 spin r: no carryover core" in fi[0]
    assert "step 2 spin r: +" in fi[1] and "step 3 spin r: +" in fi[2]
    assert fock[0] == base[0]                                                   # step 1: no carryover yet
    dims = [int(m.group(1)) for x in lines for m in [re.search(r"net dim = (\d+)", x)] if m]
    dims0 = [int(m.group(1)) for x in lines0 for m in [re.search(r"net dim = (\d+)", x)] if m]
    assert dims == dims0                                                        # dimension unchanged
