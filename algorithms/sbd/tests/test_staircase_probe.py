"""Tests for the opt-in staircase probes (sbd/staircase_probe.py).

Covers: everything is off without the env vars; trial-spec parsing; the cumulative rotation
matches successive ``rotate_electronic_properties`` calls (h_t = C^T h_0 C); the dump round-trips;
the frozen subspace is stored at T and reused after; the frozen-orbital switch.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg

from sbd import staircase_probe as sp

ENV = ("SBD_DUMP_RDM_DIR", "SBD_DUMP_RDM_TRIALS", "SBD_FREEZE_SUBSPACE_AT",
       "SBD_FREEZE_SUBSPACE_FILE", "SBD_FREEZE_ORBITALS_AT", "FE4S4_CKPT_DIR")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def _rand_orth(n, seed):
    a = np.random.default_rng(seed).normal(size=(n, n)) * 0.3
    return scipy.linalg.expm(a - a.T)


def _fake_result(norb=4):
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        energy=-1.25,
        rdm1=rng.normal(size=(norb, norb)), rdm1_b=rng.normal(size=(norb, norb)),
        rdm2=rng.normal(size=(norb,) * 4), rdm2_ab=rng.normal(size=(norb,) * 4),
        rdm2_bb=rng.normal(size=(norb,) * 4),
        alphadets=np.array([3, 5, 6]), betadets=np.array([3, 9]),
        orbital_occupancies=(np.full(norb, 0.5), np.full(norb, 0.5)),
    )


def test_all_off_by_default():
    assert not sp.dump_enabled(0)
    assert not sp.orbitals_frozen(100)
    a, b = np.array([1, 2]), np.array([3])
    fa, fb = sp.freeze_subspace(50, "r0_b0", a, b)
    assert fa is a and fb is b
    assert sp.dump_trial(0, _fake_result(), sp.RotationTracker(4), e_davidson_total=0.0) is None


@pytest.mark.parametrize("spec,expect", [
    ("all", None), ("", None), (None, None), ("30", {30}),
    ("25-27", {25, 26, 27}), ("1,3-4", {1, 3, 4}),
])
def test_parse_trials(spec, expect):
    assert sp.parse_trials(spec) == expect


def test_cumulative_rotation_matches_rotate_electronic_properties():
    from qcsc_workflow_utility.chem import ElectronicProperties
    from qcsc_workflow_utility.orbital_opt import rotate_electronic_properties

    n = 4
    rng = np.random.default_rng(1)
    h1 = rng.normal(size=(n, n)); h1 = h1 + h1.T
    g = rng.normal(size=(n,) * 4)
    ep = ElectronicProperties(
        one_body_tensor=h1, two_body_tensor=g, t2=np.zeros((2, 2, n, n)),
        initial_occupancy=(np.ones(2), np.ones(2)), nuclear_repulsion_energy=0.0,
        num_orbitals=n, num_electrons=(2, 2), open_shell=False, spin_sq=0.0, unrestricted=True,
        one_body_tensor_b=h1.copy(), two_body_tensor_ab=g.copy(), two_body_tensor_bb=g.copy(),
        t2_ab=np.zeros((2, 2, n, n)), t2_bb=np.zeros((2, 2, n, n)),
    )
    tr = sp.RotationTracker(n)
    cur = ep
    for k in range(3):
        ua, ub = _rand_orth(n, 10 + k), _rand_orth(n, 20 + k)
        cur = rotate_electronic_properties(cur, ua, ub)
        tr.apply(ua, ub)
    np.testing.assert_allclose(cur.one_body_tensor, tr.c_a.T @ h1 @ tr.c_a, atol=1e-12)
    np.testing.assert_allclose(cur.one_body_tensor_b, tr.c_b.T @ h1 @ tr.c_b, atol=1e-12)
    # ab_overlap is the alpha/beta orbital overlap, = C_a^T C_b from identity
    np.testing.assert_allclose(cur.ab_overlap, tr.c_a.T @ tr.c_b, atol=1e-12)


def test_dump_roundtrip_and_trial_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("SBD_DUMP_RDM_DIR", str(tmp_path))
    monkeypatch.setenv("SBD_DUMP_RDM_TRIALS", "2-3")
    res, tr = _fake_result(), sp.RotationTracker(4)
    tr.apply(_rand_orth(4, 1), _rand_orth(4, 2))
    assert sp.dump_trial(1, res, tr, e_davidson_total=-1.0) is None  # filtered out
    ua = _rand_orth(4, 3)
    path = sp.dump_trial(2, res, tr, e_davidson_total=-1.25, ua=ua, ub=ua, e_oo=-1.3)
    z = np.load(path)
    assert int(z["trial"]) == 2 and float(z["e_davidson_total"]) == -1.25
    np.testing.assert_array_equal(z["rdm2_ab"], res.rdm2_ab)
    np.testing.assert_array_equal(z["betadets"], res.betadets)
    np.testing.assert_allclose(z["c_a"], tr.c_a)
    np.testing.assert_allclose(z["ua"], ua)
    assert str(z["rdm2_notation"]) == "prqs"


def test_freeze_subspace(tmp_path, monkeypatch):
    monkeypatch.setenv("SBD_FREEZE_SUBSPACE_AT", "5")
    monkeypatch.setenv("FE4S4_CKPT_DIR", str(tmp_path))
    a4, b4 = np.array([1, 2]), np.array([3, 4])
    assert sp.freeze_subspace(4, "r0_b0", a4, b4)[0] is a4          # before T: untouched
    a5, b5 = np.array([7, 8, 9]), np.array([10])
    fa, fb = sp.freeze_subspace(5, "r0_b0", a5, b5)                  # at T: stored, returned
    np.testing.assert_array_equal(fa, a5)
    assert (tmp_path / "frozen_subspace.npz").exists()
    fa, fb = sp.freeze_subspace(9, "r0_b0", np.array([0]), np.array([0]))  # after T: reused
    np.testing.assert_array_equal(fa, a5)
    np.testing.assert_array_equal(fb, b5)


def test_freeze_orbitals(monkeypatch):
    monkeypatch.setenv("SBD_FREEZE_ORBITALS_AT", "30")
    assert not sp.orbitals_frozen(29)
    assert sp.orbitals_frozen(30) and sp.orbitals_frozen(80)
