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

ENV = ("SBD_DUMP_RDM_DIR", "SBD_DUMP_RDM_TRIALS", "SBD_FREEZE_SUBSPACE_AT", "SBD_BRANCH_SAVE_AT",
       "SBD_BRANCH_DIR", "SBD_BRANCH_FROM", "SBD_EARLY_OO_TRIALS", "SBD_EARLY_OO_TRUST",
       "SBD_EARLY_OO_MAXITER", "SBD_EARLY_OO_GTOL", "SBD_H2_PROBE",
       "SBD_FREEZE_SUBSPACE_FILE", "SBD_FREEZE_ORBITALS_AT", "FE4S4_CKPT_DIR",
       "SBD_DUMP_RECOVERY_STEPS", "SBD_INJECT_FILE", "SBD_INJECT_TRIALS", "SBD_INJECT_N", "SBD_EXCLUDE_EXC",
       "SBD_DUMP_ROT_DIR", "SEED")


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
        carryover_bitstrings=np.array([[True, False], [False, True]]),
        carryover_bitstrings_b=np.array([[True, True]]),
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
    np.testing.assert_array_equal(z["carryover_b"], res.carryover_bitstrings_b)
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


def _ep(n=4, seed=3):
    from qcsc_workflow_utility.chem import ElectronicProperties
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(n, n)); h1 = h1 + h1.T
    g = rng.normal(size=(n,) * 4)
    return ElectronicProperties(
        one_body_tensor=h1, two_body_tensor=g, t2=rng.normal(size=(2, 2, n, n)),
        initial_occupancy=(rng.random(n), rng.random(n)), nuclear_repulsion_energy=1.5,
        num_orbitals=n, num_electrons=(2, 2), open_shell=False, spin_sq=0.0, unrestricted=True,
        one_body_tensor_b=h1 + 0.1, two_body_tensor_ab=g + 0.2, two_body_tensor_bb=g + 0.3,
        t2_ab=rng.normal(size=(2, 2, n, n)), t2_bb=rng.normal(size=(2, 2, n, n)),
        ab_overlap=_rand_orth(n, 9),
    )


def test_branch_roundtrip(tmp_path, monkeypatch):
    from sbd.main import OptimizerState
    monkeypatch.setenv("SBD_BRANCH_DIR", str(tmp_path))
    ep = _ep()
    st = OptimizerState(energies=np.array([-1.25]), populations=np.arange(6.0).reshape(1, 6),
                        carryover=np.array([[True, False, True, False]] * 3), best_index=0)
    tr = sp.RotationTracker(4)
    tr.apply(_rand_orth(4, 1), _rand_orth(4, 2))
    path = sp.save_branch(30, ep, st, tr)
    t, ep2, st2, tr2 = sp.load_branch(path, _ep(seed=99), OptimizerState, 4)  # template is different
    assert t == 30
    for k in ("one_body_tensor", "two_body_tensor", "t2", "one_body_tensor_b", "two_body_tensor_ab",
              "two_body_tensor_bb", "t2_ab", "t2_bb", "ab_overlap"):
        np.testing.assert_array_equal(getattr(ep2, k), getattr(ep, k))
    np.testing.assert_array_equal(ep2.initial_occupancy[0], ep.initial_occupancy[0])
    assert ep2.nuclear_repulsion_energy == 1.5 and ep2.num_electrons == (2, 2)
    np.testing.assert_array_equal(st2.populations, st.populations)
    np.testing.assert_array_equal(st2.carryover, st.carryover)
    assert st2.best_index == 0 and st2.best_energy() == -1.25
    np.testing.assert_array_equal(tr2.c_a, tr.c_a)


def test_branch_save_trials(monkeypatch):
    assert sp.branch_save_trials() == set()
    monkeypatch.setenv("SBD_BRANCH_SAVE_AT", "25,30-31")
    assert sp.branch_save_trials() == {25, 30, 31}
    monkeypatch.setenv("SBD_BRANCH_SAVE_AT", "all")   # refused: would write every trial
    assert sp.branch_save_trials() == set()


def test_early_oo_overrides(monkeypatch):
    assert sp.early_oo_overrides(0) == {}                       # off by default
    monkeypatch.setenv("SBD_EARLY_OO_TRIALS", "3")
    assert sp.early_oo_overrides(0) == {}                       # window set but nothing to override
    monkeypatch.setenv("SBD_EARLY_OO_TRUST", "0.2")
    monkeypatch.setenv("SBD_EARLY_OO_MAXITER", "2000")
    monkeypatch.setenv("SBD_EARLY_OO_GTOL", "1e-6")
    assert sp.early_oo_overrides(2) == {"trust_radius": 0.2, "maxiter": 2000, "gtol": 1e-6}
    assert sp.early_oo_overrides(3) == {}                       # past the window


def test_h2_probe_switch_and_prev_dump(tmp_path, monkeypatch):
    assert not sp.h2_probe_enabled()
    monkeypatch.setenv("SBD_H2_PROBE", "1")
    assert sp.h2_probe_enabled()
    res = _fake_result()
    assert sp.dump_prev_subspace(3, res) is None            # dump dir not set -> nothing written
    monkeypatch.setenv("SBD_DUMP_RDM_DIR", str(tmp_path))
    path = sp.dump_prev_subspace(3, res)
    z = np.load(path)
    assert path.endswith("rdmdump_t003_prev.npz") and float(z["e_davidson_total"]) == -1.25
    np.testing.assert_array_equal(z["rdm2_ab"], res.rdm2_ab)


def test_recovery_step_dump(tmp_path, monkeypatch):
    res = _fake_result()
    occ = (np.full(4, 0.25), np.full(4, 0.75))
    # off without either env var
    assert not sp.recovery_dump_enabled(1)
    monkeypatch.setenv("SBD_DUMP_RECOVERY_STEPS", "1,10")
    assert not sp.recovery_dump_enabled(1)  # needs SBD_DUMP_RDM_DIR too
    monkeypatch.setenv("SBD_DUMP_RDM_DIR", str(tmp_path))
    monkeypatch.delenv("SBD_DUMP_RECOVERY_STEPS")
    assert not sp.recovery_dump_enabled(1)  # the per-trial dump alone does not turn it on
    monkeypatch.setenv("SBD_DUMP_RECOVERY_STEPS", "1,10")
    assert sp.dump_recovery_step(0, 2, res, energy_total=-1.0, occ_a=occ[0], occ_b=occ[1]) is None
    path = sp.dump_recovery_step(0, 10, res, energy_total=-1.5, occ_a=occ[0], occ_b=occ[1])
    assert path.endswith("rdmdump_t000_r010.npz")
    z = np.load(path)
    assert int(z["recovery_step"]) == 10 and float(z["e_davidson_total"]) == -1.5
    np.testing.assert_array_equal(z["rdm2_bb"], res.rdm2_bb)
    np.testing.assert_array_equal(z["alphadets"], res.alphadets)
    np.testing.assert_array_equal(z["carryover_a"], res.carryover_bitstrings)
    np.testing.assert_array_equal(z["occ_b"], occ[1])
    assert "c_a" not in z.files


def test_trial0_seed_override(monkeypatch):
    monkeypatch.delenv("SBD_TRIAL0_SEED", raising=False)
    # unchanged formula without the override
    assert [sp.walker_seed(47, t, 0) for t in range(4)] == [47, 48, 50, 53]
    monkeypatch.setenv("SBD_TRIAL0_SEED", "1001")
    assert sp.walker_seed(47, 0, 0) == 1001                   # trial 0 uses the override
    assert [sp.walker_seed(47, t, 0) for t in (1, 2, 3)] == [48, 50, 53]   # later trials unchanged


# ── Plan E: subspace injection / exclusion, rotation dump ──

def _strings(n_orb, n_el, k, rng):
    out = set()
    while len(out) < k:
        out.add(int(sum(1 << int(i) for i in rng.choice(n_orb, n_el, replace=False))))
    return np.array(sorted(out), dtype=np.int64)


def test_modify_subspace_off_by_default():
    a, b = np.array([7, 11, 13]), np.array([7, 14])
    fa, fb = sp.modify_subspace(3, a, b, 3, 3)
    assert fa is a and fb is b
    assert sp.dump_rotation(0, np.eye(2), np.eye(2), sp.RotationTracker(2)) is None


def test_exclude_singles_keeps_hf_and_others(monkeypatch):
    monkeypatch.setenv("SBD_EXCLUDE_EXC", "1")
    hf = 0b000111
    singles = [0b001011, 0b010101, 0b100110]
    doubles = [0b011001, 0b110100]
    a = np.array([hf] + singles + doubles, dtype=np.int64)
    fa, fb = sp.modify_subspace(0, a, a.copy(), 3, 3)
    assert list(fa) == [hf] + doubles and list(fb) == [hf] + doubles


def test_inject_replaces_tail_keeps_dimension(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    cur_a, cur_b = _strings(10, 4, 40, rng), _strings(10, 4, 30, rng)
    pool = [int(x) for x in _strings(10, 4, 120, rng) if x not in set(cur_a.tolist())]
    inj_a = np.array([int(cur_a[3])] + pool[:12] + [0b11], dtype=np.int64)  # one present, one wrong popcount
    inj_b = np.array(pool[:100], dtype=np.int64)
    f = tmp_path / "inj_seed43.npz"
    np.savez(f, alpha=inj_a, beta=inj_b)
    monkeypatch.setenv("SEED", "43")
    monkeypatch.setenv("SBD_INJECT_FILE", str(tmp_path / "inj_seed{SEED}.npz"))
    monkeypatch.setenv("SBD_INJECT_TRIALS", "5")
    monkeypatch.setenv("SBD_INJECT_N", "10")
    fa, fb = sp.modify_subspace(4, cur_a, cur_b, 4, 4)  # not an injection trial
    assert fa is cur_a
    fa, fb = sp.modify_subspace(5, cur_a, cur_b, 4, 4)
    assert len(fa) == len(cur_a) and len(fb) == len(cur_b)
    np.testing.assert_array_equal(fa[:30], cur_a[:30])          # head (HF/carryover side) untouched
    assert list(fa[30:]) == pool[:10]                              # first 10 new strings, in file order
    assert len(set(fa.tolist())) == len(fa)
    new_b = [x for x in fb.tolist() if x not in set(cur_b.tolist())]
    assert len(new_b) == 10 and len(set(fb.tolist())) == len(fb)
    np.testing.assert_array_equal(fb[:20], cur_b[:20])


def test_inject_requires_explicit_trials(tmp_path, monkeypatch):
    f = tmp_path / "inj.npz"
    np.savez(f, alpha=np.array([7]), beta=np.array([7]))
    monkeypatch.setenv("SBD_INJECT_FILE", str(f))
    with pytest.raises(ValueError):
        sp.modify_subspace(0, np.array([7, 11]), np.array([7, 11]), 3, 3)


def test_inject_cap_half(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    cur = _strings(10, 4, 10, rng)
    pool = [int(x) for x in _strings(10, 4, 60, rng) if x not in set(cur.tolist())]
    f = tmp_path / "inj.npz"
    np.savez(f, alpha=np.array(pool), beta=np.array(pool))
    monkeypatch.setenv("SBD_INJECT_FILE", str(f))
    monkeypatch.setenv("SBD_INJECT_TRIALS", "0")
    fa, _ = sp.modify_subspace(0, cur, cur, 4, 4)
    assert len(fa) == 10 and len(set(fa.tolist()) - set(cur.tolist())) == 4  # 10 // 2 - 1
    np.testing.assert_array_equal(fa[:6], cur[:6])


def test_dump_rotation(tmp_path, monkeypatch):
    monkeypatch.setenv("SBD_DUMP_ROT_DIR", str(tmp_path))
    tr = sp.RotationTracker(4)
    tr.apply(_rand_orth(4, 1), _rand_orth(4, 2))
    ua, ub = _rand_orth(4, 3), _rand_orth(4, 4)
    p = sp.dump_rotation(7, ua, ub, tr, e_davidson_total=-1.0, e_oo=-1.1)
    z = np.load(p)
    assert p.endswith("rot_t007.npz") and int(z["trial"]) == 7
    np.testing.assert_allclose(z["ua"], ua); np.testing.assert_allclose(z["c_b"], tr.c_b)
    assert float(z["e_oo_total"]) == -1.1
