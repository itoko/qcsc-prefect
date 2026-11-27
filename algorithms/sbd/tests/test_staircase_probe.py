"""Tests for the opt-in staircase probes (sbd/staircase_probe.py).

Covers: everything is off without the env vars; trial-spec parsing; the cumulative rotation
matches successive ``rotate_electronic_properties`` calls (h_t = C^T h_0 C); the dump round-trips;
the frozen subspace is stored at T and reused after; the frozen-orbital switch; the Plan F
Fock / HCI injection (scores = exact Slater-Condon elements, count rules, cap, decoy, filters).
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
       "SBD_DUMP_ROT_DIR", "SEED", "SBD_FOCK_INJECT_EPS", "SBD_FOCK_INJECT_N", "SBD_FOCK_INJECT_FRAC",
       "SBD_FOCK_INJECT_FMAX", "SBD_FOCK_INJECT_SCORE", "SBD_FOCK_INJECT_MODE", "SBD_FOCK_INJECT_NCORE",
       "SBD_FOCK_INJECT_NCORE_D", "SBD_FOCK_INJECT_TRIALS", "SBD_FOCK_INJECT_STEPS",
       "SBD_DUMP_ORIGIN", "SBD_DUMP_ORIGIN_TRIALS", "SBD_ORIGIN_DIR")


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


# ── Plan F: Fock / HCI injection ─────────────────────────────────────────────

NORB_F, NA_F, NB_F = 8, 4, 3


def _sym_eri(n, rng, same_spin=True):
    g = rng.standard_normal((n, n, n, n)) * 0.1
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    if same_spin:
        g = g + g.transpose(2, 3, 0, 1)
    return g


@pytest.fixture
def ham():
    rng = np.random.default_rng(7)
    n = NORB_F
    h1a = rng.standard_normal((n, n)); h1a = h1a + h1a.T
    h1b = rng.standard_normal((n, n)); h1b = h1b + h1b.T
    return SimpleNamespace(one_body_tensor=h1a, one_body_tensor_b=h1b, two_body_tensor=_sym_eri(n, rng),
                           two_body_tensor_ab=_sym_eri(n, rng, same_spin=False),
                           two_body_tensor_bb=_sym_eri(n, rng), num_orbitals=n)


def _exact_h(ham, det_a, det_b, ket_a, ket_b):
    """<det|H|ket> from PySCF's UHF FCI contraction (bit i = orbital i, chemist integrals)."""
    from pyscf.fci import cistring, direct_uhf
    n, ne = NORB_F, (NA_F, NB_F)
    na, nb = cistring.num_strings(n, NA_F), cistring.num_strings(n, NB_F)
    c = np.zeros((na, nb))
    c[cistring.str2addr(n, NA_F, ket_a), cistring.str2addr(n, NB_F, ket_b)] = 1.0
    h2 = direct_uhf.absorb_h1e((ham.one_body_tensor, ham.one_body_tensor_b),
                               (ham.two_body_tensor, ham.two_body_tensor_ab, ham.two_body_tensor_bb), n, ne, 0.5)
    hc = direct_uhf.contract_2e(h2, c, n, ne)
    return hc[cistring.str2addr(n, NA_F, det_a), cistring.str2addr(n, NB_F, det_b)]


def _bool_rows(strings, norb):
    return np.array([[(int(s) >> (norb - 1 - j)) & 1 for j in range(norb)] for s in strings], dtype=bool)


def test_fock_scores_are_exact_single_elements(ham):
    """|F_ai(I, J_ref)| = |<a^+_a a_i I, J_ref|H|I, J_ref>| for both spins (integral orientation check)."""
    par_a, par_b = 0b01101001, 0b00010101          # 4 alpha / 3 beta electrons, not the reference
    ref_a, ref_b = (1 << NA_F) - 1, (1 << NB_F) - 1
    for k, par, ref_other in ((0, par_a, ref_b), (1, par_b, ref_a)):
        h1, g, oj = sp._spin_integrals(ham, k, (NB_F, NA_F)[k])
        ch, s, w = sp.score_candidates(np.array([par]), sp.make_fock_of(h1, g, oj), NORB_F)
        assert np.all(w == 1.0)
        for x, sx in zip(ch.tolist(), s.tolist()):
            e = _exact_h(ham, x, ref_other, par, ref_other) if k == 0 else _exact_h(ham, ref_other, x, ref_other, par)
            assert abs(e) == pytest.approx(sx, abs=1e-10)
        assert len(ch) == bin(par).count("1") * (NORB_F - bin(par).count("1"))


def test_hci_double_scores_are_exact(ham):
    par = 0b01101001
    ref_b = (1 << NB_F) - 1
    ch, s, w = sp.score_doubles(np.array([par, 0b00001111]), ham.two_body_tensor, NORB_F, 4)
    assert len(ch) == 2 * 6 * 6 and np.all(np.isin(w, [1.0, 0.75]))
    for x, sx, wx in list(zip(ch.tolist(), s.tolist(), w.tolist()))[:36]:
        assert bin(x).count("1") == NA_F and bin(x ^ par).count("1") == 4
        assert abs(_exact_h(ham, x, ref_b, par, ref_b)) == pytest.approx(sx / wx, abs=1e-10)


def test_carry_ints_matches_subsample_layout():
    s = [0b00001111, 0b10010011]
    rows = np.vstack([_bool_rows(s, 8), np.zeros((1, 8), bool)])   # + one zero-padding row
    np.testing.assert_array_equal(sp.carry_ints(rows, 8, 4), s)


def _inject_case(ham, rng):
    """A list [HF, carryover (12), sampled (40)] per spin and the carryover rows (weight order)."""
    from itertools import combinations

    def all_str(ne):
        return [sum(1 << i for i in c) for c in combinations(range(NORB_F), ne)]
    out = []
    for ne in (NA_F, NB_F):
        hf = (1 << ne) - 1
        pool = [x for x in all_str(ne) if x != hf]
        rng.shuffle(pool)
        co, samp = pool[:12], pool[12:12 + 20 if ne == NB_F else 12 + 40]
        out.append((np.array([hf] + co + samp, dtype=np.int64), _bool_rows(co, NORB_F)))
    return out


def test_fock_inject_off_and_filters(ham, monkeypatch):
    (la, ca), (lb, cb) = _inject_case(ham, np.random.default_rng(3))
    fa, fb = sp.fock_inject(3, 0, la, lb, ca, cb, ham, NA_F, NB_F)
    assert fa is la and fb is lb
    monkeypatch.setenv("SBD_FOCK_INJECT_EPS", "0.1"); monkeypatch.setenv("SBD_FOCK_INJECT_N", "5")
    with pytest.raises(ValueError):
        sp.fock_inject(3, 0, la, lb, ca, cb, ham, NA_F, NB_F)
    monkeypatch.delenv("SBD_FOCK_INJECT_N")
    monkeypatch.setenv("SBD_FOCK_INJECT_TRIALS", "0-2")
    assert sp.fock_inject(3, 0, la, lb, ca, cb, ham, NA_F, NB_F)[0] is la
    monkeypatch.setenv("SBD_FOCK_INJECT_TRIALS", "3"); monkeypatch.setenv("SBD_FOCK_INJECT_STEPS", "2")
    assert sp.fock_inject(3, 0, la, lb, ca, cb, ham, NA_F, NB_F)[0] is la       # recovery step 1
    assert sp.fock_inject(3, 1, la, lb, ca, cb, ham, NA_F, NB_F)[0] is not la   # recovery step 2
    empty = np.zeros((0, NORB_F), bool)                                           # trial 0: no carryover
    fa, fb = sp.fock_inject(3, 1, la, lb, empty, empty, ham, NA_F, NB_F)
    np.testing.assert_array_equal(fa, la); np.testing.assert_array_equal(fb, lb)


def test_fock_inject_eps_rule_tail_and_cap(ham, monkeypatch):
    (la, ca), (lb, cb) = _inject_case(ham, np.random.default_rng(4))
    # Make the top-scoring candidate of each spin already present (kept from an earlier injection):
    # the threshold must still come from it.
    for k, (cur, co, ne) in enumerate(((la, ca, NA_F), (lb, cb, NB_F))):
        h1, g, oj = sp._spin_integrals(ham, k, (NB_F, NA_F)[k])
        ch, s, _ = sp.score_candidates(sp.carry_ints(co, NORB_F, ne), sp.make_fock_of(h1, g, oj), NORB_F)
        cur[-1] = ch[np.argmax(s)] if ch[np.argmax(s)] not in set(cur.tolist()) else cur[-1]
    monkeypatch.setenv("SBD_FOCK_INJECT_EPS", "0.3")
    fa, fb = sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F)
    for k, (cur, co, new, ne) in enumerate(((la, ca, fa, NA_F), (lb, cb, fb, NB_F))):
        core = sp.carry_ints(co, NORB_F, ne)
        h1, g, oj = sp._spin_integrals(ham, k, (NB_F, NA_F)[k])
        ch, s, _ = sp.score_candidates(core, sp.make_fock_of(h1, g, oj), NORB_F)
        absent = ~np.isin(ch, cur)
        want = set(ch[absent & (s >= 0.3 * s.max())].tolist())     # threshold from the max over ALL candidates
        cap = min(int(0.5 * (len(cur) - 13)), len(cur) // 2 - 1)
        n_add = min(len(want), cap)
        assert n_add > 0 and len(new) == len(cur) and len(set(new.tolist())) == len(new)
        np.testing.assert_array_equal(new[:len(cur) - n_add], cur[:len(cur) - n_add])   # HF + carryover kept
        added = new[len(cur) - n_add:]
        assert set(added.tolist()) <= want and not np.isin(added, cur).any()
        if len(want) <= cap:
            assert set(added.tolist()) == want
    monkeypatch.setenv("SBD_FOCK_INJECT_EPS", "0.01")                                # rule > cap
    monkeypatch.setenv("SBD_FOCK_INJECT_FMAX", "0.1")                                # cap = 0.1 * 40 = 4
    fa, _ = sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F)
    assert len(set(fa.tolist()) - set(la.tolist())) == 4


def test_fock_inject_n_frac_decoy_hci(ham, monkeypatch):
    (la, ca), (lb, cb) = _inject_case(ham, np.random.default_rng(5))
    monkeypatch.setenv("SBD_FOCK_INJECT_N", "6")
    sel_a, _ = sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F)
    added_sel = sel_a[-6:]
    assert len(set(sel_a.tolist()) - set(la.tolist())) == 6
    monkeypatch.setenv("SBD_FOCK_INJECT_MODE", "decoy"); monkeypatch.setenv("SEED", "42")
    dec_a, _ = sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F)
    added_dec = np.array(sorted(set(dec_a.tolist()) - set(la.tolist())))
    assert 0 < len(added_dec) <= 6 and not set(added_dec.tolist()) & set(added_sel.tolist())
    core = sp.carry_ints(ca, NORB_F, NA_F)
    ch, s, _ = sp.score_candidates(core, sp.make_fock_of(*sp._spin_integrals(ham, 0, NB_F)), NORB_F)
    sc = dict(zip(ch.tolist(), s.tolist()))
    assert max(sc[x] for x in added_dec.tolist()) < min(sc[x] for x in added_sel.tolist())
    np.testing.assert_array_equal(dec_a, sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F)[0])  # reproducible
    monkeypatch.delenv("SBD_FOCK_INJECT_MODE"); monkeypatch.delenv("SBD_FOCK_INJECT_N")
    monkeypatch.setenv("SBD_FOCK_INJECT_FRAC", "0.1")                                # round(0.1 * 53) = 5
    fa, _ = sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F)
    assert len(set(fa.tolist()) - set(la.tolist())) == 5
    monkeypatch.delenv("SBD_FOCK_INJECT_FRAC")
    monkeypatch.setenv("SBD_FOCK_INJECT_N", "15"); monkeypatch.setenv("SBD_FOCK_INJECT_SCORE", "hci")
    fa, _ = sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F)
    added = set(fa.tolist()) - set(la.tolist())
    singles = {par & ~(1 << i) | (1 << a) for par in core.tolist() for i in range(NORB_F) for a in range(NORB_F)
               if (par >> i) & 1 and not (par >> a) & 1}
    assert len(added) == 15 and added - singles                                     # some same-spin doubles



# ── TR plan: Fock / HCI injection on the closed-shell (RHF) path ─────────────

NA_R = 4                                            # closed shell: 4 alpha + 4 beta in NORB_F orbitals


@pytest.fixture
def ham_r():
    rng = np.random.default_rng(11)
    h1 = rng.standard_normal((NORB_F, NORB_F)); h1 = h1 + h1.T
    return SimpleNamespace(one_body_tensor=h1, two_body_tensor=_sym_eri(NORB_F, rng), num_orbitals=NORB_F,
                           unrestricted=False, one_body_tensor_b=None, two_body_tensor_ab=None,
                           two_body_tensor_bb=None)


def _exact_h_rhf(ham, det_a, det_b, ket_a, ket_b):
    """<det|H|ket> from PySCF's spin-restricted FCI contraction (bit i = orbital i)."""
    from pyscf.fci import cistring, direct_spin1
    n, ne = NORB_F, (NA_R, NA_R)
    ns = cistring.num_strings(n, NA_R)
    c = np.zeros((ns, ns))
    c[cistring.str2addr(n, NA_R, ket_a), cistring.str2addr(n, NA_R, ket_b)] = 1.0
    h2 = direct_spin1.absorb_h1e(ham.one_body_tensor, ham.two_body_tensor, n, ne, 0.5)
    hc = direct_spin1.contract_2e(h2, c, n, ne)
    return hc[cistring.str2addr(n, NA_R, det_a), cistring.str2addr(n, NA_R, det_b)]


def test_rhf_fock_scores_are_exact_single_elements(ham_r):
    par, ref = 0b01101001, (1 << NA_R) - 1
    h1, g, oj = sp._spin_integrals(ham_r, 0, NA_R)
    assert h1 is not None and g is not None and np.allclose(oj, np.einsum("pqjj->pq", g[:, :, :NA_R, :NA_R]))
    ch, s, w = sp.score_candidates(np.array([par]), sp.make_fock_of(h1, g, oj), NORB_F)
    assert len(ch) == NA_R * (NORB_F - NA_R) and np.all(w == 1.0)
    for x, sx in zip(ch.tolist(), s.tolist()):
        assert abs(_exact_h_rhf(ham_r, x, ref, par, ref)) == pytest.approx(sx, abs=1e-10)   # alpha excitation
        assert abs(_exact_h_rhf(ham_r, ref, x, ref, par)) == pytest.approx(sx, abs=1e-10)   # same for beta


def test_rhf_hci_double_scores_are_exact(ham_r):
    par, ref = 0b01101001, (1 << NA_R) - 1
    _, g, _ = sp._spin_integrals(ham_r, 0, NA_R)
    ch, s, w = sp.score_doubles(np.array([par]), g, NORB_F, 1)
    assert len(ch) == 6 * 6
    for x, sx in zip(ch.tolist(), s.tolist()):
        assert bin(x).count("1") == NA_R and bin(x ^ par).count("1") == 4
        assert abs(_exact_h_rhf(ham_r, x, ref, par, ref)) == pytest.approx(sx, abs=1e-10)


def _inject_case_r(rng):
    """One closed-shell list [HF, carryover (12), sampled (40)] and the carryover rows."""
    from itertools import combinations
    hf = (1 << NA_R) - 1
    pool = [sum(1 << i for i in c) for c in combinations(range(NORB_F), NA_R)]
    pool = [x for x in pool if x != hf]
    rng.shuffle(pool)
    co, samp = pool[:12], pool[12:52]
    return np.array([hf] + co + samp, dtype=np.int64), _bool_rows(co, NORB_F)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *a):
        self.lines.append(msg % a)


@pytest.mark.parametrize("score", ["fock", "hci"])
def test_rhf_fock_inject_one_list_once(ham_r, monkeypatch, score):
    cur, co = _inject_case_r(np.random.default_rng(8))
    off = sp.fock_inject(4, 0, cur, cur, co, co, ham_r, NA_R, NA_R)
    assert off[0] is cur and off[1] is cur                                   # off: unchanged
    monkeypatch.setenv("SBD_FOCK_INJECT_EPS", "0.1"); monkeypatch.setenv("SBD_FOCK_INJECT_SCORE", score)
    log = _Log()
    fa, fb = sp.fock_inject(4, 0, cur, cur, co, co, ham_r, NA_R, NA_R, logger=log)
    np.testing.assert_array_equal(fa, fb)                                     # alpha list == beta list
    assert len(fa) == len(cur) and len(set(fa.tolist())) == len(fa)
    n_add = len(set(fa.tolist()) - set(cur.tolist()))
    assert n_add > 0
    np.testing.assert_array_equal(fa[:len(cur) - n_add], cur[:len(cur) - n_add])   # HF + carryover kept
    assert len(log.lines) == 1 and " spin r: +%d " % n_add in log.lines[0]   # injected once
    # The same selection as the UHF code would make for one spin with the same integrals.
    h1, g, oj = sp._spin_integrals(ham_r, 0, NA_R)
    core = sp.carry_ints(co, NORB_F, NA_R)
    ch, s, w = sp.score_candidates(core, sp.make_fock_of(h1, g, oj), NORB_F)
    if score == "hci":
        dch, ds, dw = sp.score_doubles(core[:71], g, NORB_F, len(core))
        ch, s, w = sp._merge_max(np.concatenate([ch, dch]), np.concatenate([s, ds]), np.concatenate([w, dw]))
    absent = ~np.isin(ch, cur)
    order = np.flatnonzero(absent)[np.argsort(-s[absent], kind="stable")]
    want = (s[absent] >= 0.1 * s.max()).sum()
    cap = min(int(0.5 * (len(cur) - 13)), len(cur) // 2 - 1)
    np.testing.assert_array_equal(fa[len(cur) - n_add:], ch[order[:min(want, cap)]])
    empty = np.zeros((0, NORB_F), bool)                                       # no carryover: nothing
    fa, fb = sp.fock_inject(4, 0, cur, cur, empty, empty, ham_r, NA_R, NA_R)
    np.testing.assert_array_equal(fa, cur); np.testing.assert_array_equal(fb, cur)


def test_rhf_decoy_runs(ham_r, monkeypatch):
    cur, co = _inject_case_r(np.random.default_rng(9))
    monkeypatch.setenv("SBD_FOCK_INJECT_N", "6"); monkeypatch.setenv("SBD_FOCK_INJECT_MODE", "decoy")
    monkeypatch.setenv("SEED", "42")
    fa, fb = sp.fock_inject(4, 0, cur, cur, co, co, ham_r, NA_R, NA_R)
    np.testing.assert_array_equal(fa, fb)
    assert 0 < len(set(fa.tolist()) - set(cur.tolist())) <= 6 and len(fa) == len(cur)


# ── Plan Q: protected Plan E strings, origin of the subspace strings ─────────

def test_fock_inject_keeps_protected_strings(ham, monkeypatch):
    (la, ca), (lb, cb) = _inject_case(ham, np.random.default_rng(6))
    monkeypatch.setenv("SBD_FOCK_INJECT_N", "6")
    base_a, base_b = sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F)
    # protect=None is the old behaviour; an empty protect list too
    np.testing.assert_array_equal(base_a, sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F, protect=None)[0])
    prot = (la[-3:], lb[-2:])                       # what Plan E would just have put at the tail
    fa, fb = sp.fock_inject(5, 0, la, lb, ca, cb, ham, NA_F, NB_F, protect=prot)
    for cur, new, pr in ((la, fa, prot[0]), (lb, fb, prot[1])):
        assert len(new) == len(cur) and len(set(new.tolist())) == len(new)
        assert set(pr.tolist()) <= set(new.tolist())                       # Plan E strings survive
        assert set(cur[:13].tolist()) <= set(new.tolist())                 # HF + carryover kept
        assert new[0] == cur[0]
        assert len(set(new.tolist()) - set(cur.tolist())) == 6


def test_classify_origin_priority_and_counts():
    ne = 3
    hf = 0b0111
    strings = np.array([hf, 0b1011, 0b1101, 0b1110, 0b10011, 0b10101, 0b11001, 0b100011], dtype=np.int64)
    code = sp.classify_origin(strings, num_elec=ne, inject_e=[0b100011], inject_f=[0b11001],
                              carryover=[0b1011, hf], sample=[0b1101, 0b1011], recovered=[0b1110, 0b1101],
                              seed=[0b10011])
    names = [sp.ORIGIN_NAMES[c] for c in code]
    assert names == ["hf", "carryover", "sample", "recovered", "seed", "other", "inject_f", "inject_e"]
    txt = sp.origin_counts(code)
    assert txt.startswith("total=8 ") and "carryover=1" in txt and "pre=" not in txt
    c = np.bincount(code, minlength=len(sp.ORIGIN_NAMES))
    assert c.sum() == len(strings)


def test_origin_registry_keeps_first_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("SBD_ORIGIN_DIR", str(tmp_path))
    oc = {n: i for i, n in enumerate(sp.ORIGIN_NAMES)}
    reg = sp.update_origin_registry(0, 0, (np.array([7, 11]), np.array([3])),
                                    (np.array([oc["hf"], oc["sample"]]), np.array([oc["hf"]])))
    # trial 1: 11 is now carryover (keeps "sample"), 13 is new (recovered)
    reg = sp.update_origin_registry(0, 1, (np.array([7, 11, 13]), np.array([3, 5])),
                                    (np.array([oc["hf"], oc["carryover"], oc["recovered"]]),
                                     np.array([oc["hf"], oc["inject_f"]])))
    fa = sp.first_origin(np.array([13, 11, 99]), reg[0])
    assert [sp.ORIGIN_NAMES[c] for c in fa] == ["recovered", "sample", "pre"]
    assert sp.ORIGIN_NAMES[sp.first_origin(np.array([5]), reg[1])[0]] == "inject_f"
    z = np.load(tmp_path / "origin_first_w0.npz")
    assert z["trial_a"][list(z["keys_a"]).index(13)] == 1
    monkeypatch.setenv("SBD_DUMP_ORIGIN", "1")
    assert not sp.origin_dump_enabled(1)
    monkeypatch.setenv("SBD_DUMP_ORIGIN_TRIALS", "1,5")
    assert sp.origin_dump_enabled(1) and not sp.origin_dump_enabled(2)
