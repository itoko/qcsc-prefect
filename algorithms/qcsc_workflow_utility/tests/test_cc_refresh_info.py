"""refresh_cc_seed(return_info=True): the CC convergence record closed-loop SQD-OO logs per iteration.

Stretched H2O/sto-3g (7 orbitals) as RHF and as an unrestricted copy of the same integrals keeps
every solve to a fraction of a second.

  * The record matches the solve: e_tot equals the return_energy value, the per-cycle history has
    one entry per cycle, and the last dE / norm(t1,t2) are those of the last history entry.
  * A converged solve has |dE_last| below conv_tol; a solve capped by SEED_CCSD_MAX_CYCLE reports
    converged=False and cycles == max_cycle.
  * warm_start=False (PySCF's MP2 guess) converges to the same energy as the warm start.
  * The PySCF log goes into info["log"], not to stdout; without return_info the old return values
    are unchanged.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

pytest.importorskip("pyscf")

H2O = "O 0 0 0; H 0 1.5 1.16; H 0 -1.5 1.16"


@pytest.fixture
def chem(monkeypatch):
    import qcsc_workflow_utility.chem as chem

    monkeypatch.setattr(chem, "get_run_logger", lambda: mock.MagicMock())
    monkeypatch.delenv("OO_INIT", raising=False)
    monkeypatch.delenv("SEED_CCSD_MAX_CYCLE", raising=False)
    return chem


@pytest.fixture
def ep_rhf(chem):
    return chem.compute_molecular_integrals_from_geometry.fn(atom=H2O, basis="sto-3g")


@pytest.fixture
def ep_uhf(chem, ep_rhf):
    """Unrestricted copy of the RHF integrals (aa = ab = bb), zero t2 warm start."""
    ep = ep_rhf
    n = ep.num_orbitals
    na, nb = ep.num_electrons
    return chem.ElectronicProperties(
        one_body_tensor=ep.one_body_tensor, one_body_tensor_b=ep.one_body_tensor,
        two_body_tensor=ep.two_body_tensor, two_body_tensor_ab=ep.two_body_tensor,
        two_body_tensor_bb=ep.two_body_tensor,
        t2=np.zeros((na, na, n - na, n - na)), t2_ab=np.zeros((na, nb, n - na, n - nb)),
        t2_bb=np.zeros((nb, nb, n - nb, n - nb)),
        initial_occupancy=ep.initial_occupancy, nuclear_repulsion_energy=ep.nuclear_repulsion_energy,
        num_orbitals=n, num_electrons=(na, nb), open_shell=False, spin_sq=0.0, unrestricted=True,
    )


@pytest.mark.parametrize("which,method", [("rhf", "RCCSD"), ("uhf", "UCCSD")])
def test_record_matches_solve(chem, ep_rhf, ep_uhf, which, method, capsys):
    ep = ep_rhf if which == "rhf" else ep_uhf
    capsys.readouterr()
    props, e_tot, info = chem.refresh_cc_seed(ep, return_energy=True, return_info=True)
    out = capsys.readouterr().out
    assert props is not None
    assert "cycle =" not in out, "the CC log must go to info['log'], not stdout"
    assert info["method"] == method
    assert info["converged"] is True
    assert info["warm_start"] is True
    assert info["e_tot"] == pytest.approx(e_tot, abs=1e-12)
    hist = info["history"]
    assert len(hist) == info["cycles"] > 0
    assert [h[0] for h in hist] == list(range(1, len(hist) + 1))
    assert info["de_last"] == hist[-1][2]
    assert info["norm_t_last"] == hist[-1][3]
    assert abs(info["de_last"]) < info["conv_tol"]
    assert "cycle =" in info["log"]
    assert info["max_cycle"] == 200
    assert info["wall_s"] >= 0.0
    t2 = [props.t2] if which == "rhf" else [props.t2, props.t2_ab, props.t2_bb]
    assert info["t2_norm"] == pytest.approx(np.sqrt(sum(np.linalg.norm(x) ** 2 for x in t2)))


@pytest.mark.parametrize("which", ["rhf", "uhf"])
def test_capped_solve_reports_not_converged(chem, ep_rhf, ep_uhf, which, monkeypatch):
    ep = ep_rhf if which == "rhf" else ep_uhf
    monkeypatch.setenv("SEED_CCSD_MAX_CYCLE", "2")
    _, info = chem.refresh_cc_seed(ep, return_info=True, warm_start=False)
    assert info["converged"] is False
    assert info["cycles"] == info["max_cycle"] == 2
    assert len(info["history"]) == 2


@pytest.mark.parametrize("which", ["rhf", "uhf"])
def test_cold_start_reaches_same_energy(chem, ep_rhf, ep_uhf, which):
    ep = ep_rhf if which == "rhf" else ep_uhf
    _, e_warm, _ = chem.refresh_cc_seed(ep, return_energy=True, return_info=True)
    _, e_cold, info = chem.refresh_cc_seed(
        ep, return_energy=True, return_info=True, warm_start=False
    )
    assert info["warm_start"] is False and info["converged"] is True
    assert e_cold == pytest.approx(e_warm, abs=1e-7)


def test_default_returns_unchanged(chem, ep_rhf):
    props = chem.refresh_cc_seed(ep_rhf)
    assert isinstance(props, chem.ElectronicProperties)
    props2, e = chem.refresh_cc_seed(ep_rhf, return_energy=True)
    assert isinstance(props2, chem.ElectronicProperties) and isinstance(e, float)
