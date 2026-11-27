"""SBD_DUMP_WF (Plan Q, opt-in): the solver gets --dump_matrix_form_wf wf_matrix.bin only when set."""

from __future__ import annotations

from types import SimpleNamespace

from sbd.solver_job import _build_solver_args


def _solver(**kw):
    base = dict(task_comm_size=1, adet_comm_size=1, bdet_comm_size=1, block=20, iteration=1,
                tolerance=0.005, carryover_ratio=0.5, carryover_type=1, do_rdm=1, solver_mode="gpu",
                method="uhf", user_args=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_dump_wf_off_by_default(monkeypatch):
    monkeypatch.delenv("SBD_DUMP_WF", raising=False)
    assert "--dump_matrix_form_wf" not in _build_solver_args(_solver())
    monkeypatch.setenv("SBD_DUMP_WF", "0")
    assert "--dump_matrix_form_wf" not in _build_solver_args(_solver())


def test_dump_wf_binary(monkeypatch):
    monkeypatch.setenv("SBD_DUMP_WF", "1")
    args = _build_solver_args(_solver())
    i = args.index("--dump_matrix_form_wf")
    assert args[i + 1] == "wf_matrix.bin"          # ".bin" -> binary float64, not the huge text dump
    assert args.count("--dump_matrix_form_wf") == 1
