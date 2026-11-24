"""sqd._saved_pool_path: which saved pool a quantum_source='saved' trial reads.

Without SBD_SAVED_PER_TRIAL the old rule holds (a pool per walker, else the first one shared by all
walkers); with it, trial i reads the file named by the format string (the Batch-100K control: trial
i uses device job i of the initial pool).
"""

from __future__ import annotations

import pytest

from sbd.sqd import _saved_pool_path


def test_default_per_walker_then_shared(monkeypatch):
    monkeypatch.delenv("SBD_SAVED_PER_TRIAL", raising=False)
    assert _saved_pool_path(["a.npz", "b.npz"], 1, 7) == "b.npz"
    assert _saved_pool_path(["a.npz"], 3, 7) == "a.npz"


def test_per_trial(monkeypatch):
    monkeypatch.setenv("SBD_SAVED_PER_TRIAL", "file:///x/pool_batch10M.batch{trial:02d}.npz")
    assert _saved_pool_path(["file:///x/pool_t000.npz"], 0, 0) == "file:///x/pool_batch10M.batch00.npz"
    assert _saved_pool_path(["file:///x/pool_t000.npz"], 0, 37) == "file:///x/pool_batch10M.batch37.npz"


def test_per_trial_needs_trial_field(monkeypatch):
    monkeypatch.setenv("SBD_SAVED_PER_TRIAL", "file:///x/pool_batch10M.npz")
    with pytest.raises(ValueError, match="trial"):
        _saved_pool_path(["a.npz"], 0, 1)
