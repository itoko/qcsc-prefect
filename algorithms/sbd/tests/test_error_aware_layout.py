"""Tests for the error-aware LUCJ layout + alpha-beta coupling-density knob.

These cover the two mechanisms that replace the old noise-only line/SatMapper mapping:
  1. `_build_ab_indices` -- priority-ordered alpha-beta pairs (stock stride-4 anchors first,
     densification pairs after) so a denser request degrades gracefully if the hardware can't fit it.
  2. `generate_lucj_pass_manager` -- LUCJ-aware, error-aware mapping that requests the ab pairs and
     returns the subset actually realized on the backend.
"""

from __future__ import annotations

import numpy as np
import pytest
import qiskit
from qiskit.providers.fake_provider import GenericBackendV2

from sbd.flow_params import CircuitParameters
from sbd.main import _build_ab_indices


def test_ab_stride_4_reproduces_stock_layout():
    """stride=4 must be identical to the historical [(p, p) for p in range(0, norb, 4)]."""
    for norb in (20, 36, 72):
        assert _build_ab_indices(norb, 4) == [(p, p) for p in range(0, norb, 4)]


@pytest.mark.parametrize("norb", [20, 36])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_ab_anchors_come_first(norb, stride):
    """The stock stride-4 anchors must be the leading (highest-priority) entries.

    generate_lucj_pass_manager drops from the END of the list, so anchors-first guarantees a
    denser request never loses the couplings the stock layout already relied on.
    """
    anchors = [(p, p) for p in range(0, norb, 4)]
    ab = _build_ab_indices(norb, stride)
    assert ab[: len(anchors)] == anchors
    # No duplicates, all on the diagonal, all in range.
    assert len(ab) == len(set(ab))
    assert all(p == q and 0 <= p < norb for p, q in ab)


def test_denser_stride_requests_more_pairs():
    """Smaller stride requests at least as many alpha-beta pairs (up to norb at stride 1)."""
    norb = 36
    counts = {s: len(_build_ab_indices(norb, s)) for s in (4, 3, 2, 1)}
    assert counts[4] == 9  # 36 / 4
    assert counts[1] == norb  # every orbital
    assert counts[1] >= counts[2] >= counts[4]


def test_circuit_parameters_defaults_preserve_stock_behavior():
    """Defaults must keep the pre-change behavior: stride 4, Sabre layout (error-aware off)."""
    c = CircuitParameters()
    assert c.ab_stride == 4
    assert c.use_error_aware_layout is False
    assert c.two_qubit_error_threshold == 1.0
    assert c.readout_error_threshold == 0.1
    assert c.layout_connectivity == "heavy-hex"


@pytest.mark.parametrize("stride", [4, 2, 1])
def test_error_aware_mapper_realizes_requested_ab_pairs(stride):
    """The ffsim mapper accepts our (aa, ab, bb) request and returns the realized ab subset.

    On a generic (well-connected) fake backend every requested pair should be realizable, so the
    realized set equals the requested set -- confirming the priority-ordered ab list flows through.
    """
    from ffsim import UCJOpSpinBalanced
    from ffsim.qiskit import (
        PrepareHartreeFockJW,
        UCJOpSpinBalancedJW,
        generate_lucj_pass_manager,
    )

    norb, nelec = 4, (2, 2)
    aa = [(p, p + 1) for p in range(norb - 1)]
    ab = _build_ab_indices(norb, stride)

    # A non-trivial t2 so the LUCJ gates are not optimized to identity.
    rng = np.random.default_rng(0)
    t2 = rng.standard_normal((2, 2, 2, 2)) * 0.05
    op = UCJOpSpinBalanced.from_t_amplitudes(t2, n_reps=1, interaction_pairs=(aa, ab))

    qc = qiskit.QuantumCircuit(2 * norb)
    qc.append(PrepareHartreeFockJW(norb, nelec), range(2 * norb))
    qc.append(UCJOpSpinBalancedJW(op), range(2 * norb))
    qc.measure_all()

    backend = GenericBackendV2(num_qubits=27, seed=1)
    pm, realized_ab = generate_lucj_pass_manager(
        backend, norb, "heavy-hex", (aa, ab, aa),
        two_qubit_error_threshold=1.0, readout_error_threshold=0.1,
        optimization_level=3, seed_transpiler=6538,
    )
    isa = pm.run(qc)

    assert isa is not None and isa.num_qubits >= 2 * norb
    assert len(realized_ab) == len(ab)  # generic backend fits all requested pairs
    assert set(realized_ab) == set(ab)
