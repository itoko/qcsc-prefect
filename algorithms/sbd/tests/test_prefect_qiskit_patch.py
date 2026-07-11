"""Regression test for the twirling schema patch (audit P1).

prefect_qiskit 0.2.0's sampler Options model has no `twirling` field and drops it silently
(extra="ignore"), so --measure-twirling never reached the device. apply_twirling_patch() must make
`twirling` survive full SamplerV2Schema validation while leaving `dynamical_decoupling` intact.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from sbd.prefect_qiskit_patch import apply_twirling_patch


def _validate_options(twirling: dict) -> dict:
    """Run a params dict through the real SamplerV2Schema and return the surviving options dump."""
    import prefect_qiskit.vendors.ibm_quantum.models as models

    qc = QuantumCircuit(1)
    qc.measure_all()
    params = {
        "pubs": [[qc, [], 100]],
        "options": {
            "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"},
            "twirling": twirling,
        },
    }
    validated = models.SamplerV2Schema.model_validate(params)
    return validated.options.model_dump(exclude_none=True)


def test_patch_makes_twirling_survive_validation():
    assert apply_twirling_patch() is True
    dump = _validate_options({"enable_gates": False, "enable_measure": True})
    # DD must still work, twirling must now survive with our exact values.
    assert dump.get("dynamical_decoupling", {}).get("enable") is True
    assert "twirling" in dump, "twirling was dropped even after the patch"
    assert dump["twirling"]["enable_measure"] is True
    assert dump["twirling"]["enable_gates"] is False


def test_patch_is_idempotent():
    assert apply_twirling_patch() is True
    assert apply_twirling_patch() is True  # second call is a no-op, still reports success
    import prefect_qiskit.vendors.ibm_quantum.models as models

    # exactly one twirling field, no duplication/corruption
    assert "twirling" in models.Options.model_fields
