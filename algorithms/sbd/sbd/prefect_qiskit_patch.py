"""Local patch: make prefect_qiskit's Sampler Options carry the ``twirling`` block.

Bug (verified 2026-07-11, prefect_qiskit 0.2.0): the sampler ``Options`` model in
``prefect_qiskit.vendors.ibm_quantum.models`` declares only ``default_shots``,
``dynamical_decoupling``, ``execution`` -- it has **no ``twirling`` field**, and its pydantic
``model_config`` uses the default ``extra="ignore"``. So when ``create_blocks.py`` writes
``options["twirling"] = {"enable_measure": True, ...}``, ``SamplerV2Schema.model_validate(params)``
(client.py) silently drops it before the payload is forwarded to the IBM Runtime REST API. Result:
every ``--measure-twirling`` run actually ran WITHOUT twirling, despite the confirming log line.

The real IBM ``qiskit_ibm_runtime.options.SamplerOptions`` DOES have a top-level ``twirling`` field,
and prefect_qiskit already ships a ``Twirling`` model (with ``enable_gates`` / ``enable_measure`` /
``num_randomizations`` / ``shots_per_randomization`` / ``strategy``) -- it is simply not wired into
``Options``. There is no upstream fix (0.2.0 is the latest release).

This module rebuilds the ``Options`` model with a ``twirling: Twirling | None`` field and
``extra="allow"`` (belt-and-suspenders), then re-points the ``SamplerV2Schema.options`` annotation at
the patched model so validation keeps ``twirling`` instead of discarding it. Import this module once,
early (create_blocks.py does), and call ``apply_twirling_patch()``. Idempotent and side-effect-free
beyond the two model rebuilds. Gate twirling stays a caller decision -- keep it OFF for fractional
LUCJ gates (IBM error 1519); measure twirling is the one we actually want ON.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_twirling_patch() -> bool:
    """Add a ``twirling`` field to prefect_qiskit's sampler Options so it reaches the device.

    Returns True if the patch was applied (or already present), False if prefect_qiskit's layout
    changed enough that the patch could not be applied (logged, non-fatal -- the caller still runs,
    just without twirling, exactly as before).
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import prefect_qiskit.vendors.ibm_quantum.models as models
        from pydantic import ConfigDict, Field
    except Exception:  # pragma: no cover - prefect_qiskit missing
        logger.warning("prefect_qiskit not importable; twirling patch skipped.")
        return False

    Options = getattr(models, "Options", None)
    Twirling = getattr(models, "Twirling", None)
    SamplerV2Schema = getattr(models, "SamplerV2Schema", None)
    if Options is None or Twirling is None or SamplerV2Schema is None:
        logger.warning(
            "prefect_qiskit models layout unexpected (Options/Twirling/SamplerV2Schema missing); "
            "twirling patch skipped -- runs will proceed WITHOUT twirling."
        )
        return False

    # Already patched upstream / by a prior call?
    if "twirling" in Options.model_fields:
        _PATCHED = True
        return True

    # Rebuild Options with the twirling field + extra="allow" (safety net for any future keys).
    # A plain subclass is the cleanest pydantic idiom; bind the concrete Twirling type directly in
    # the annotation namespace so no forward-ref resolution is needed.
    from typing import Optional

    ns = {
        "__module__": Options.__module__,
        "__annotations__": {"twirling": Optional[Twirling]},
        "twirling": Field(default=None),
        "model_config": ConfigDict(extra="allow"),
    }
    patched_options = type("Options", (Options,), ns)

    # Re-point the module attribute and the SamplerV2Schema field so validation uses the patched
    # model. Rebuild the schema so the new annotation takes effect.
    models.Options = patched_options
    try:
        SamplerV2Schema.model_fields["options"].annotation = patched_options | None
        SamplerV2Schema.model_rebuild(force=True)
    except Exception:
        logger.warning(
            "Could not rebuild SamplerV2Schema after patching Options; twirling may still be "
            "dropped. Verify with the round-trip test."
        )
        return False

    _PATCHED = True
    logger.info("Applied prefect_qiskit twirling patch (sampler Options now carries `twirling`).")
    return True
