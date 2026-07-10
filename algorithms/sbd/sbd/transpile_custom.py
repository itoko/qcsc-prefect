# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)

from time import perf_counter

from ffsim.qiskit import PRE_INIT, generate_lucj_pass_manager
from prefect import task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import RUN_ID
from prefect.logging import get_run_logger
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import Layout, Target, generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    SabreLayout,
    SetLayout,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

TRANSPILE_SEED = 6538
SLOW_LAYOUT_TRIAL_THRESHOLD = 4_096


@task
def transpile_circuit(
    circuit: QuantumCircuit,
    target: Target,
    layout: Layout,
    optimization_level: int,
) -> QuantumCircuit:
    cusotm_pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILE_SEED,
        target=target,
    )
    cusotm_pm.pre_init = PRE_INIT
    cusotm_pm.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SetLayout(
                layout=layout,
            ),
            FullAncillaAllocation(coupling_map=target.build_coupling_map()),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ]
    )
    if "rzz" in target.operation_names:
        cusotm_pm.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
                RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
            ]
        )
    return cusotm_pm.run(circuit)


# Cache only within a flow run without explicit filesystem locking.
cache_policy = RUN_ID


@task(
    cache_policy=cache_policy,
)
def find_optimal_layout(
    test_circuit: QuantumCircuit,
    target: Target,
    optimization_level: int,
    max_iterations: int,
    swap_trials: int,
    layout_trials: int,
) -> Layout:
    logger = get_run_logger()
    coupling_map = target.build_coupling_map()
    num_qubits = len(test_circuit.qubits)

    logger.info(
        "Starting SABRE layout search for %s qubits "
        "(optimization_level=%s, max_iterations=%s, swap_trials=%s, layout_trials=%s).",
        num_qubits,
        optimization_level,
        max_iterations,
        swap_trials,
        layout_trials,
    )
    if layout_trials > SLOW_LAYOUT_TRIAL_THRESHOLD:
        logger.warning(
            "SABRE layout search is configured with %s layout_trials. "
            "IBM Quantum submission starts only after this search finishes, so large values "
            "can make the closed loop look stuck on Fugaku. For troubleshooting, start around "
            "256-1024 layout trials.",
            layout_trials,
        )

    test_pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILE_SEED,
        target=target,
    )
    test_pm.pre_init = PRE_INIT
    if "rzz" in target.operation_names:
        test_pm.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
                RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
            ]
        )
    test_pm.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SabreLayout(
                coupling_map=coupling_map,
                seed=TRANSPILE_SEED,
                max_iterations=max_iterations,
                layout_trials=layout_trials,
                swap_trials=swap_trials,
            ),
            ConditionalController(
                tasks=[
                    FullAncillaAllocation(coupling_map=coupling_map),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda propset: propset["final_layout"] is None,
            ),
        ]
    )
    start = perf_counter()
    isa_trial = test_pm.run(test_circuit)
    elapsed = perf_counter() - start
    depth = isa_trial.depth(lambda inst: inst.operation.name not in ("rz", "barrier", "measure"))
    logger.info(
        "Completed SABRE layout search in %.2fs. Circuit depth = %s. Instruction counts = %s",
        elapsed,
        depth,
        dict(isa_trial.count_ops()),
    )
    final_sabre_layout = isa_trial.layout.initial_virtual_layout(filter_ancillas=True)

    layout_info = []
    for vi, pi in final_sabre_layout.get_virtual_bits().items():
        qubit = {
            "v_index": test_circuit.qubits.index(vi),
            "p_index": pi,
            "t1": None,
            "t2": None,
        }
        try:
            qubit_prop = target.qubit_properties[pi]
            qubit["t1"] = qubit_prop.t1 * 1e6
            qubit["t2"] = qubit_prop.t2 * 1e6
        except (IndexError, TypeError):
            pass
        layout_info.append(qubit)

    create_table_artifact(
        table=layout_info,
        key="isa-qubit-properties",
    )

    return final_sabre_layout


@task
def transpile_lucj_error_aware(
    circuit: QuantumCircuit,
    backend: BackendV2,
    norb: int,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    bb_indices: list[tuple[int, int]] | None,
    connectivity: str,
    two_qubit_error_threshold: float,
    readout_error_threshold: float,
    optimization_level: int,
) -> QuantumCircuit:
    """Map the LUCJ circuit with ffsim's LUCJ-aware, error-aware pass manager.

    Replaces the noise-only Sabre search (find_optimal_layout + transpile_circuit). ffsim's
    generate_lucj_pass_manager lays the ansatz onto the heavy-hex/square topology honoring the
    aa/ab/bb interaction structure, *requests* the ab (alpha-beta) coupling pairs in priority
    order (dropping the lowest-priority ones the hardware cannot accommodate), drops
    coupling-graph edges with 2q error >= two_qubit_error_threshold and qubits with readout
    error >= readout_error_threshold, and runs VF2PostLayout for a noise-aware isomorphic
    subgraph search. This is the modern replacement for line/SatMapper mapping: it minimizes
    2q + readout error *and* preserves/densifies alpha-beta coupling.

    Reference: Motta, Sung, Whaley, Head-Gordon, Shee, "Bridging physical intuition and
    hardware efficiency ... the local unitary cluster Jastrow ansatz." (2023),
    https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02516k
    """
    logger = get_run_logger()
    interaction_pairs = (
        aa_indices,
        ab_indices,
        bb_indices if bb_indices is not None else aa_indices,
    )
    start = perf_counter()
    pass_manager, realized_ab = generate_lucj_pass_manager(
        backend=backend,
        norb=norb,
        connectivity=connectivity,
        interaction_pairs=interaction_pairs,
        two_qubit_error_threshold=two_qubit_error_threshold,
        readout_error_threshold=readout_error_threshold,
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILE_SEED,
    )
    isa_circuit = pass_manager.run(circuit)
    elapsed = perf_counter() - start

    requested = len(ab_indices)
    kept = len(realized_ab)
    dropped = [p for p in ab_indices if p not in realized_ab]
    depth = isa_circuit.depth(
        lambda inst: inst.operation.name not in ("rz", "barrier", "measure")
    )
    logger.info(
        "Error-aware LUCJ layout (%s): requested %s alpha-beta pairs, hardware kept %s%s. "
        "Thresholds: 2q_err>=%.3g dropped, readout_err>=%.3g dropped. "
        "Completed in %.2fs. Circuit depth = %s. Instruction counts = %s",
        connectivity,
        requested,
        kept,
        f" (dropped {dropped})" if dropped else "",
        two_qubit_error_threshold,
        readout_error_threshold,
        elapsed,
        depth,
        dict(isa_circuit.count_ops()),
    )
    create_table_artifact(
        table=[
            {
                "requested_ab_pairs": requested,
                "realized_ab_pairs": kept,
                "dropped_ab_pairs": str(dropped),
                "connectivity": connectivity,
                "two_qubit_error_threshold": two_qubit_error_threshold,
                "readout_error_threshold": readout_error_threshold,
                "isa_depth": depth,
            }
        ],
        key="lucj-error-aware-layout",
    )
    return isa_circuit
