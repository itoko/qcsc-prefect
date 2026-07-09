# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)

from typing import Literal

from pydantic import BaseModel, Field


class CircuitParameters(BaseModel):
    """Configuration for LUCJ circuit construction."""

    n_lucj_layers: int = Field(
        default=2,
        description="Number of LUCJ circuit block repetitions.",
        title="LUCJ Circuit Layers",
        ge=1,
    )

    use_reset_mitigation: bool = Field(
        default=True,
        description="Set True to use reset error mitigation scheme.",
        title="Reset Mitigation",
    )

    optimization_level: int = Field(
        default=3,
        description="Optimization level of Qiskit transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
    )

    sabre_max_iterations: int = Field(
        default=8,
        description=(
            "The number of forward-backward routing iterations to refine the layout "
            "and reduce routing costs."
        ),
        title="Sabre Max Iteration",
        ge=1,
    )

    sabre_swap_trials: int = Field(
        default=10,
        description=(
            "The number of routing trials for each layout, refining gate placement "
            "for better routing."
        ),
        title="Sabre SWAP Trials",
        ge=1,
    )

    sabre_layout_trials: int = Field(
        default=1_024,
        description="The number of random seed trials to run layout with.",
        title="Sabre Layout Trials",
        ge=1,
    )


class DEParameters(BaseModel):
    """Configuration for differential evoluation."""

    num_walkers: int = Field(
        default=4,
        description=(
            "Number of populations for differential evolution. "
            "Differential-evolution mutation (iterations > 1) needs at least 4 walkers; "
            "a single evaluation pass (iterations = 1, e.g. a recovery-depth test) accepts 1."
        ),
        title="Walkers",
        ge=1,
    )

    iterations: int = Field(
        default=1,
        description="Number of DE optimization iterations.",
        title="Differential Evolution Iterations",
        ge=1,
    )

    randomization_factor: float = Field(
        default=0.2,
        description="Degree of ansatz parameter perturbation from CCSD amplitude.",
        title="Randomization Factor",
        ge=0,
    )

    fxc: float = Field(
        default=0.6,
        description=(
            "Factor to scale the difference between individuals when generating mutants. "
            "Controls step size and search aggressiveness (typically 0.4 to 1.0)."
        ),
        title="Scaling Factor",
        gt=0.0,
        le=1.0,
    )

    cr_prob: float = Field(
        default=0.9,
        description=(
            "Probability of mixing components from the mutant vector into the trial solution. "
            "Controls exploration vs. exploitation (0 to 1)."
        ),
        title="Crossover Rate",
        gt=0.0,
        le=1.0,
    )


class FlowParameters(BaseModel):
    """Workflow Parameters."""

    fcidump: str = Field(
        description="A path to pySCF FCIDump file of the target molecule.",
        title="FCIDump File",
    )

    sqd_dim: int = Field(
        default=100_000_0,
        description="Dimension of subsampled bitstrings for diagonalization.",
        title="SQD Subspace Dimension",
        ge=1,
    )

    n_recovery_steps: int = Field(
        default=1,
        description=(
            "Number of SQD self-consistency configuration-recovery passes per walker. "
            "Each pass re-runs configuration recovery (from the same quantum samples) using "
            "the orbital occupancies from the previous pass's diagonalization, then "
            "re-subsamples and re-diagonalizes. 1 = single pass (no self-consistency); "
            "3-5 is the canonical SQD recovery loop."
        ),
        title="SQD Recovery Steps",
        ge=1,
    )

    n_batches: int = Field(
        default=1,
        description=(
            "Number of independent subsample batches per recovery pass (the K-batch SQD scheme "
            "of arXiv:2405.05068). Each batch draws its own subspace of dimension sqd_dim from the "
            "recovered distribution and is diagonalized separately; the reported energy is the "
            "minimum over batches and the occupancies fed to the next recovery pass are the "
            "average over batches. 1 = single batch (legacy). 5-10 stabilizes the occupancy and "
            "improves the chance of capturing low-excitation configurations. Cost scales linearly "
            "with this value."
        ),
        title="SQD Batches per Recovery",
        ge=1,
    )

    seed_cisd: int = Field(
        default=0,
        description=(
            "Classically seed the SQD subspace with low-order excitations of the Hartree-Fock "
            "reference, so those determinants are ALWAYS present regardless of what the (noisy) "
            "quantum sampler produced. This is the QSCI+SD / SCI-augmentation scheme: sampled "
            "configurations are augmented with classically-generated single/double excitations to "
            "restore the low-excitation determinants that hardware noise drops or underweights. "
            "Levels: 0 = OFF (default, pure quantum-sampled subspace); 1 = SINGLES only; "
            "2 = DOUBLES only; 3 = SINGLES + DOUBLES. The seed determinants are generated per spin "
            "channel by exciting electrons from occupied into virtual orbitals, conserving each "
            "spin's electron count (so the (Na, Nb) sector and hence Sz are preserved exactly), and "
            "are forced into the subspace alongside the Hartree-Fock and carryover determinants "
            "(deduplicated). If the seed set is larger than the sqd_dim budget it is capped "
            "(singles kept first, then as many doubles as fit) so the run always proceeds at the "
            "requested sqd_dim. Purely classical integer bitstring generation -> identical on CPU "
            "and GPU, no extra QPU cost. "
            "References: Enhancing Accuracy of Quantum-Selected Configuration Interaction, "
            "J. Chem. Theory Comput. (PMC12423809); Molecular Quantum Computations on a Protein, "
            "arXiv:2512.17130; Auto-regressive NQS Sampling for Selected CI, arXiv:2603.24728."
        ),
        title="Seed CISD Level",
        ge=0,
        le=3,
    )

    seed_budget_frac: float = Field(
        default=1.0,
        description=(
            "Fraction (0 < f <= 1) of the per-spin subspace budget the CISD seed may occupy when "
            "seed_cisd > 0. f=1.0 (default) forces as much of the singles/doubles manifold as fits. "
            "f<1.0 caps the seed to f*budget and reserves the remaining slots for the sample's "
            "higher-excitation determinants (triples and above) -- 'partial-CISD + heavy mixing'. "
            "Because pure singles+doubles has a fixed energy ceiling (~CISD), admitting sampled "
            "higher excitations on top is what can push the energy below that ceiling toward "
            "CCSD(T)/DMRG. Ignored when seed_cisd == 0."
        ),
        title="Seed CISD Budget Fraction",
        gt=0.0,
        le=1.0,
    )

    quantum_source: Literal["real-device", "random", "saved"] = Field(
        default="real-device",
        description=(
            "Select the SQD sample source: 'real-device' (IBM Quantum Runtime), "
            "'random' (deterministic random bitstrings), or 'saved' (reload a previously "
            "persisted sample pool and diagonalize offline, no IBM call)."
        ),
        title="Quantum Source",
    )

    random_seed: int = Field(
        default=24,
        description="Base RNG seed used when Quantum Source is 'random'.",
        title="Random Seed",
        ge=0,
    )

    circ_params: CircuitParameters = Field(
        default_factory=CircuitParameters,
        title="Circuit Parameters",
    )

    de_params: DEParameters = Field(
        default_factory=DEParameters,
        title="Differential Evoluation Parameters",
    )

    solver_block_ref: str = Field(
        default="sbd_solver_job/davidson-solver",
        description="Solver block reference in '<block_type_slug>/<block_document_name>' format.",
    )

    # ── Orbital-optimization (two-step MCSCF) controls ───────────────────────────────────────
    # Orbital optimization runs between DE trials when the solver writes RDMs (solver.do_rdm != 0).
    # These govern the CASSCF-style stopping so the run halts at the physical minimum on its own.
    oo_grad_tol: float = Field(
        default=1e-3,
        description=(
            "Orbital-gradient convergence threshold for the two-step MCSCF loop. When the orbital "
            "gradient norm returned by optimize_orbitals falls below this, the orbitals are "
            "stationary (generalized Brillouin condition, g -> 0) and the basis is frozen for the "
            "remaining DE trials. This is the reference-free convergence criterion used by CASSCF "
            "codes (MOLCAS/ORCA/Molpro) -- it needs no near-exact (DMRG/FCI) energy floor, so it "
            "works for large systems. Default 1e-3 (ORCA-like)."
        ),
        title="OO Gradient Tolerance",
        gt=0.0,
    )
    oo_trust_radius: float = Field(
        default=0.5,
        description=(
            "Trust radius (max |rotation parameter| per spin channel) passed to optimize_orbitals "
            "each DE trial. Small values (e.g. 0.05) force small orbital steps so the two-step "
            "loop takes many micro-moves and the gradient can decrease gradually toward "
            "oo_grad_tol without over-rotating on the fixed RDMs; large values allow bigger steps "
            "but risk the fixed-RDM decoupling artifact. Default 0.5 rad."
        ),
        title="OO Trust Radius",
        gt=0.0,
    )
    oo_maxiter: int = Field(
        default=300,
        description=(
            "Max L-BFGS iterations inside a single optimize_orbitals call (one DE trial). Small "
            "values (e.g. 20-50) keep each orbital step short so RDMs are refreshed frequently by "
            "the next SQD trial (closer to proper micro/macro-iteration MCSCF); large values let a "
            "single trial optimize the orbitals fully against the fixed RDMs. Default 300."
        ),
        title="OO Max Iterations",
        ge=1,
    )
    oo_selfconsistency_tol: float = Field(
        default=0.05,
        description=(
            "Self-consistency guard (Ha) for the two-step MCSCF loop. The orbital-optimization "
            "energy is evaluated on the PREVIOUS trial's FIXED RDMs; if it drops more than this "
            "below the solver energy of the same state, the fixed RDMs have decoupled from the "
            "rotated Hamiltonian (a non-variational artifact -- the energy would appear to fall "
            "below the true ground state). The loop then stops rotating and freezes the basis. "
            "Default 0.05 Ha (50 mHa)."
        ),
        title="OO Self-Consistency Tolerance",
        gt=0.0,
    )
    oo_resolve_rdms: bool = Field(
        default=False,
        description=(
            "DOCUMENTED OPTION (not yet enabled by default). Fully self-consistent inner loop: "
            "after each orbital rotation, RE-RUN the SQD diagonalization in the rotated basis to "
            "obtain FRESH RDMs before the next orbital step, instead of reusing the previous "
            "trial's fixed RDMs. This removes the fixed-RDM approximation entirely and makes the "
            "optimization rigorously variational (the energy descends onto the true minimum from "
            "above, as in coupled/one-step CASSCF), at the cost of an extra SQD solve per orbital "
            "iteration -- much slower on quantum-sampled subspaces. The default two-step scheme "
            "(oo_resolve_rdms=False) refreshes RDMs once per DE trial and relies on the "
            "trust-radius + gradient/self-consistency stopping above to stay physical, which is "
            "the standard, faster production choice. Set True for maximum rigor on small systems "
            "or a final refinement. Implemented via resolve_orbitals_self_consistent(), which "
            "re-diagonalizes the SAME fixed CI subspace in the rotated basis IN-PROCESS "
            "(solve_fermion) each orbital step -- no GPU child job and no re-sampling, so the extra "
            "cost is one small dense CI solve per macro-iteration, not a full GPU solve."
        ),
        title="OO Re-solve RDMs (rigorous, self-consistent)",
    )
    oo_resolve_maxdim: int = Field(
        default=4_000_000,
        description=(
            "Max CI subspace size (net alpha*beta configurations) used for the in-process "
            "solve_fermion re-solves when oo_resolve_rdms=True. A full production subspace "
            "(millions of configs) makes each dense selected-CI re-solve minutes-to-hours; since "
            "the orbital rotation is well-determined by the dominant low-excitation determinants, "
            "the orbitals are converged on a truncated subspace of ~sqrt(oo_resolve_maxdim) "
            "determinants per spin (ranked by excitation level from HF) and the resulting rotated "
            "basis is applied to the full SQD run. Standard MCSCF active-space economy. Default "
            "4e6 (~2000 dets/spin -> seconds per re-solve). Raise for more accuracy, lower for "
            "speed; set to a huge value to effectively disable truncation."
        ),
        title="OO Re-solve Max Subspace",
        ge=1,
    )
