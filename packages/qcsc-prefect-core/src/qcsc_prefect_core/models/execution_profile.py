from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Launcher = Literal["single", "srun", "mpirun", "mpiexec", "mpiexec.hydra"]


@dataclass(frozen=True)
class ExecutionProfile:
    """
    Minimal execution profile for Miyabi PBS template.

    NOTE:
    - This is intentionally "template-driven" MVP.
    - Later you can evolve this into common+overrides and canonicalization.
    """

    command_key: str

    # PBS resources
    num_nodes: int
    mpiprocs: int | None = None
    ompthreads: int | None = None
    walltime: str | None = None  # "HH:MM:SS"
    mem: str | None = None  # scheduler-specific unit, e.g. "32gb" (PBS select chunk mem=)

    # runtime
    launcher: Launcher = "single"
    mpi_options: list[str] = field(default_factory=list)

    # environment (command-specific is OK per your design)
    modules: list[str] = field(default_factory=list)
    pre_commands: list[str] = field(default_factory=list)
    environments: dict[str, str] = field(default_factory=dict)

    # args (optional)
    arguments: list[str] = field(default_factory=list)
