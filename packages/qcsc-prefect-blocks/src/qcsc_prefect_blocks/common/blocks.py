from __future__ import annotations

from typing import Literal

from prefect.blocks.core import Block
from pydantic import Field


class CommandBlock(Block):
    """Command definition (what to run). Common for all HPC systems."""

    _block_type_name = "HPC Command"
    _block_type_slug = "hpc_command"

    command_name: str = Field(title="Command Name")
    executable_key: str = Field(title="Executable Key")
    description: str | None = Field(default=None, title="Description")
    default_args: list[str] = Field(default_factory=list, title="Default Args")


class ExecutionProfileBlock(Block):
    """Execution baseline (how to run). Common for all HPC systems."""

    _block_type_name = "Execution Profile"
    _block_type_slug = "execution_profile"

    profile_name: str = Field(title="Profile Name")
    command_name: str = Field(title="Command Name")
    resource_class: Literal["cpu", "gpu"] = Field(default="cpu", title="Resource Class")

    num_nodes: int = Field(default=1, gt=0, title="Nodes")
    mpiprocs: int = Field(default=1, gt=0, title="MPI Procs per Node")
    ompthreads: int | None = Field(default=None, gt=0, title="OMP Threads")
    walltime: str = Field(default="00:05:00", title="Walltime (HH:MM:SS)")
    mem: str | None = Field(
        default=None, title="Memory per Node (scheduler-specific unit, e.g. '32gb')"
    )
    launcher: Literal["single", "srun", "mpirun", "mpiexec", "mpiexec.hydra"] = Field(
        default="single",
        title="Launcher",
    )
    mpi_options: list[str] = Field(default_factory=list, title="MPI Options")
    modules: list[str] = Field(default_factory=list, title="Modules")
    pre_commands: list[str] = Field(default_factory=list, title="Pre Commands")
    environments: dict[str, str] = Field(default_factory=dict, title="Environment Variables")


class HPCProfileBlock(Block):
    """Execution target profile for local and supported HPC runtimes."""

    _block_type_name = "HPC Profile"
    _block_type_slug = "hpc_profile"

    hpc_target: Literal["local", "miyabi", "fugaku", "slurm"] = Field(
        default="miyabi", title="Execution Target"
    )

    # Queue/partition/resource-group names (terminology differs by system)
    queue_cpu: str = Field(default="regular-c", title="CPU Queue/Partition/Resource Group")
    queue_gpu: str = Field(default="regular-g", title="GPU Queue/Partition/Resource Group")

    # Project/group/account
    project_cpu: str = Field(default="", title="CPU Project/Group/Account")
    project_gpu: str = Field(default="", title="GPU Project/Group/Account")

    # Executable mapping
    executable_map: dict[str, str] = Field(default_factory=dict, title="Executable Map")

    # Slurm-specific options (ignored for other systems)
    slurm_qpu: str | None = Field(default=None, title="Slurm: QPU")
    slurm_memory: str | None = Field(default=None, title="Slurm: Memory")
    slurm_ntasks: int | None = Field(default=None, gt=0, title="Slurm: Tasks")
    slurm_gres: str | None = Field(default=None, title="Slurm: Generic Resource (--gres)")

    # Fugaku-specific options (ignored for other systems)
    gfscache: str | None = Field(default=None, title="Fugaku: GFS Cache Path (PJM_LLIO_GFSCACHE)")
    spack_modules: list[str] = Field(default_factory=list, title="Fugaku: Spack Modules")
    mpi_options_for_pjm: list[str] = Field(
        default_factory=list, title="Fugaku: MPI Options for PJM"
    )
    pjm_resources: list[str] = Field(
        default_factory=list,
        title="Fugaku: Additional PJM -L Directives",
    )
