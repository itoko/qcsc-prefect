from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qcsc_prefect_adapters.base.jinja_env import make_env
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

_ENV = make_env("qcsc_prefect_adapters.miyabi")
_TEMPLATE = "batch.pbs.j2"


@dataclass(frozen=True)
class MiyabiJobRequest:
    """Target-specific request fields required to build a Miyabi PBS job.

    Attributes:
        queue_name: Miyabi PBS queue name, for example ``"regular-c"``.
        project: Miyabi group/project name passed to the PBS template.
        executable: Absolute or scheduler-visible command path to execute.
    """

    queue_name: str
    project: str
    executable: str


def to_miyabi_template_kwargs(*, exec_profile: ExecutionProfile, req: MiyabiJobRequest) -> dict:
    """Build template variables for the Miyabi PBS script.

    Args:
        exec_profile: Scheduler-independent execution profile.
        req: Miyabi-specific scheduler request fields.

    Returns:
        A dictionary that can be passed to the Miyabi Jinja template.
    """

    kw: dict = {
        "queue_name": req.queue_name,
        "project": req.project,
        "num_nodes": exec_profile.num_nodes,
        "launcher": exec_profile.launcher,
        "executable": req.executable,
    }
    if exec_profile.mpiprocs is not None:
        kw["mpiprocs"] = exec_profile.mpiprocs
    if exec_profile.ompthreads is not None:
        kw["ompthreads"] = exec_profile.ompthreads
    if exec_profile.mem is not None:
        kw["mem"] = exec_profile.mem
    if exec_profile.walltime is not None:
        kw["walltime"] = exec_profile.walltime
    if exec_profile.modules:
        kw["modules"] = list(exec_profile.modules)
    if exec_profile.pre_commands:
        kw["pre_commands"] = list(exec_profile.pre_commands)
    if exec_profile.environments:
        kw["environments"] = dict(exec_profile.environments)
    if exec_profile.mpi_options:
        kw["mpi_options"] = list(exec_profile.mpi_options)
    if exec_profile.arguments:
        kw["arguments"] = list(exec_profile.arguments)
    return kw


def render_script(*, work_dir: Path, exec_profile: ExecutionProfile, req: MiyabiJobRequest) -> str:
    """Render a Miyabi job script text from the configured Jinja template.

    .. note::
        The template file is configured by module constant ``_TEMPLATE`` and
        is expected to be a ``.j2`` file.

    Args:
        work_dir: Working directory injected into the template.
        exec_profile: Scheduler-independent execution profile.
        req: Miyabi-specific scheduler request fields.

    Returns:
        Rendered PBS script text.
    """

    template = _ENV.get_template(_TEMPLATE)
    kwargs = to_miyabi_template_kwargs(exec_profile=exec_profile, req=req)
    return template.render(work_dir=str(work_dir), **kwargs)


def write_script_file(*, work_dir: Path, filename: str, text: str) -> Path:
    """Write a rendered job script into the work directory.

    .. note::
        This function is expected to be called inside
        `qcsc_prefect_executor.miyabi.run.run_miyabi_job`.
        Workflow authors normally do not need to call it directly.

    .. note::
        The ``text`` argument is expected to come from `render_script`,
        which renders the ``.j2`` template specified by ``_TEMPLATE``.

    Args:
        work_dir: Base working directory where the script file is created.
        filename: Script file name (for example ``batch.pbs``).
        text: Rendered script text.

    Returns:
        Absolute path to the created job script file.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / filename
    path.write_text(text)
    return path
