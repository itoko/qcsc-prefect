from __future__ import annotations

from pathlib import Path

from qcsc_prefect_adapters.miyabi.builder import MiyabiJobRequest, render_script
from qcsc_prefect_core.models.execution_profile import ExecutionProfile


def test_render_miyabi_script(tmp_path: Path):
    profile = ExecutionProfile(
        command_key="hello",
        num_nodes=2,
        mpiprocs=16,
        walltime="00:10:00",
        launcher="mpirun",
        modules=["intelmpi"],
        pre_commands=["unset OMPI_MCA_mca_base_env_list"],
        environments={"OMP_NUM_THREADS": "1"},
        arguments=["--foo", "bar"],
    )
    req = MiyabiJobRequest(queue_name="normal", project="z30541", executable="/path/to/hello")

    text = render_script(work_dir=tmp_path, exec_profile=profile, req=req)

    assert f"#PBS -q {req.queue_name}" in text
    assert f"#PBS -l select={profile.num_nodes}" in text
    assert f":mpiprocs={profile.mpiprocs}" in text
    assert ":ompthreads=" not in text
    assert f"#PBS -l walltime={profile.walltime}" in text
    assert f"#PBS -W group_list={req.project}" in text
    assert str(tmp_path) in text
    assert "module load intelmpi" in text
    assert "unset OMPI_MCA_mca_base_env_list" in text
    assert 'export OMP_NUM_THREADS="1"' in text
    assert 'QCSC_PREFECT_EXECUTABLE="/path/to/hello"' in text
    assert "QCSC Prefect preflight failed: executable '${QCSC_PREFECT_EXECUTABLE}'" in text
    assert req.executable in text
    assert text.index("unset OMPI_MCA_mca_base_env_list") < text.index('export OMP_NUM_THREADS="1"')
    assert text.index('export OMP_NUM_THREADS="1"') < text.index("QCSC_PREFECT_EXECUTABLE")
    assert 'mpirun "${QCSC_PREFECT_EXECUTABLE}" --foo bar' in text


def test_render_miyabi_script_with_mem(tmp_path: Path):
    profile = ExecutionProfile(
        command_key="hello",
        num_nodes=2,
        mpiprocs=1,
        ompthreads=1,
        mem="32gb",
        launcher="mpirun",
    )
    req = MiyabiJobRequest(queue_name="normal", project="z30541", executable="/path/to/hello")

    text = render_script(work_dir=tmp_path, exec_profile=profile, req=req)

    assert f"#PBS -l select={profile.num_nodes}:mpiprocs=1:ompthreads=1:mem=32gb" in text


def test_render_miyabi_script_without_mem_omits_clause(tmp_path: Path):
    profile = ExecutionProfile(
        command_key="hello",
        num_nodes=2,
        mpiprocs=1,
        launcher="mpirun",
    )
    req = MiyabiJobRequest(queue_name="normal", project="z30541", executable="/path/to/hello")

    text = render_script(work_dir=tmp_path, exec_profile=profile, req=req)

    assert ":mem=" not in text
