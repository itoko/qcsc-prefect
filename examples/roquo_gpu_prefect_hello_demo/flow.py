"""ROQUO (GB200 GPU) Prefect hello-demo flow.

Runs one trivial in-allocation GPU command on ROQUO through the block-driven
`run_job_from_blocks` entrypoint, using the blocks created by
`create_blocks.py` (`cmd-roquo-gpu-hello`, `exec-sbd-slurm-gpu`, `hpc-roquo`).
This is a smoke test for the Slurm+GPU block plumbing; the production SBD sweep
uses `algorithms/sbd/sweep/run_fe4s4_gpu_roquo.py`.
"""
from __future__ import annotations

from pathlib import Path

from prefect import flow
from qcsc_prefect_executor import run_job_from_blocks


@flow(name="roquo-gpu-prefect-block-hello-demo")
async def roquo_gpu_hello_flow(
    *,
    command_block_name: str = "cmd-roquo-gpu-hello",
    execution_profile_block_name: str = "exec-sbd-slurm-gpu",
    hpc_profile_block_name: str = "hpc-roquo",
    work_dir: str = "./work/roquo_gpu_prefect_block_hello",
    script_filename: str = "roquo_gpu_hello.slurm",
    timeout_seconds: float = 600.0,
):
    result = await run_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=Path(work_dir).expanduser().resolve(),
        script_filename=script_filename,
        watch_poll_interval=5.0,
        timeout_seconds=timeout_seconds,
        metrics_artifact_key="roquo-gpu-hello-demo-metrics",
    )
    return {
        "job_id": getattr(result, "job_id", None),
        "exit_status": getattr(result, "exit_status", None),
        "state": getattr(result, "state", None),
        "work_dir": str(Path(work_dir).expanduser().resolve()),
    }
