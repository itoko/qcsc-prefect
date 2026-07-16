# ROQUO (GB200 GPU) Prefect Block Hello Demo

This example runs a trivial in-allocation **GPU** command on RIKEN R-CCS
**ROQUO** (GB200 / Blackwell on Arm Neoverse-V2) through Prefect Blocks, using the
Slurm executor and the shared block generator. It is the GPU counterpart of the
[Fugaku](../fugaku_prefect_hello_demo/) and Miyabi hello demos.

For the full walkthrough see
[`docs/tutorials/create_qcsc_workflow_for_roquo_gpu.md`](../../docs/tutorials/create_qcsc_workflow_for_roquo_gpu.md).

## Prerequisites

- A ROQUO account with a Slurm account code and access to the `roquo` partition
- At least one GPU available (`--gres=gpu:1`)
- `module load cuda/13.2 nvhpc/26.5 hpcx/2.50` available on compute nodes
- Prefect backend reachable (Cloud, or the ephemeral local server)

## 1) Sync dependencies

```bash
cd ~/qcsc-prefect
uv sync
```

## 2) Create the ROQUO GPU blocks

```bash
cd ~/qcsc-prefect
export ROQUO_ACCOUNT=your_account_code   # required
export ROQUO_PARTITION=roquo             # optional (default: roquo)
export ROQUO_GRES=gpu:1                  # optional (default: gpu:1)
uv run python examples/roquo_gpu_prefect_hello_demo/create_blocks.py
```

This wraps `algorithms/sbd/create_blocks.py --hpc-target slurm --solver-mode gpu`
and creates:

- `cmd-roquo-gpu-hello`
- `exec-sbd-slurm-gpu`
- `hpc-roquo`

## 3) Run the demo (from a GPU allocation)

```bash
cd ~/qcsc-prefect
sbatch --account=$ROQUO_ACCOUNT --partition=roquo --gres=gpu:1 --time=00:10:00 \
       examples/roquo_gpu_prefect_hello_demo/hello_gpu.sh
```

`hello_gpu.sh` prints `nvidia-smi` (confirming a GPU is visible in the
allocation) and then runs the flow.

## Example return value

```text
{'job_id': '12345', 'exit_status': 0, 'state': 'COMPLETED', 'work_dir': '.../work/roquo_gpu_prefect_block_hello'}
```

## Files

| File | Purpose |
| --- | --- |
| `create_blocks.py` | Create the ROQUO GPU Prefect blocks (wraps the shared generator) |
| `flow.py` | `roquo_gpu_hello_flow` — runs one GPU command via `run_job_from_blocks` |
| `hello_gpu.sh` | Slurm wrapper: `nvidia-smi` probe + run the flow |

## Environment variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `ROQUO_ACCOUNT` | Slurm `--account` code | (required) |
| `ROQUO_PARTITION` | Slurm partition | `roquo` |
| `ROQUO_GRES` | Slurm `--gres` | `gpu:1` |
| `QCSC_REPO` | repo root (for `hello_gpu.sh`) | `$HOME/qcsc-prefect` |

## Next step: the real SBD GPU sweep

Once the demo works, run the iron-sulfur recovery loop on the GPU solver:

```bash
cd ~/qcsc-prefect/algorithms/sbd/sweep
METHOD=rhf FE4S4_POOL=/path/to/raw_samples.npz \
  sbatch --account=$ROQUO_ACCOUNT run_fe4s4_gpu_roquo.sh
```

See the tutorial for the full environment-variable reference and the multi-GPU
notes.
