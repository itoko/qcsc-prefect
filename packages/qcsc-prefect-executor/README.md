# QCSC Prefect Executor

Prefect executor integrations for local processes and QCSC HPC scheduler workflows.

## Local execution

Set `HPCProfileBlock.hpc_target` to `"local"` and map the command's executable
key to a path or command available on the Prefect worker. Local execution does
not generate a job script and does not call a scheduler. It invokes the command
directly with `cwd=work_dir`.

With `launcher="single"`, the command is:

```text
executable default_args... user_args...
```

With another launcher, it is:

```text
launcher mpi_options... executable default_args... user_args...
```

`environments` are merged into the worker process environment. Values are
passed literally without shell expansion. `modules` and `pre_commands` are
explicitly unsupported for local execution and cause `ValueError` before the
process starts.

## Bulk execution

`BulkJobSpec` supports optional per-job `execution_profile_block` and
`hpc_profile_block` overrides. `None` uses the runner/API default blocks. The
single-submit bulk paths use these effective blocks for submission and group
monitoring by effective `hpc_profile_block`; native PJM bulk mode rejects per-job
block overrides because one generated script/profile is shared by all subjobs.
