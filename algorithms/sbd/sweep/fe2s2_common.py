"""Shared configuration + helpers for the Fe2S2 RHF-vs-UHF multi-iteration recovery study.

One parent directory holds every run:

    sweep/runs/
      fe2s2_uhf/            <- per (molecule, method) run
        samples/           <- persisted merged sample pool (sample ONCE, reuse forever)
        recover/           <- per-recovery-step telemetry JSON (offline re-diagonalization)
        post/              <- plots + energies.csv
        run.log
      fe2s2_rhf/
        ...
      fe2s2_post/          <- combined RHF-vs-UHF plots
      refs.json            <- DMRG / UCCSD / CCSD(T) / HCI reference energies

Design rules (locked with the user):
  * RHF vs UHF is chosen by the SOLVER BLOCK (create_blocks --method), NOT a FlowParameters field.
  * Sample once per method (own ansatz), persist the pool, then re-diagonalize offline with
    quantum_source="saved" -- never re-sample the device for the recovery sweep.
  * Credentials come from the environment ONLY. Nothing secret is hardcoded or committed.
    The launcher .sh files source a gitignored sweep/.env.local for IBM_API_KEY / IBM_CRN /
    IBM_BACKEND.

This module is import-only (no side effects) so both the sampling and recovery scripts, and the
plotting script, can share the exact same path conventions.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# System under study
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Molecule registry. Select with FE_MOL (default "fe2s2" so existing Fe2S2 scripts are unchanged).
# RHF and UHF are both built from a molecule's single FCIDUMP; the difference is in the solver.
#   fe2s2: 2Fe-2S, 30e in 20 orbitals, MS2=0 -> 40 qubits
#   fe4s4: 4Fe-4S, 54e in 36 orbitals, MS2=0 -> 72 qubits
# Each molecule's FCIDUMP path can be overridden by <MOL>_FCIDUMP (e.g. FE4S4_FCIDUMP); the default
# points at the Fugaku location, and the ROQUO launchers export the local path.
# ---------------------------------------------------------------------------------------------
MOLECULES: dict[str, dict] = {
    "fe2s2": {
        "norb": 20, "nelec": 30, "ms2": 0,
        "default_fcidump": "/2ndfs/ra010014/u14924_space/sweep/fe2s2_40q.fcidump",
    },
    "fe4s4": {
        "norb": 36, "nelec": 54, "ms2": 0,
        "default_fcidump": "/2ndfs/ra010014/u14924_space/sweep/fcidump_Fe4S4_MO.txt",
    },
}

MOLECULE = os.environ.get("FE_MOL", "fe2s2").strip().lower()
if MOLECULE not in MOLECULES:
    raise ValueError(f"FE_MOL must be one of {tuple(MOLECULES)}, got {MOLECULE!r}")


def mol_config(mol: str | None = None) -> dict:
    """Return the registry entry for a molecule (defaults to the FE_MOL selection)."""
    m = (mol or MOLECULE).strip().lower()
    if m not in MOLECULES:
        raise ValueError(f"molecule must be one of {tuple(MOLECULES)}, got {m!r}")
    return MOLECULES[m]


def fcidump_path(mol: str | None = None) -> str:
    """FCIDUMP path for a molecule: <MOL>_FCIDUMP env override, else the registry default."""
    m = (mol or MOLECULE).strip().lower()
    return os.environ.get(f"{m.upper()}_FCIDUMP", mol_config(m)["default_fcidump"])


# Module-level default for the selected molecule (back-compat with scripts that read C.FCIDUMP).
FCIDUMP = fcidump_path(MOLECULE)

METHODS = ("uhf", "rhf")


# ---------------------------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------------------------
def sweep_root() -> Path:
    """The sweep/ directory (this file's parent)."""
    return Path(__file__).resolve().parent


def runs_root() -> Path:
    return sweep_root() / "runs"


def run_dir(method: str, mol: str | None = None) -> Path:
    """runs/<mol>[<RUN_SUFFIX>]_<method>/ for the given method ("uhf" | "rhf"). mol defaults to FE_MOL.

    RUN_SUFFIX (env) namespaces otherwise-identical runs so their persisted sample pools do not
    collide -- e.g. an alpha-beta coupling sweep sets RUN_SUFFIX=_ab2 / _ab3 / _ab4 to keep the
    stride-2/3/4 pools in separate run dirs (each has its own PREFECT_HOME/storage).
    """
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    m = (mol or MOLECULE).strip().lower()
    suffix = os.environ.get("RUN_SUFFIX", "").strip()
    return runs_root() / f"{m}{suffix}_{method}"


def run_subdirs(method: str, mol: str | None = None) -> dict[str, Path]:
    """Create (if missing) and return the samples/recover/post subdirs for a run."""
    base = run_dir(method, mol)
    dirs = {name: base / name for name in ("samples", "recover", "post")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def combined_post_dir(mol: str | None = None) -> Path:
    m = (mol or MOLECULE).strip().lower()
    d = runs_root() / f"{m}_post"
    d.mkdir(parents=True, exist_ok=True)
    return d


def refs_path(mol: str | None = None) -> Path:
    """Molecule-scoped references file: runs/<mol>_refs.json (fe2s2 keeps legacy runs/refs.json)."""
    m = (mol or MOLECULE).strip().lower()
    return runs_root() / ("refs.json" if m == "fe2s2" else f"{m}_refs.json")


def prefect_home(method: str, mol: str | None = None) -> Path:
    """Isolated Prefect home per run so concurrent flows never clash on one SQLite db."""
    return run_dir(method, mol) / "prefect_home"


# ---------------------------------------------------------------------------------------------
# create_blocks.py argument builder (parameterized by method)
# ---------------------------------------------------------------------------------------------
def sbd_paths() -> dict[str, str]:
    """Resolve the sbd package dir, diag binaries, and venv python from the environment.

    MY_PROJECT is set by ~/load_env.sh on Fugaku. Falls back to walking up from this file so the
    module still imports for a local dry-run.
    """
    proj = os.environ.get("MY_PROJECT")
    sbd = Path(proj) / "algorithms/sbd" if proj else sweep_root().parent
    diag = sbd / "native"
    py = sbd / ".venv/bin/python"
    return {
        "sbd": str(sbd),
        "diag": str(diag),
        "diag_uhf": str(diag / "diag_uhf"),
        "diag_rhf": str(diag / "diag"),
        "python": str(py),
    }


def build_create_blocks_cmd(
    method: str,
    *,
    work_dir: Path,
    num_nodes: int = 1,
    ompthreads: int = 48,
    queue: str = "small",
    shots: int = 5_000_000,
    n_shot_batches: int = 5,
    iteration: int = 5,
    block: int = 20,
    carryover_ratio: float = 0.5,
    carryover_type: int = 1,
    solver_timeout_seconds: int = 43_200,
    saved_samples: list[str] | None = None,
    walltime: str = "2:00:00",
) -> list[str]:
    """Assemble the create_blocks.py command for a given method.

    The only method-dependent bits are --method and the UHF binary path; everything else (shots,
    error mitigation, carryover) is held identical between RHF and UHF so the comparison is clean.
    Error mitigation follows qcsc-error-mitigation memory: DD (XY4) + measure-twirling, gate-
    twirling OFF (it fails on fractional LUCJ rzz gates).
    """
    p = sbd_paths()
    cmd = [
        p["python"], os.path.join(p["sbd"], "create_blocks.py"),
        "--hpc-target", "fugaku",
        "--method", method.lower(),
        "--solver-mode", "fugaku",
        "--project", "ra010014",
        "--group", "ra010014",
        "--queue", queue,
        "--fugaku-gfscache", "/vol0004:/vol0002",
        "--num-nodes", str(num_nodes),
        "--mpiprocs", str(num_nodes),
        "--ompthreads", str(ompthreads),
        "--walltime", walltime,
        "--carryover-ratio", str(carryover_ratio),
        "--carryover-type", str(carryover_type),
        "--solver-timeout-seconds", str(solver_timeout_seconds),
        "--work-dir", str(work_dir),
        "--shots", str(shots),
        "--n-shot-batches", str(n_shot_batches),
        "--iteration", str(iteration),
        "--block", str(block),
        "--dynamical-decoupling",
        "--dd-sequence", "XY4",
        "--measure-twirling",
        "--sbd-executable", p["diag_rhf"],
        "--sbd-executable-uhf", p["diag_uhf"],
    ]
    if saved_samples:
        cmd += ["--saved-samples", ",".join(saved_samples)]
    return cmd


def run_create_blocks(method: str, **kwargs) -> None:
    """Run create_blocks.py, streaming output, failing loudly on non-zero exit."""
    cmd = build_create_blocks_cmd(method, **kwargs)
    subprocess.run(cmd, cwd=sbd_paths()["sbd"], check=True)


def save_ibm_runner_block() -> None:
    """Persist the IBM Quantum Runtime block from environment credentials (never hardcoded)."""
    for key in ("IBM_API_KEY", "IBM_CRN", "IBM_BACKEND"):
        if not os.environ.get(key):
            raise RuntimeError(
                f"{key} is not set. Export it (the launcher sources sweep/.env.local) before "
                "sampling from the real device."
            )
    from prefect_qiskit import QuantumRuntime
    from prefect_qiskit.vendors.ibm_quantum.credentials import IBMQuantumCredentials

    QuantumRuntime(
        resource_name=os.environ["IBM_BACKEND"],
        credentials=IBMQuantumCredentials(
            api_key=os.environ["IBM_API_KEY"],
            crn=os.environ["IBM_CRN"],
        ),
    ).save("ibm-runner", overwrite=True)


def find_saved_pools(method: str) -> list[str]:
    """Return persisted sample-pool artifact paths (file:// URIs) for a run, sorted by walker.

    The persist step in walker_sqd writes one npz per (trial, walker) named
    ``raw_samples_t<NN>_w<W>_<hex>.npz`` under PREFECT_HOME/storage/sqd_data/<flow>/<run>/. We set
    PREFECT_HOME to the run dir (see prefect_home()), so pools live under
    runs/<mol>_<method>/prefect_home/storage/sqd_data/**. An empty list means "not sampled yet";
    the sampler then knows to actually call the device.
    """
    home = prefect_home(method)
    storage = home / "storage" / "sqd_data"
    if not storage.is_dir():
        return []
    pools = sorted(
        str(p) for p in storage.rglob("raw_samples_t*_w*.npz") if p.is_file()
    )
    return [f"file://{p}" for p in pools]


def samples_manifest_path(method: str) -> Path:
    """Authoritative list of persisted pool URIs written by the sampler after a device run."""
    return run_dir(method) / "samples" / "pool_manifest.json"
