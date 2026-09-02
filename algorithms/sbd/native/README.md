# Custom Program for Selected Basis Diagonalization

This submodule provides a custom C++ implementation for configuration recovery,
adapted from the [selected basis diagonalization sample](https://github.com/r-ccs-cms/sbd/tree/main/samples/selected_basis_diagonalization) in the SBD project.
Refer to the upstream repository for detailed information about the original software.

## Output Files

The modified program generates the following output files:
- `davidson_energy.txt`: Final energy computed by the Davidson solver
- `occ_a.txt`: Average occupancies of alpha orbitals
- `occ_b.txt`: Average occupancies of beta orbitals
- `carryover.bin`: Carryover bitstrings in `uint8` format with big-endian ordering

These files are consumed by the Prefect integration block to construct a corresponding Python data class.

## Building the Executable on Miyabi

To build the executable, run the provided build script:

```bash
bash ./build_sbd.sh
```

> [!NOTE]
> This build script is adapted for the Miyabi-C environment.
> Build in other environments may require modification of the compiler command and flags.

## Building the GPU Executable on Miyabi-G

For Miyabi-G, you can build `diag-gpu` with the dedicated shell script:

```bash
./build_sbd_gpu.sh
```

This script compiles `main.cc` directly, so a separate `CMakeLists.txt` is not required.
It defaults to a local checkout under:

```text
algorithms/sbd/native/sbd
```

and produces:

```text
algorithms/sbd/native/diag-gpu
```

> [!NOTE]
> The default compiler is `mpic++`, assuming it is backed by the Miyabi-G NVHPC toolchain.
> If the local `sbd` checkout does not exist yet, the script clones `https://github.com/r-ccs-cms/sbd.git`.
> If your site uses a different wrapper, source tree, or BLAS/LAPACK link flags, override them via `CCCOM`, `CCFLAGS`, `SYSLIB`, `SBD_DIR`, or `SBD_REPO_URL`.

## Building the Executable on Fugaku

For Fugaku, use the dedicated script:

```bash
bash ./build_sbd_fugaku.sh
```

> [!NOTE]
> Ensure a Fugaku MPI C++ compiler is available in `PATH` (for example `mpiFCCpx`).
> The script defaults to `mpiFCCpx` with Fugaku-oriented flags and `-SSL2`.
> You can override `CCCOM`, `CCFLAGS`, and `SYSLIB` via environment variables if your site requires different settings.

Upon successful compilation, an executable named `diag` will be created in this directory:

```text
algorithms/sbd/native/diag
```

Set the absolute path to this executable in the SBD block configuration file:

```toml
# algorithms/sbd/sbd_blocks.toml
sbd_executable = "/abs/path/to/qcsc-prefect/algorithms/sbd/native/diag"
```

## Building the UHF (open-shell) Executable

All three build scripts accept `UHF=1`, which appends `-D_UHF` and emits a separate `diag_uhf`
(`diag-gpu_uhf` for the GPU script) binary, leaving the restricted `diag` target untouched. The
`-D_UHF` build switches the integral container and FCIDUMP parser to the spin-resolved
(interleaved spin-orbital) layout and reads a beta determinant list distinct from alpha.

Build both restricted and unrestricted binaries side by side, e.g. on Fugaku:

```bash
bash ./build_sbd_fugaku.sh           # -> diag      (restricted, unchanged)
UHF=1 bash ./build_sbd_fugaku.sh     # -> diag_uhf  (unrestricted / open-shell)
```

The UHF binary additionally accepts `--adetfile` / `--bdetfile` to load separate alpha and beta
determinant files and writes a `carryover_b.bin` alongside `carryover.bin`. Without `--bdetfile`
it falls back to `bdet = adet` (identical to the restricted binary). Register the UHF path under a
distinct executable key (`sbd_diag_uhf`) so RHF and UHF blocks can coexist; the `SBDSolverJob`
selects it when `method="uhf"`.

### Optional environment overrides

You can override upstream repository location:

```bash
SBD_REPO_URL="https://github.com/r-ccs-cms/sbd.git" \
SBD_DIR="/path/to/local/sbd" \
bash ./build_sbd.sh
```

For Fugaku build settings:

```bash
CCCOM="mpiFCCpx" \
CCFLAGS="-Nclang -std=c++17 -stdlib=libc++ -Kfast,openmp -Xpreprocessor -fopenmp" \
SYSLIB="-SSL2" \
bash ./build_sbd_fugaku.sh
```

For Miyabi-G build settings:

```bash
SBD_REPO_URL="https://github.com/r-ccs-cms/sbd.git" \
SBD_DIR="/path/to/local/sbd" \
CCCOM="mpic++" \
CCFLAGS="-std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST" \
SYSLIB="-lblas -llapack" \
./build_sbd_gpu.sh
```

## Building the Multi-GPU (NCCL) Executable on ROQUO (GB200)

`build_sbd_gpu.sh` produces a **single-GPU** binary (`-cuda -gpu=mem:unified -DSBD_THRUST`, no NCCL,
no multi-rank path). To split the Davidson diagonalization across multiple GPUs (so subspace
dimensions that a single GPU's HBM cannot hold, e.g. `sqd_dim = 1e9`, become tractable), build the
**NCCL multi-GPU** binaries with `build_sbd_mpi.sh` (RHF) and `build_sbd_mpi_uhf.sh` (UHF). These
target a fork that adds the multi-rank / non-CUDA-aware-MPI support the upstream tree lacks.

Two build recipes are proven and tracked in this repo — pick one:

- **`build_sbd_mpi_uhf_hpcx250.sh`** (native, recommended default): links against the standalone
  `hpcx/2.50` module (HPC-X OpenMPI 5), *not* NVHPC's own bundled `comm_libs/hpcx` and *not*
  `nvhpc-openmpi/26.5` — that distinction fixed a real NCCL bootstrap ring deadlock seen at
  64–100 ranks on the wrong MPI stack. This is what every proven production chain on ROQUO
  actually runs (see job 31118 below).
- **`build_sbd_mpi_uhf_container.sh`** (container-isolated): compiles entirely inside
  `nvcr.io/nvidia/nvhpc:26.5-devel` (its own bundled `nvc++`/`hpcx-2.50`/CUDA), with zero
  involvement of ROQUO's shared module filesystem. Use this if you hit a fresh-compile issue that
  a native rebuild can't explain (e.g. a binary that resolves identical libraries/flags to a
  known-working one but still fails at MPI init) — building inside the container rules out
  anything tied to shared-module/toolchain state on the host. Otherwise, prefer the native recipe;
  the container path exists as an isolation tool, not because it's faster or better maintained.

### GenerateExcitation OMP livelock fix — `OMP_NUM_THREADS` can now be `>1`

Under `-gpu=mem:unified`, `GenerateExcitation` (`include/sbd/chemistry/tpb/helper.h`) used to
reallocate two small `std::vector<int>` scratch buffers (`cr`, `an`) on the heap every loop
iteration, inside a `#pragma omp parallel for` region. Every allocation routes through NVHPC's
managed-memory pool allocator; with tens of thousands of iterations × many OMP threads, contention
on the pool allocator's lock livelocks (observed 600+ billion CAS retries in a live hang —
[r-ccs-cms/sbd#80](https://github.com/r-ccs-cms/sbd/issues/80)). **Fix:** hoist `cr`/`an` outside
the parallel loop, one allocation per thread, reused via `OrbitalDifference`'s own
`.clear()` at the top of each call (correctness-preserving — confirmed `OrbitalDifference` always
clears both vectors before writing). Apply this patch to your `sbd_mpi/` checkout before building
either recipe above; a PR is pending upstream.

Validated at `OMP_NUM_THREADS=8` on a real 36-node/144-rank production step (~4h11m wall-clock),
matching the native `OMP_NUM_THREADS=1` per-step pace (~5h08m/step, measured from checkpoint
timestamps on job 31118) — no regression from raising the thread count.

> [!NOTE]
> `OMP_NUM_THREADS` does **not** speed up the Davidson/mult solve itself — there are zero
> `#pragma omp` directives in `davidson_thrust.h` or `mult_thrust.h` (that work is GPU-parallel via
> CUDA/Thrust, not CPU-threaded). It only affects the `GenerateExcitation`/"helper construction"
> setup phase that runs once per recovery step. Don't expect raising it to make Davidson faster —
> the actual lever for that is the MPI grid (`adet_comm_size`/`bdet_comm_size`, more nodes).

### 1. Clone the fork (not vendored in this repo)

The multi-GPU source lives in `native/sbd_mpi/` (gitignored — do not commit the ~large tree). Clone
Ryusei Wakizaka's fork, `non-cuda-aware-mpi` branch, then apply the NCCL warm-up fix and the
GenerateExcitation fix above:

```bash
cd algorithms/sbd/native
git clone --branch non-cuda-aware-mpi https://github.com/rwakizaka/sbd.git sbd_mpi
# NCCL warm-up fix: in include/sbd/chemistry/tpb/sbdiag.h the warm-up all-reduce allocates a
# full-width device_vector; on ranks holding a different-width W slice this deadlocks when ndets is
# not divisible by bdet_comm_size. Shrink the warm-up buffer to length 1 (4 sites):
#   thrust::device_vector<double> A(W.size(), 0.0)   ->   A(1, 0.0)
```

The fork's `apps/.../main.cc` writes the Prefect result files (`occ_a/occ_b.txt`,
`davidson_energy.txt`, `carryover.bin` / `carryover_b.bin`) only under `#ifdef SBD_PREFECT` — the
build scripts pass `-DSBD_PREFECT` so `SBDSolverJob` can read them back. (The fork's carryover.bin
block references an undeclared `cobits`; rewrite it to use the fork's own `co_adet`/`co_bdet`
carryover vectors before building — see `build_sbd_mpi*.sh` header.)

### 2. Build (on a GB200 compute/login node with the modules)

Native recipe (recommended default):

```bash
sbatch build_sbd_mpi_uhf_hpcx250.sh    # -> native/diag-gpu_uhf-mpi-hpcx250
```

Container-isolated recipe (only if you need to rule out shared-module/toolchain drift):

```bash
sbatch build_sbd_mpi_uhf_container.sh  # -> native/diag-gpu_uhf-mpi-container
```

The older, generic scripts (`build_sbd_mpi.sh` / `build_sbd_mpi_uhf.sh`, targeting whatever
`nvhpc`/`hpcx` module happens to resolve on `PATH`) still work but don't pin the MPI stack — use
the `_hpcx250` recipe unless you have a specific reason not to. All three use the same CMake
`nvhpc-thrust` preset with NCCL + cuBLAS + rank-distribution:

```bash
module load gcc/14 nvhpc/26.5
module load hpcx/2.50   # load AFTER nvhpc so its mpic++/PATH takes precedence over nvhpc's bundled hpcx
cmake -S . -B build/hpcx250-thrust --toolchain cmake/toolchains/nvhpc.cmake \
  -DSBD_GPU_BACKEND=thrust -DSBD_GPU_ARCH=cc100 \
  -DSBD_USE_NCCL=ON -DSBD_USE_CUBLAS=ON \
  -DSBD_USE_RANK_DISTRIBUTION=ON -DSBD_USE_BLOCK_RANK_DISTRIBUTION=ON \
  -DCMAKE_CXX_FLAGS="-D_UHF -DSBD_PREFECT -DSBD_NON_CUDA_AWARE_MPI -I<nccl_include>" \
  -DCMAKE_EXE_LINKER_FLAGS="-L<nccl>/lib -lnccl -L<cublas> -lcublas -cudalib=cublas"
cmake --build build/hpcx250-thrust --target tpb_diag
```

Verify the link: `ldd native/diag-gpu_uhf-mpi-hpcx250 | grep -iE 'nccl|mpi|cublas'` should show
`libnccl.so.2`, `libmpi.so.40` (the standalone HPC-X OpenMPI 5, not nvhpc's bundled copy — check
with `readelf -d` that `RUNPATH` points at `hpcx/2.50`'s tree), and `libcublas`.

### 3. Run it — `srun --mpi=pmix`, validated up to 36 nodes / 144 ranks

The native `_hpcx250` binary launches with plain `srun`, using the PMIx MPI plugin:

```bash
#SBATCH --nodes=32 --gres=gpu:4 --ntasks-per-node=4
module load gcc/14 nvhpc/26.5
module load hpcx/2.50
srun --mpi=pmix --gpu-bind=closest ./diag-gpu_uhf-mpi-hpcx250 \
  --task_comm_size 1 --adet_comm_size 8 --bdet_comm_size 16 \
  --block 20 --iteration 5 --tolerance 0.01 \
  --adetfile AlphaDets.bin --bdetfile BetaDets.bin --carryoverfile carryover.txt
```

This is the actual launcher a real production job (32 nodes, `a=8 × b=16 = 128` ranks) has run
continuously across 150+ recovery steps. A 36-node / `a=8 × b=18 = 144`-rank grid has also been
validated (Fe4S4 UHF at `sqd_dim=3e10`). Both far exceed the earlier single-node/4-GPU-only
testing — multi-node `srun` launches are proven, not experimental.

- The rank product `task_comm_size × adet_comm_size × bdet_comm_size × h_comm_size` must equal the
  total MPI world size (`nodes × ntasks-per-node`) — same constraint documented for Fugaku sizing.
  Pick `adet_comm_size`/`bdet_comm_size` as close to a square grid as the node count allows.
- GPU binding is automatic: `main.cc` does `cudaSetDevice(mpi_rank % cudaGetDeviceCount())`, so with
  `--gres=gpu:4` rank *i* lands on GPU *i*. Do **not** additionally pass srun's `--gpu-bind` if the
  binary already does its own device binding — `--gpu-bind=closest` above is for NUMA/PCIe
  locality, not device selection.

**Running the container-isolated binary** requires `apptainer exec`, not bare `srun` — the binary
depends on the container's own glibc/library ABI (Ubuntu 24.04, glibc 2.39) and will fail to load
on the bare host. Wrap it per-rank:

```bash
SIF="$HOME/qcsc-prefect/algorithms/sbd/native/containers/nvhpc-26.5-devel.sif"
OMPI_LIB="/opt/nvidia/hpc_sdk/Linux_aarch64/26.5/comm_libs/13.2/hpcx/hpcx-2.50/ompi5"
apptainer exec --nv \
  --bind /var/spool/slurmd:/var/spool/slurmd --bind /tmp:/tmp \
  --env "OPAL_PREFIX=${OMPI_LIB}" --env "TMPDIR=<a container-visible writable dir>" \
  --env "PATH=${OMPI_LIB}/bin:$PATH" \
  "$SIF" ./diag-gpu_uhf-mpi-container --adet_comm_size 8 --bdet_comm_size 18 ...
```

Then launch that wrapper with the same `srun --mpi=pmix --gpu-bind=closest` as above (one wrapper
invocation per rank). Three flags are required, each fixing a real failure mode hit at multi-rank
scale:

- `--bind /var/spool/slurmd:/var/spool/slurmd` — SLURM's PMIx rendezvous point lives at
  `/var/spool/slurmd/pmix.<jobid>.0/`, outside Apptainer's default bind set (`$HOME`, `/tmp`,
  `/proc`, `/sys`, `/dev`). Without this bind, every rank silently falls back to a size-1
  singleton MPI world (`rank=0/1` for every rank) instead of joining the real job — no error, just
  wrong behavior, so check `MPI_Init_thread`'s reported rank/size explicitly if debugging this.
- `--env OPAL_PREFIX=...` — the container's OpenMPI has a build-time-baked help-file path
  (`/proj/libraries/nv/...`) that doesn't exist on ROQUO; this override points it at the
  container's real install location instead.
- `--env TMPDIR=...` — OpenMPI's default session directory (`/scratch`) is read-only inside the
  container's mount namespace; point `TMPDIR` at a path under `$HOME` (bind-mounted automatically)
  instead.

This exact recipe was validated at 36 nodes / 144 ranks (`sqd_dim=3e10`, real Fe4S4 UHF recovery).

The Fe4S4 sweep launcher (`sweep/run_fe4s4_gpu_roquo.py`, gitignored — not part of this repo's
tracked source, synced separately) wires all of the above via env vars:
`FE4S4_ADET_COMM_SIZE`/`FE4S4_BDET_COMM_SIZE` for the grid, `FE4S4_HPCX250=1` to select the native
`-hpcx250` binary, and `FE4S4_CONTAINER_TALKATIVE=1` to select the container-isolated binary (this
also switches the internal launcher options to include the `apptainer exec` wrapper automatically
— no manual wrapper script needed when going through the sweep launcher).
