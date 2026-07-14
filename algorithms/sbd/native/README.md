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

### 1. Clone the fork (not vendored in this repo)

The multi-GPU source lives in `native/sbd_mpi/` (gitignored — do not commit the ~large tree). Clone
Ryusei Wakizaka's fork, `non-cuda-aware-mpi` branch, then apply the NCCL warm-up fix:

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

```bash
sbatch build_sbd_mpi.sh        # -> native/diag-gpu-mpi       (RHF / restricted)
sbatch build_sbd_mpi_uhf.sh    # -> native/diag-gpu_uhf-mpi   (UHF / open-shell, adds -D_UHF)
```

Both use the CMake `nvhpc-thrust` preset with NCCL + cuBLAS + rank-distribution:

```bash
module load cuda/13.2 nvhpc/26.5
cmake --preset nvhpc-thrust \
  -DSBD_USE_NCCL=ON -DSBD_USE_CUBLAS=ON \
  -DSBD_USE_RANK_DISTRIBUTION=ON -DSBD_USE_BLOCK_RANK_DISTRIBUTION=ON \
  -DCMAKE_CXX_FLAGS="-DSBD_PREFECT -DSBD_NON_CUDA_AWARE_MPI -I<nccl_include>" \
  -DCMAKE_EXE_LINKER_FLAGS="-L<nccl>/lib -lnccl -L<cublas> -lcublas -cudalib=cublas"
cmake --build build/nvhpc-thrust --target tpb_diag        # add -D_UHF to CXX_FLAGS for the UHF binary
```

Verify the link: `ldd native/diag-gpu-mpi | grep -iE 'nccl|mpi|cublas'` should show
`libnccl.so.2`, `libmpi.so.40` (HPC-X OpenMPI 5), and `libcublas`.

### 3. Run it — **use `mpirun`, not `srun`**

**Critical:** ROQUO's `srun` cannot bootstrap the HPC-X OpenMPI 5 that these binaries link — under
every available `--mpi` plugin (`none`, `pmi2`, `cray_shasta`) each rank initializes as its own
`MPI_COMM_WORLD` of size 1, and the solver aborts with
`std::invalid_argument: MPI Size of twister is not a square of a integer`. Only `mpirun -np N`
(from the loaded `hpcx` module) gives a real N-rank world. This is what the tutorial's
"Open MPI fails with PMI / PMIx errors under `srun`" note means — set `launcher="mpirun"`.

For a 4-GPU single-node run, request a full node and split the beta-determinant communicator:

```bash
#SBATCH --gres=gpu:4 --ntasks-per-node=4
module load cuda/13.2 nvhpc/26.5            # puts hpcx mpirun on PATH
mpirun -np 4 ./diag-gpu_uhf-mpi \
  --task_comm_size 1 --adet_comm_size 1 --bdet_comm_size 4 \
  --block 20 --iteration 5 --tolerance 0.01 ... \
  --adetfile AlphaDets.bin --bdetfile BetaDets.bin --carryoverfile carryover.txt
```

- The rank product `task_comm_size × adet_comm_size × bdet_comm_size × h_comm_size` must equal the
  MPI world size (`-np N`). For 4 GPUs on one node: `bdet_comm_size=4`, the rest `1`
  (`h_comm_size` is derived as `world / (task·adet·bdet)`).
- GPU binding is automatic: `main.cc` does `cudaSetDevice(mpi_rank % cudaGetDeviceCount())`, so with
  `--gres=gpu:4` rank *i* lands on GPU *i*. Do **not** pass srun's `--gpu-bind`.
- `-DSBD_NON_CUDA_AWARE_MPI` (CPU-staged MPI) is compiled in so the binary is **multi-node-ready**,
  but only 4-GPU single-node has been validated to date; a genuine 2+ node run is untested.

The sweep launcher `sweep/run_fe4s4_mgpu.sh` wires all of this via env
(`FE4S4_LAUNCHER=mpirun`, `FE4S4_MPI_OPTIONS=-np,4`, `FE4S4_BDET_COMM_SIZE=4`,
`FE4S4_DIAG_BIN=diag-gpu-mpi`, `FE4S4_DIAG_BIN_UHF=diag-gpu_uhf-mpi`).
