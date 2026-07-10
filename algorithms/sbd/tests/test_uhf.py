"""Regression tests for UHF (open-shell) support.

These lock the invariants that were validated end-to-end against a locally-built ``diag_uhf``
(``-D_UHF``) binary and PySCF FCI during development:

* the interleaved-spin-orbital FCIDUMP writer matches the ``_UHF SetupIntegrals`` layout,
* RHF behaviour is unchanged when ``method="rhf"`` / ``unrestricted=False``,
* the executable-key derivation stays in sync between create_blocks and SBDSolverJob.

The full numeric FCI comparison requires MPI + the compiled binary and is not run here; this file
covers the pure-Python structure that protects against regressions in the format and wiring.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _silence_prefect_logger(monkeypatch):
    """The integral/IO helpers call Prefect's get_run_logger; stub it for unit tests."""
    import qcsc_workflow_utility.chem as chem
    from sbd import solver_job

    monkeypatch.setattr(chem, "get_run_logger", lambda: mock.MagicMock())
    monkeypatch.setattr(solver_job, "get_run_logger", lambda: mock.MagicMock())


def test_rhf_electronic_properties_unchanged():
    import qcsc_workflow_utility.chem as chem

    ep = chem.compute_molecular_integrals_from_geometry.fn(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
    )
    assert ep.unrestricted is False
    assert ep.one_body_tensor_b is None
    assert ep.two_body_tensor_ab is None and ep.two_body_tensor_bb is None
    assert ep.t2_ab is None and ep.t2_bb is None


def test_uhf_electronic_properties_shapes():
    import qcsc_workflow_utility.chem as chem

    ep = chem.compute_molecular_integrals_from_geometry.fn(
        atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", unrestricted=True, spin=1
    )
    norb = ep.num_orbitals
    assert ep.unrestricted is True
    assert ep.num_electrons == (5, 4)
    assert ep.one_body_tensor.shape == (norb, norb)
    assert ep.one_body_tensor_b.shape == (norb, norb)
    for blk in (ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb):
        assert blk.shape == (norb, norb, norb, norb)
    # UCCSD t2 tuple: aa, ab, bb each (nocc, nocc, nvir, nvir) per spin.
    assert ep.t2.ndim == 4 and ep.t2_ab.ndim == 4 and ep.t2_bb.ndim == 4
    assert len(ep.initial_occupancy[0]) == norb
    assert len(ep.initial_occupancy[1]) == norb


def test_uhf_fcidump_writer_roundtrip_indices(tmp_path: Path):
    """The writer must emit 1-based spin-orbital records (alpha=2p+1, beta=2p+2) and a core row."""
    from sbd.solver_job import _write_uhf_fcidump

    norb = 2
    h1a = np.array([[-1.0, 0.1], [0.1, -0.5]])
    h1b = np.array([[-1.2, 0.0], [0.0, -0.4]])
    aa = np.zeros((norb,) * 4)
    aa[0, 0, 0, 0] = 0.7
    ab = np.zeros((norb,) * 4)
    ab[0, 0, 1, 1] = 0.3  # mixed (00|11): must survive the writer's symmetry handling
    bb = np.zeros((norb,) * 4)
    bb[1, 1, 1, 1] = 0.6

    path = tmp_path / "fcidump.txt"
    _write_uhf_fcidump(
        path, h1_a=h1a, h1_b=h1b, h2_aa=aa, h2_ab=ab, h2_bb=bb, norb=norb, nelec=(2, 1), ecore=1.5
    )
    text = path.read_text()
    assert "NORB=2" in text and "MS2=1" in text
    records = [
        r
        for r in text.splitlines()
        if r and not r.lstrip().startswith("&") and "ORBSYM" not in r and "ISYM" not in r
    ]
    # Core energy row is i=j=k=l=0.
    assert any(r.split()[1:] == ["0", "0", "0", "0"] for r in records)
    # Alpha one-body record uses odd 1-based index 1 (=2*0+1); beta uses 2 (=2*0+2).
    assert any(r.split()[1:] == ["1", "1", "0", "0"] for r in records)  # h1a[0,0]
    assert any(r.split()[1:] == ["2", "2", "0", "0"] for r in records)  # h1b[0,0]
    # Mixed (00 alpha | 11 beta): alpha pair (1,1), beta pair (4,4).
    assert any(r.split()[1:] == ["1", "1", "4", "4"] for r in records)


def test_uhf_fcidump_spin_block_convention(tmp_path: Path):
    """Lock the spin-block index pattern against the authoritative make_uhf-fcidump.py spec.

    Reference rule (request.md): a two-electron record (ij|kl) expands to spin-orbital records
    using alpha=2n-1, beta=2n (1-based spatial n). Same-spin pairs only; the three emitted blocks
    are aa|aa, aa|bb, bb|bb (bb|aa recovered by the reader via (ij|kl)=(kl|ij)). For a single
    RHF-derived integral set (all blocks equal) our writer's records must be a subset of the
    reference 4-block expansion with identical values.
    """
    from sbd.solver_job import _write_uhf_fcidump

    norb = 2
    # One off-diagonal two-electron integral; replicate to all blocks (RHF-derived UHF).
    g = np.zeros((norb,) * 4)
    g[1, 0, 0, 0] = 0.25  # chemist (10|00)
    h1 = np.array([[-1.0, 0.0], [0.0, -0.5]])
    path = tmp_path / "fcidump.txt"
    _write_uhf_fcidump(
        path, h1_a=h1, h1_b=h1, h2_aa=g, h2_ab=g, h2_bb=g, norb=norb, nelec=(1, 1)
    )
    records = {
        tuple(r.split()[1:]): float(r.split()[0])
        for r in path.read_text().splitlines()
        if r and not r.lstrip().startswith("&") and "ORBSYM" not in r and "ISYM" not in r
    }
    # Reference expansion of (10|00) [1-based spatial 2,1,1,1] with alpha=2n-1, beta=2n:
    #   aa|aa (3,1,1,1), aa|bb (3,1,2,2), bb|aa (4,2,1,1), bb|bb (4,2,2,2).
    # Our 3-block writer emits aa|aa, aa|bb, bb|bb (drops the bb|aa duplicate).
    assert records[("3", "1", "1", "1")] == 0.25  # aa|aa
    assert records[("3", "1", "2", "2")] == 0.25  # aa|bb
    assert records[("4", "2", "2", "2")] == 0.25  # bb|bb
    # bb|aa is intentionally omitted (reconstructed by the reader's (ij|kl)=(kl|ij) symmetry).
    assert ("4", "2", "1", "1") not in records


def test_executable_key_matrix():
    from sbd.solver_job import _executable_key

    class _S:
        def __init__(self, mode, method):
            self.solver_mode = mode
            self.method = method

    assert _executable_key(_S("cpu", "rhf")) == "sbd_diag"
    assert _executable_key(_S("fugaku", "rhf")) == "sbd_diag"
    assert _executable_key(_S("cpu", "uhf")) == "sbd_diag_uhf"
    assert _executable_key(_S("gpu", "rhf")) == "sbd_diag_gpu"
    assert _executable_key(_S("gpu", "uhf")) == "sbd_diag_gpu_uhf"


def test_subsample_open_shell_independent_spins():
    from sbd import sqd

    norb, na, nb = 4, 3, 2
    rng = np.random.default_rng(0)
    bm = rng.integers(0, 2, size=(200, 2 * norb)).astype(bool)
    probs = np.ones(200) / 200
    empty = np.empty((0, norb), dtype=bool)

    ci_a, ci_b = sqd.subsample_open_shell.fn(
        bitstring_matrix=bm,
        probabilities=probs,
        carryover_a=empty,
        carryover_b=empty,
        subspace_dim=64,
        norb=norb,
        num_elec_a=na,
        num_elec_b=nb,
    )
    assert int(ci_a[0]) == (1 << na) - 1  # alpha HF at index 0
    assert int(ci_b[0]) == (1 << nb) - 1  # beta HF at index 0
    assert len(set(ci_a.tolist())) == len(ci_a)
    assert len(set(ci_b.tolist())) == len(ci_b)
    assert not np.array_equal(ci_a, ci_b)


def _draw_open_shell(seed, *, rng=None):
    """Draw an alpha subspace large enough to exercise the rng.choice (random-pick) path.

    norb=8 gives C(8,4)=70 possible halves per spin; with a small subspace_dim the unique pool
    exceeds sqrt(subspace_dim), so the routine must randomly choose -- the branch the per-walker
    rng controls. Returns the alpha CI string list.
    """
    from sbd import sqd

    norb, na, nb = 8, 4, 4
    bm_rng = np.random.default_rng(seed)
    bm = bm_rng.integers(0, 2, size=(4000, 2 * norb)).astype(bool)
    probs = np.ones(4000) / 4000
    empty = np.empty((0, norb), dtype=bool)

    ci_a, _ = sqd.subsample_open_shell.fn(
        bitstring_matrix=bm,
        probabilities=probs,
        carryover_a=empty,
        carryover_b=empty,
        subspace_dim=400,  # sqrt = 20 << 70 unique halves -> rng.choice path is taken
        norb=norb,
        num_elec_a=na,
        num_elec_b=nb,
        rng=rng,
    )
    return ci_a


def test_subsample_open_shell_per_walker_rng_independent():
    """Different per-walker generators must draw DIFFERENT subspaces from the same samples.

    Regression guard for the module-global MODULE_RNG that made concurrent walkers correlated (and
    raced under the threaded task runner). The bitstring pool is fixed (same seed=7); only the
    subsample generator differs between the two walkers.
    """
    ci_walker0 = _draw_open_shell(7, rng=np.random.default_rng(1000))
    ci_walker1 = _draw_open_shell(7, rng=np.random.default_rng(2000))

    # Same HF anchor, but the randomly chosen non-HF strings must differ between walkers.
    assert ci_walker0[0] == ci_walker1[0]
    assert not np.array_equal(ci_walker0, ci_walker1)


def test_subsample_open_shell_rng_reproducible():
    """Same per-walker seed -> identical draw (deterministic, resumable)."""
    a = _draw_open_shell(7, rng=np.random.default_rng(42))
    b = _draw_open_shell(7, rng=np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_subsample_open_shell_rng_default_falls_back():
    """rng=None must still work (legacy callers) via the module fallback generator."""
    ci_a = _draw_open_shell(7, rng=None)
    assert int(ci_a[0]) == (1 << 4) - 1  # HF anchored at index 0
    assert len(set(ci_a.tolist())) == len(ci_a)


def test_uhf_breaks_spin_symmetry_for_correlated_singlet():
    """Regression: the UHF path must reach the genuine broken-symmetry solution for a
    strongly-correlated closed-shell SINGLET, not stay at the RHF stationary point.

    Bug (fixed): mf.to_uhf().kernel() / scf.UHF(mol).run() started from the symmetric RHF density
    and, for a singlet, converged back to RHF (<S^2>=0, identical energy) -> the "UHF" reference
    and its UCCSD amplitude seed were restricted in disguise. _converge_broken_symmetry_uhf follows
    the spin instability to the true UHF minimum. Stretched H4 is the canonical test: RHF and the
    broken-symmetry UHF differ by hundreds of mHa.
    """
    import numpy as np
    from pyscf import gto, scf
    import qcsc_workflow_utility.chem as chem

    mol = gto.M(atom="H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0", basis="sto-3g", spin=0, verbose=0)
    e_rhf = scf.RHF(mol).run(verbose=0).e_tot

    mf = chem._converge_broken_symmetry_uhf(scf.UHF(mol))
    ss, _ = mf.spin_square()
    assert ss > 0.5, f"UHF stayed restricted (<S^2>={ss:.3f}); spin symmetry not broken"
    assert mf.e_tot < e_rhf - 1e-2, (
        f"broken-symmetry UHF E={mf.e_tot:.6f} not below RHF E={e_rhf:.6f}"
    )
