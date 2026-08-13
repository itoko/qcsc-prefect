"""Compute reference energies for the Fe2S2 40q active space and write runs/refs.json.

Anchors, cheapest first:
  UHF, RHF        -- SCF baselines (from the FCIDUMP via to_scf)
  CCSD, CCSD(T)   -- coupled cluster on the UHF reference
  HCI             -- heat-bath selected CI (pyscf shci/selected-ci), if available
  DMRG            -- near-exact reference (block2, SymmetryTypes.SZ, MS2 from header)

Run this in the reference venv (refenv on Fugaku, which has pyscf + block2). It is DEFERRED until
the live mem2 job finishes (one-job-at-a-time), then submitted via run_fe2s2_refs.sh. The plot
script reads whatever keys are present in refs.json, so a partial run (e.g. no HCI) still plots.

Usage:
  FE2S2_FCIDUMP=<path> DMRG_M=100,200,400,800,1200 REFS_OUT=<runs/refs.json> \
      python build_fe2s2_refs.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump

# Molecule-agnostic: point REF_FCIDUMP at any FCIDUMP; norb/nelec/ms2 are read from its header.
# REFS_OUT is where the JSON of reference energies is written. DMRG_M is the bond-dimension ramp.
# (FE2S2_FCIDUMP kept as a fallback for older Fe2S2 launchers.)
FCIDUMP = os.environ.get("REF_FCIDUMP") or os.environ.get(
    "FE2S2_FCIDUMP", "/2ndfs/ra010014/u14924_space/sweep/fe2s2_40q.fcidump"
)
REFS_OUT = os.environ.get("REFS_OUT", "runs/refs.json")
DMRG_M = [int(x) for x in os.environ.get("DMRG_M", "100,200,400,800,1200").split(",") if x.strip()]

refs: dict[str, float] = {}
ctx = fcidump.read(FCIDUMP)
norb = int(ctx["NORB"])
nelec = int(ctx["NELEC"])
ms2 = int(ctx["MS2"])
ecore = float(ctx["ECORE"])
print(f"refs: norb={norb} nelec={nelec} ms2={ms2} ecore={ecore:.6f} src={FCIDUMP}", flush=True)


def _record(name: str, energy: float) -> None:
    refs[name] = float(energy)
    print(f"REF {name} = {energy:.8f}", flush=True)


# --- SCF + coupled cluster ---------------------------------------------------------------------
try:
    from pyscf import cc

    mf = fcidump.to_scf(FCIDUMP)
    mf = mf.to_uhf()
    mf.max_cycle = int(os.environ.get("SCF_MAX_CYCLE", "300"))
    mf.conv_tol = 1e-9
    mf.kernel()
    if not mf.converged:
        print("[warn] UHF SCF not converged; retrying with DIIS damping + level shift", flush=True)
        mf.diis_space = 12
        mf.level_shift = 0.2
        mf.kernel()
    _record("UHF", mf.e_tot)

    # Strongly-correlated Fe-S CCSD needs many iterations; default 50 does not converge. Bump
    # max_cycle and DIIS space (override via CCSD_MAX_CYCLE). Report convergence honestly.
    mycc = cc.UCCSD(mf)
    mycc.max_cycle = int(os.environ.get("CCSD_MAX_CYCLE", "300"))
    mycc.diis_space = int(os.environ.get("CCSD_DIIS_SPACE", "12"))
    mycc.conv_tol = 1e-7
    mycc.conv_tol_normt = 1e-5
    mycc.kernel()
    if not mycc.converged:
        print(f"[warn] UCCSD NOT converged after {mycc.max_cycle} cycles "
              f"(E_corr={mycc.e_corr:.8f}); recording anyway.", flush=True)
    # pyscf: cc.e_tot is a float attribute, NOT callable (mycc.e_tot() raises
    # "'numpy.float64' object is not callable").
    _record("UCCSD", float(mycc.e_tot))
    try:
        et = mycc.ccsd_t()
        _record("CCSD(T)", float(mycc.e_tot) + float(et))
    except Exception as exc:
        print(f"[skip] CCSD(T): {exc}", flush=True)
except Exception as exc:
    print(f"[skip] SCF/CC block: {exc}", flush=True)

# --- HCI (heat-bath selected CI) ---------------------------------------------------------------
# Optional. Selected-CI is intractable for large active spaces (e.g. 4Fe-4S 36o/54e HANGS for
# hours), so skip above HCI_MAX_NORB (default 24) unless HCI_MAX_NORB is raised explicitly.
try:
    hci_max = int(os.environ.get("HCI_MAX_NORB", "24"))
    if norb > hci_max:
        raise RuntimeError(f"norb={norb} > HCI_MAX_NORB={hci_max}; skipping HCI (intractable).")
    from pyscf import fci

    eri8 = ao2mo.restore(8, ctx["H2"], norb)
    h1 = ctx["H1"]
    na, nb = (nelec + ms2) // 2, (nelec - ms2) // 2
    sci = fci.SCI()
    sci.conv_tol = 1e-10
    e_hci, _ = sci.kernel(h1, eri8, norb, (na, nb), ecore=ecore)
    _record("HCI", float(np.atleast_1d(e_hci)[0]))
except Exception as exc:
    print(f"[skip] HCI: {exc}", flush=True)

# --- DMRG (near-exact) -------------------------------------------------------------------------
try:
    if not DMRG_M:
        raise RuntimeError("DMRG_M empty; skipping DMRG (run it as a separate job).")
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    h1 = ctx["H1"]
    eri1 = ao2mo.restore(1, ctx["H2"], norb)
    scratch = os.environ.get("DMRG_SCRATCH", "./dmrg_scratch_fe2s2")
    driver = DMRGDriver(
        scratch=scratch,
        symm_type=SymmetryTypes.SZ,
        n_threads=int(os.environ.get("DMRG_THREADS", "8")),
    )
    driver.initialize_system(n_sites=norb, n_elec=nelec, spin=ms2)
    mpo = driver.get_qc_mpo(h1e=h1, g2e=eri1, ecore=ecore, iprint=0)
    e_dmrg = None
    for m in DMRG_M:
        ket = driver.get_random_mps(tag=f"KET{m}", bond_dim=m, nroots=1)
        e_dmrg = driver.dmrg(
            mpo, ket, n_sweeps=20, bond_dims=[m],
            noises=[1e-4, 1e-5, 0.0], thrds=[1e-8] * 3, iprint=0,
        )
        print(f"DMRG_POINT M={m} E={e_dmrg:.8f}", flush=True)
    if e_dmrg is not None:
        _record("DMRG", e_dmrg)
except Exception as exc:
    print(f"[skip] DMRG: {exc}", flush=True)

os.makedirs(os.path.dirname(REFS_OUT) or ".", exist_ok=True)
with open(REFS_OUT, "w") as fh:
    json.dump(refs, fh, indent=2)
print(f"WROTE {REFS_OUT}: {refs}", flush=True)
