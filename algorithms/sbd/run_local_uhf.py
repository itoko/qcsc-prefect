"""
UHF 対応ローカル実行スクリプト（Prefect サーバ・HPC・IBM Quantum 不要）

目標:
  1. open-shell 系（UHF）で SQD クローズドループを最小構成で動かす
  2. 各イテレーション後に qiskit_addon_sqd.fermion.solve_fermion で
     正確なスピン分離 1-RDM / 2-RDM を取得する
  3. Orbital Rotation を実施してエネルギーを下げる
  4. α/β 独立回転（UHF 固有）と spin-free 回転（比較用）の両方を提供する

=== 積分テンソルの記法（確認済み） ===

  ep.one_body_tensor           : α 1 体積分 h[p,q]  (norb, norb)
  ep.two_body_tensor           : αα 2 体積分 (pq|rs) chemist's notation
  ep.two_body_tensor_ab        : αβ 2 体積分 (pq|rs) chemist's notation
  ep.two_body_tensor_bb        : ββ 2 体積分 (pq|rs) chemist's notation

  pyscf make_rdm1s 戻り値      : rdm1[p,q] = <q†p>   (norb, norb)
  pyscf make_rdm2s 戻り値 (pqrs格納) : rdm2[p,q,r,s] = <p†r†sq>
    → 正しい縮約: 0.5 * einsum('pqrs,pqrs->', h2_chem, rdm2_pqrs)
  内部で使う prqs格納 : rdm2[p,r,q,s] = <p†r†sq>
    → 正しい縮約: 0.5 * einsum('pqrs,prqs->', h2_chem, rdm2_prqs)
  変換: rdm2_prqs = rdm2_pqrs.transpose(0,2,1,3)

  solve_fermion は pqrs格納の rdm2 を返す（make_rdm2s と同一 convention）。
  solve_fermion の eri 引数は chemist's (pq|rs) を期待し、
  内部で spin-free Hamiltonian として使用する。

=== Orbital Rotation の方針 ===

  solve_fermion は spin-free（α/β 平均）Hamiltonian で対角化する。
  OrbRot の目的関数もこれと整合した spin-free Hamiltonian を使う。

  モード 1（デフォルト）: UHF α/β 独立回転
    - UHF エネルギー式 E(Uα, Uβ) を最小化（orbital_optimizer_uhf.py と共通実装）
    - 1-RDM 回転: γ_rot = U γ U^T  (rotate_one_body: h_new=U^T h U と整合)
    - 2-RDM 回転: g2[p,r,q,s] = Σ_{PRQS} U[p,P]U[r,R] rdm2[P,R,Q,S] U[q,Q]U[s,S]
                  (rotate_two_body と整合: 次イテレーションで E(H_new,rdm) == E_opt)
    - spin-free 近似ではないが、次イテレーションの Hamiltonian と RDM が一致

  モード 2（--spin_free）: spin-free 単一回転
    - ffsim.optimize_orbitals を使用（solve_fermion との完全整合）
    - spin-free 近似だが、対角化 Hamiltonian と RDM が一致
    - 収束が安定

ブランチ前提: local/uhf-work

実行方法:
  conda activate qcsc
  cd /Users/yutolt/Documents/qcsc-prefect

  # OH radical — UHF α/β 独立回転
  python algorithms/sbd/run_local_uhf.py --mol OH --iters 3 --sqd_dim 5000

  # spin-free 単一回転（ffsim.optimize_orbitals）
  python algorithms/sbd/run_local_uhf.py --mol OH --iters 3 --sqd_dim 5000 --spin_free

  # Orbital Rotation なし（ベースライン）
  python algorithms/sbd/run_local_uhf.py --mol OH --iters 3 --sqd_dim 5000 --no_orbopt

  # H3 radical
  python algorithms/sbd/run_local_uhf.py --mol H3 --iters 3 --sqd_dim 2000 --walkers 2

  # 結果を JSON に保存（orbital_optimizer_rhf.py と同じインターフェース）
  python algorithms/sbd/run_local_uhf.py --mol OH --iters 3 --optrot /tmp/optrot_uhf.json
"""

import argparse
import json
import logging
import os

import numpy as np
import scipy.optimize

# ─────────────────────────────────────────────────────────────────────────────
# ロギング
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_local_uhf")

# ─────────────────────────────────────────────────────────────────────────────
# Prefect get_run_logger パッチ
# ─────────────────────────────────────────────────────────────────────────────
import prefect.logging.loggers as _plg
_orig = _plg.get_run_logger
def _safe(*a, **kw):
    try:
        return _orig(*a, **kw)
    except Exception:
        return log
_plg.get_run_logger = _safe
import prefect.logging as _pl; _pl.get_run_logger = _safe
from qcsc_workflow_utility import chem as _chem; _chem.get_run_logger = _safe
from sbd import solver_job as _sj; _sj.get_run_logger = _safe

# ─────────────────────────────────────────────────────────────────────────────
# 必要モジュール
# ─────────────────────────────────────────────────────────────────────────────
import ffsim
from qcsc_workflow_utility.chem import compute_molecular_integrals_from_geometry
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform
from qiskit_addon_sqd.fermion import solve_fermion

from sbd.lucj import initialize_ucj_parameters
from sbd.sqd import postselect_bitstrings, subsample_open_shell


# ─────────────────────────────────────────────────────────────────────────────
# 分子定義
# ─────────────────────────────────────────────────────────────────────────────
MOLECULES = {
    "H3":  dict(atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48", basis="sto-3g", spin=1),
    "OH":  dict(atom="O 0 0 0; H 0 0 0.97",              basis="sto-3g", spin=1),
    "CH2": dict(atom="C 0 0 0; H 0 0.94 0.54; H 0 -0.94 0.54", basis="sto-3g", spin=2),
}


# ─────────────────────────────────────────────────────────────────────────────
# 記法ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def phys_to_chem(h2_phys: np.ndarray) -> np.ndarray:
    """physicist's <pq|rs> → chemist's (pq|rs): h_chem[p,q,r,s] = h_phys[p,r,q,s]"""
    return h2_phys.transpose(0, 2, 1, 3)


def chem_to_phys(h2_chem: np.ndarray) -> np.ndarray:
    """chemist's (pq|rs) → physicist's <pq|rs>: h_phys[p,q,r,s] = h_chem[p,r,q,s]"""
    return h2_chem.transpose(0, 2, 1, 3)


def spin_free_energy_from_rdm(
    rdm1_aa: np.ndarray,
    rdm1_bb: np.ndarray,
    rdm2_aa: np.ndarray,   # physicist's
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,   # chemist's
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    nuc: float,
) -> float:
    """spin-free (solve_fermion と同じ) エネルギーを計算する。

    solve_fermion が最小化するのはこのエネルギーなので、
    軌道回転がこの値を下げるかどうかで適用可否を判断する。
    """
    hcore_avg = (h1_a + h1_b) / 2.0
    eri_avg = (h2_aa + h2_ab + h2_ab.transpose(2, 3, 0, 1) + h2_bb) / 4.0
    rdm1_sum = rdm1_aa + rdm1_bb
    rdm2_sum = rdm2_aa + rdm2_bb + rdm2_ab + rdm2_ab.transpose(2, 3, 0, 1)
    return float(
        nuc
        + np.einsum("pq,pq->", hcore_avg, rdm1_sum)
        + 0.5 * np.einsum("pqrs,prqs->", eri_avg, rdm2_sum)
    )


def uhf_energy_from_rdm(
    rdm1_aa: np.ndarray,
    rdm1_bb: np.ndarray,
    rdm2_aa: np.ndarray,  # physicist's <p†r†sq>, pqrs-storage (solve_fermion convention)
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,   # chemist's (pq|rs)
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    nuc: float,
) -> float:
    """UHF エネルギーを RDM から計算。

    h2 は chemist's (pq|rs)、rdm2 は physicist's pqrs-storage なので
    縮約は einsum('pqrs,pqrs->', h2_chem, rdm2_pqrs) を使う。
    """
    return float(
        nuc
        + np.einsum("pq,pq->", h1_a, rdm1_aa)
        + np.einsum("pq,pq->", h1_b, rdm1_bb)
        + 0.5 * np.einsum("pqrs,pqrs->", h2_aa, rdm2_aa)
        + np.einsum("pqrs,pqrs->", h2_ab, rdm2_ab)
        + 0.5 * np.einsum("pqrs,pqrs->", h2_bb, rdm2_bb)
    )


# ─────────────────────────────────────────────────────────────────────────────
# UHF 固有の Orbital Rotation（α/β 独立）
# ─────────────────────────────────────────────────────────────────────────────

def _unitary_from_skew(x: np.ndarray, n: int) -> np.ndarray:
    """反対称行列 A の指数写像 U = expm(A) から実直交行列を作る。"""
    from scipy.linalg import expm
    A = np.zeros((n, n), dtype=np.float64)
    A[np.triu_indices(n, k=1)] = x
    A -= A.T
    return expm(A)


def optimize_orbitals_uhf(
    rdm1_aa: np.ndarray,
    rdm1_bb: np.ndarray,
    rdm2_aa: np.ndarray,   # physicist's <p†r†sq>, pqrs-storage (solver convention)
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,     # chemist's (pq|rs)
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    nuclear_repulsion_energy: float,
    options: dict | None = None,
    use_jax: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """α/β を独立に回転して UHF エネルギーを最小化する。

    入力 rdm2 は pqrs格納（solve_fermion / PySCF make_rdm2s の convention）:
        rdm2[p,q,r,s] = <p†r†sq>   E = 0.5*einsum('pqrs,pqrs->', h2, rdm2)

    内部では prqs格納 rdm2[p,r,q,s] = <p†r†sq> に変換して最適化する:
        E = 0.5*einsum('pqrs,prqs->', h2_chem, rdm2_prqs)

    1-RDM の回転 (physicist's dm1[p,q] = <q†p>):
        γ_rot[P,Q] = Σ_pq U[p,P] γ[p,q] U[q,Q]  = U γ U^T  (実数の場合)
        （rotate_one_body: h_new = U^T h U と整合するため）

    2-RDM の回転 (prqs格納: rdm2[p,r,q,s] = <p†r†sq>):
        g2[p,r,q,s] = Σ_{PRQS} U[p,P] U[r,R] rdm2[P,R,Q,S] U[q,Q] U[s,S]
        これは rotate_two_body と整合する。
    """
    norb = h1_a.shape[0]
    n_p = norb * (norb - 1) // 2  # パラメータ数（スピン 1 つあたり）

    # pqrs格納 (solver convention) → prqs格納 (内部 convention)
    rdm2_aa = np.asarray(rdm2_aa).transpose(0, 2, 1, 3)
    rdm2_ab = np.asarray(rdm2_ab).transpose(0, 2, 1, 3)
    rdm2_bb = np.asarray(rdm2_bb).transpose(0, 2, 1, 3)

    def _rot_rdm2(rdm2_prqs: np.ndarray, Ubra: np.ndarray, Uket: np.ndarray) -> np.ndarray:
        """physicist's prqs 格納の 2-RDM を正しく回転する。

        g2[p,r,q,s] = Σ_{PRQS} Ubra[p,P] Uket[r,R] rdm2[P,R,Q,S] Ubra[q,Q] Uket[s,S]

        2ステップ O(n^5) 分解:
          step1: Ubra で生成演算子 (axes 0,1) を縮約
          step2: Uket で消滅演算子 (axes 2,3) を縮約
        """
        tmp = np.einsum("pP,rR,PRQS->prQS", Ubra, Uket, rdm2_prqs, optimize="greedy")
        return np.einsum("prQS,qQ,sS->prqs", tmp, Ubra, Uket, optimize="greedy")

    def energy(x: np.ndarray) -> float:
        Ua = _unitary_from_skew(x[:n_p], norb)
        Ub = _unitary_from_skew(x[n_p:], norb)

        # 1-RDM の回転: γ_rot = Ua @ γ @ Ua^T
        # (rotate_one_body が h_new = U^T h U を適用するため、対応する RDM 変換は U γ U^T)
        g1a = Ua @ rdm1_aa @ Ua.T
        g1b = Ub @ rdm1_bb @ Ub.T

        # 2-RDM の回転: physicist's prqs → 生成/消滅を正しく分離
        g2aa = _rot_rdm2(rdm2_aa, Ua, Ua)
        g2ab = _rot_rdm2(rdm2_ab, Ua, Ub)
        g2bb = _rot_rdm2(rdm2_bb, Ub, Ub)

        # h2 は chemist's (pq|rs)、g2 は physicist's prqs → 縮約は 'pqrs,prqs->'
        return float(
            nuclear_repulsion_energy
            + np.einsum("pq,pq->", h1_a, g1a)
            + np.einsum("pq,pq->", h1_b, g1b)
            + 0.5 * np.einsum("pqrs,prqs->", h2_aa, g2aa)
            + np.einsum("pqrs,prqs->", h2_ab, g2ab)
            + 0.5 * np.einsum("pqrs,prqs->", h2_bb, g2bb)
        )

    # ── JAX analytical gradient (optional fast path) ──────────────────────
    # Import here to reuse the same JAX objective/grad builder from orbital_opt.
    from qcsc_workflow_utility.orbital_opt import _make_jax_uhf_obj_and_grad

    _build_jax = (use_jax is not False)  # True or None → attempt JAX
    if _build_jax:
        obj_jax, grad_jax = _make_jax_uhf_obj_and_grad(
            norb, n_p,
            rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
            h1_a, h1_b, h2_aa, h2_ab, h2_bb,
            nuclear_repulsion_energy,
        )
        if use_jax is True and obj_jax is None:
            raise ImportError(
                "use_jax=True was requested but JAX is not importable."
            )
    else:
        obj_jax, grad_jax = None, None

    _use_jax = obj_jax is not None
    if _use_jax:
        log.info("  [OrbOpt] Using JAX analytical gradient")
    else:
        log.info("  [OrbOpt] Using NumPy objective (no analytical gradient)")

    x0 = np.zeros(2 * n_p, dtype=np.float64)
    eval_obj = obj_jax if _use_jax else energy
    e_before = eval_obj(x0)

    minimize_kwargs: dict = dict(
        fun=eval_obj,
        x0=x0,
        method="L-BFGS-B",
        options=options or {"maxiter": 300, "ftol": 1e-12, "gtol": 1e-10},
    )
    if _use_jax:
        minimize_kwargs["jac"] = grad_jax

    result = scipy.optimize.minimize(**minimize_kwargs)
    Ua = _unitary_from_skew(result.x[:n_p], norb)
    Ub = _unitary_from_skew(result.x[n_p:], norb)
    e_after = float(result.fun)
    log.info(
        "  UHF OrbRot: E before=%.8f  E after=%.8f  ΔE=%.2e Ha  "
        "converged=%s  nit=%d",
        e_before, e_after, e_after - e_before, result.success, result.nit,
    )
    return Ua, Ub, e_before, e_after


def optimize_orbitals_spin_free(
    rdm1_sum: np.ndarray,
    rdm2_sum: np.ndarray,   # physicist's spin-summed, pqrs-storage (solver convention)
    h1_avg: np.ndarray,
    h2_avg: np.ndarray,     # chemist's spin-free
    nuclear_repulsion_energy: float,
) -> np.ndarray:
    """spin-free 単一回転（比較用）。ffsim.optimize_orbitals を使う。

    solve_fermion は spin-free Hamiltonian で対角化するため、
    この回転は対角化 H と完全に整合している。

    入力 rdm2_sum は pqrs格納（solver convention）: rdm2[p,q,r,s] = <p†r†sq>
    ffsim.ReducedDensityMatrix は prqs格納を期待するため transpose して渡す。
    ffsim のエネルギー式: 0.5 * einsum('abcd,abcd->', h2_phys, rdm2_prqs)
    h2 は physicist's に変換してから渡す。
    """
    # pqrs格納 → prqs格納 (ffsim convention)
    rdm2_sum_prqs = np.asarray(rdm2_sum).transpose(0, 2, 1, 3)
    # chemist's h2 → physicist's h2 (ffsim が期待する形)
    h2_avg_phys = chem_to_phys(h2_avg)
    rdm = ffsim.ReducedDensityMatrix(one_rdm=rdm1_sum, two_rdm=rdm2_sum_prqs)
    hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor=h1_avg,
        two_body_tensor=h2_avg_phys,
        constant=nuclear_repulsion_energy,
    )
    U = ffsim.optimize_orbitals(rdm=rdm, hamiltonian=hamiltonian, real=True)
    e_after = float(rdm.rotated(U).expectation(hamiltonian))
    e_before = float(rdm.expectation(hamiltonian))
    log.info(
        "  SpinFree OrbRot: E before=%.8f  E after=%.8f  ΔE=%.2e Ha",
        e_before, e_after, e_after - e_before,
    )
    return U


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian 回転ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def rotate_one_body(h1: np.ndarray, U: np.ndarray) -> np.ndarray:
    """h1_new = U^T h1 U  (dm1[p,q] = <q†p> の場合の整合的な変換)"""
    return U.T @ h1 @ U


def rotate_two_body(h2_chem: np.ndarray, U_bra: np.ndarray, U_ket: np.ndarray) -> np.ndarray:
    """chemist's (pq|rs) テンソルへの直交変換。

    (PQ|RS) = Σ_{pqrs} U_bra[p,P] U_bra[q,Q] (pq|rs) U_ket[r,R] U_ket[s,S]
    同スピン (aa,bb): U_bra == U_ket
    混合スピン (ab):  U_bra=Uα (bra side = α), U_ket=Uβ (ket side = β)
    """
    return np.einsum(
        "pP,qQ,pqrs,rR,sS->PQRS",
        U_bra, U_bra, h2_chem, U_ket, U_ket,
        optimize="greedy",
    )


# ─────────────────────────────────────────────────────────────────────────────
# solve_fermion から RDM を取得するヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def diagonalize_and_get_rdm(
    ci_a: np.ndarray,
    ci_b: np.ndarray,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,   # chemist's (pq|rs)
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    nuclear_repulsion_energy: float,
) -> tuple[
    float,
    np.ndarray, np.ndarray,           # occ_a, occ_b
    np.ndarray, np.ndarray,           # rdm1_aa, rdm1_bb  shape=(norb,norb)
    np.ndarray, np.ndarray, np.ndarray,  # rdm2_aa, rdm2_ab, rdm2_bb  physicist's
    float,                            # spin_sq
]:
    """qiskit_addon_sqd.fermion.solve_fermion で部分空間を対角化。

    solve_fermion の引数:
      hcore : spin-free 1 体積分 (norb, norb)  → (h1_a + h1_b) / 2
      eri   : chemist's (pq|rs) spin-free      → 4 スピンブロックの平均

    RDM:
      rdm1: sci_state.rdm(rank=1, spin_summed=False) → (rdm1_aa, rdm1_bb)
            make_rdm1s convention: dm1[p,q] = <q†p>
      rdm2: sci_state.rdm(rank=2, spin_summed=False) → (rdm2_aa, rdm2_ab, rdm2_bb)
            make_rdm2s convention: dm2[p,q,r,s] = <p†r†sq>  (physicist's)
    """
    na, nb = nelec

    # spin-free Hamiltonian（solve_fermion 内部の対角化に使用）
    hcore_avg = (h1_a + h1_b) / 2.0
    # h2 は chemist's のまま渡す（ba ブロック = ab^T）
    eri_avg = (h2_aa + h2_ab + h2_ab.transpose(2, 3, 0, 1) + h2_bb) / 4.0

    spin_sq_target = float((na - nb) / 2 * ((na - nb) / 2 + 1))

    energy_raw, sci_state, (occ_a, occ_b), spin_sq_val = solve_fermion(
        (ci_a, ci_b),
        hcore_avg,
        eri_avg,
        open_shell=True,
        spin_sq=spin_sq_target,
    )
    total_energy = float(energy_raw) + nuclear_repulsion_energy

    # スピン分離 RDM
    rdm1_aa, rdm1_bb = sci_state.rdm(rank=1, spin_summed=False)
    rdm2_aa, rdm2_ab, rdm2_bb = sci_state.rdm(rank=2, spin_summed=False)

    return (
        total_energy,
        np.asarray(occ_a), np.asarray(occ_b),
        rdm1_aa, rdm1_bb,
        rdm2_aa, rdm2_ab, rdm2_bb,
        float(spin_sq_val),
    )


# ─────────────────────────────────────────────────────────────────────────────
# メインループ
# ─────────────────────────────────────────────────────────────────────────────

def run(
    mol_key: str = "OH",
    n_iters: int = 3,
    sqd_dim: int = 5000,
    n_lucj_layers: int = 2,
    num_walkers: int = 4,
    apply_orbital_rotation: bool = True,
    spin_free: bool = False,
    random_seed: int = 42,
    optrot_path: str | None = None,
    report_path: str | None = None,
    compute_fci: bool = False,
    use_jax: bool | None = None,
):
    """
    Parameters
    ----------
    apply_orbital_rotation : bool
        True  → Orbital Rotation を実施
        False → ベースライン（回転なし）
    spin_free : bool
        True  → spin-free 単一回転（ffsim.optimize_orbitals）
                 solve_fermion の spin-free H と完全整合
        False → UHF α/β 独立回転（optimize_orbitals_uhf）
    optrot_path : str | None
        指定した場合、orbital rotation パラメータを JSON に書き出す
        （orbital_optimizer_rhf.py / orbital_optimizer_uhf.py と同じスキーマ）
    report_path : str | None
        指定した場合、検証レポート JSON を書き出す。
        SCI エネルギー履歴・OrbOpt 効果・FCI/UHF 参照を含む統一スキーマ。
        fugaku 等で実行した結果を比較・可視化するための基本出力。
    compute_fci : bool
        True にすると実行前に FCI エネルギーを計算してレポートに含める。
        小系のみ現実的（norb <= ~14）。
    """
    mol_cfg = MOLECULES[mol_key]
    # 各イテレーションの orbital rotation 履歴と最後の Ua/Ub を収集
    _orbrot_histories: list[list[float]] = []
    _last_Ua: np.ndarray | None = None
    _last_Ub: np.ndarray | None = None
    _norb_for_json: int | None = None
    # per-iter レポートデータ
    _report_iters: list[dict] = []
    log.info("=== UHF SQD local run ===")
    log.info("Molecule : %s  (%s)", mol_key, mol_cfg["atom"])
    log.info("Iters    : %d  sqd_dim=%d  walkers=%d", n_iters, sqd_dim, num_walkers)
    log.info("OrbRot   : %s  mode=%s", apply_orbital_rotation,
             "spin_free" if spin_free else "uhf_independent" if apply_orbital_rotation else "none")

    # ── 1. UHF 分子積分（HF + UCCSD）────────────────────────────────────────
    log.info("[1] UHF integral computation ...")
    ep = compute_molecular_integrals_from_geometry.fn(
        atom=mol_cfg["atom"],
        basis=mol_cfg["basis"],
        spin=mol_cfg["spin"],
        unrestricted=True,
    )
    assert ep.unrestricted, "Expected UHF ElectronicProperties"
    norb  = ep.num_orbitals
    _norb_for_json = norb
    nelec = ep.num_electrons
    na, nb = nelec
    log.info("  norb=%d  nelec=(%d,%d)  2Sz=%d", norb, na, nb, na - nb)

    # ループ中に更新される Hamiltonian（chemist's notation）
    h1_a  = ep.one_body_tensor.copy()       # (norb, norb)
    h1_b  = ep.one_body_tensor_b.copy()
    h2_aa = ep.two_body_tensor.copy()       # chemist's (pq|rs)
    h2_ab = ep.two_body_tensor_ab.copy()
    h2_bb = ep.two_body_tensor_bb.copy()
    nuc   = ep.nuclear_repulsion_energy
    e_uhf = float(ep.nuclear_repulsion_energy)  # placeholder; overwritten below
    e_fci: float | None = None

    # UHF エネルギーを参照値として取得（積分から再現）
    try:
        from pyscf import gto as _gto, scf as _scf
        _mol = _gto.Mole()
        _mol.build(atom=mol_cfg["atom"], basis=mol_cfg["basis"],
                   spin=mol_cfg["spin"], verbose=0)
        _mf = _scf.UHF(_mol).run()
        e_uhf = float(_mf.e_tot)
        log.info("  E_UHF = %.8f Ha", e_uhf)
        if compute_fci:
            from pyscf import fci as _fci
            _mo_a = _mf.mo_coeff[0]
            _cisolver = _fci.FCI(_mol, _mo_a)
            _e_fci, _ = _cisolver.kernel(nelec=_mol.nelec)
            e_fci = float(_e_fci)
            log.info("  E_FCI = %.8f Ha", e_fci)
    except Exception as _exc:
        log.warning("  Could not compute E_UHF/E_FCI reference: %s", _exc)

    # ── 2. UCJ パラメータ初期化 ──────────────────────────────────────────────
    aa_indices = [(p, p + 1) for p in range(norb - 1)]
    ab_indices = [(p, p) for p in range(0, norb, 4)]
    log.info("[2] UCJ parameter init (UHF, num_walkers=%d) ...", num_walkers)
    ucj_params = initialize_ucj_parameters.fn(
        elec_props=ep,
        aa_indices=aa_indices,
        ab_indices=ab_indices,
        num_walkers=num_walkers,
        randomization_factor=0.2,
        n_lucj_layers=n_lucj_layers,
    )

    # alpha / beta carryover kept as independent (n_a, norb) and (n_b, norb) bool arrays.
    # This avoids zero-padding artefacts that would inject hamming-weight-0 CI strings.
    carryover_a = np.zeros((0, norb), dtype=bool)
    carryover_b = np.zeros((0, norb), dtype=bool)
    best_energy = None
    rng = np.random.default_rng(random_seed)

    def _ci_strs_to_bool(ci_strs: np.ndarray, n: int) -> np.ndarray:
        """Convert int64 CI strings to a (len, n) bool occupancy matrix.

        Encoding: ci_str = Σ_j occ[j] * 2^(n-1-j)  (bit j=0 is the MSB).
        """
        powers = np.array([2 ** (n - 1 - j) for j in range(n)], dtype=np.int64)
        return ((ci_strs[:, None] & powers[None, :]) > 0)

    for iteration in range(n_iters):
        log.info("── Iteration %d/%d ─────────────────────────────", iteration + 1, n_iters)
        walker_results = []

        _best_ci_a = np.zeros(0, dtype=np.int64)
        _best_ci_b = np.zeros(0, dtype=np.int64)
        walker_ci: list[tuple[np.ndarray, np.ndarray]] = []

        for w_idx in range(num_walkers):
            log.info("  [Walker %d] sampling & SQD ...", w_idx)

            seed = int(random_seed + iteration * num_walkers + w_idx)
            bit_array = generate_bit_array_uniform(
                num_samples=sqd_dim, num_bits=norb * 2, rand_seed=seed,
            )
            raw_bs, raw_probs = bit_array_to_arrays(bit_array)
            bs, probs = recover_configurations(
                bitstring_matrix=raw_bs,
                probabilities=raw_probs,
                avg_occupancies=ep.initial_occupancy,
                num_elec_a=na,
                num_elec_b=nb,
                rand_seed=rng,
            )
            bs_post, probs_post = postselect_bitstrings.fn(
                bitstring_matrix=bs,
                probabilities=probs,
                hamming_right=na,
                hamming_left=nb,
            )

            ci_a, ci_b = subsample_open_shell.fn(
                bitstring_matrix=bs_post,
                probabilities=probs_post,
                carryover_a=carryover_a,
                carryover_b=carryover_b,
                subspace_dim=sqd_dim,
                norb=norb,
                num_elec_a=na,
                num_elec_b=nb,
                rng=rng,
            )
            log.info("    ci_a=%d  ci_b=%d  net_dim=%d",
                     len(ci_a), len(ci_b), len(ci_a) * len(ci_b))

            (energy, occ_a, occ_b,
             rdm1_aa, rdm1_bb,
             rdm2_aa, rdm2_ab, rdm2_bb,
             sq) = diagonalize_and_get_rdm(
                ci_a=ci_a, ci_b=ci_b,
                h1_a=h1_a, h1_b=h1_b,
                h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
                norb=norb, nelec=nelec,
                nuclear_repulsion_energy=nuc,
            )
            expected_sq = float((na - nb) / 2 * ((na - nb) / 2 + 1))
            # UHF エネルギー（参考: h2 chemist と rdm2 physicist の正しい縮約）
            e_uhf_rdm = uhf_energy_from_rdm(
                rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
                h1_a, h1_b, h2_aa, h2_ab, h2_bb, nuc,
            )
            log.info(
                "    SCI E=%.8f Ha  UHF_rdm E=%.8f Ha  <S^2>=%.4f  (expected %.4f)",
                energy, e_uhf_rdm, sq, expected_sq,
            )
            walker_results.append((energy, occ_a, occ_b,
                                   rdm1_aa, rdm1_bb,
                                   rdm2_aa, rdm2_ab, rdm2_bb,
                                   sq))
            walker_ci.append((ci_a, ci_b))

        best_w = int(np.argmin([r[0] for r in walker_results]))
        (best_e, best_occ_a, best_occ_b,
         best_rdm1_aa, best_rdm1_bb,
         best_rdm2_aa, best_rdm2_ab, best_rdm2_bb,
         best_sq) = walker_results[best_w]
        log.info("  Best walker %d: SCI E=%.8f Ha  <S^2>=%.4f", best_w, best_e, best_sq)

        if best_energy is None or best_e < best_energy:
            best_energy = best_e
            log.info("  ★ New best energy: %.8f Ha", best_energy)

        # ── carryover: carry best CI strings into the next iteration ─────────
        _best_ci_a, _best_ci_b = walker_ci[best_w]
        carryover_a = _ci_strs_to_bool(_best_ci_a, norb)
        carryover_b = _ci_strs_to_bool(_best_ci_b, norb)
        log.info("  Carryover updated: %d alpha / %d beta strings",
                 len(carryover_a), len(carryover_b))

        # ── Orbital Rotation ─────────────────────────────────────────────────
        _iter_orbrot_applied = False
        _iter_orbopt_e_before: float | None = None
        _iter_orbopt_e_after: float | None = None

        if apply_orbital_rotation:
            if spin_free:
                # ── spin-free 単一回転: solve_fermion の H と整合 ─────────────
                log.info("  [OrbRot] spin-free (ffsim.optimize_orbitals) ...")
                h1_avg = (h1_a + h1_b) / 2.0
                h2_avg = (h2_aa + h2_ab + h2_ab.transpose(2, 3, 0, 1) + h2_bb) / 4.0
                # rdm2_sum は pqrs格納 (solver convention)
                # transpose(2,3,0,1) は pqrs格納での αβ→βα 対称化
                rdm2_sum_pqrs = (best_rdm2_aa + best_rdm2_bb
                                 + best_rdm2_ab + best_rdm2_ab.transpose(2, 3, 0, 1))
                rdm1_sum = best_rdm1_aa + best_rdm1_bb
                U = optimize_orbitals_spin_free(
                    rdm1_sum=rdm1_sum, rdm2_sum=rdm2_sum_pqrs,
                    h1_avg=h1_avg, h2_avg=h2_avg,
                    nuclear_repulsion_energy=nuc,
                )
                # e_after_sf: ffsim は prqs格納を期待する
                rdm2_sum_prqs = rdm2_sum_pqrs.transpose(0, 2, 1, 3)
                _rdm_sf = ffsim.ReducedDensityMatrix(one_rdm=rdm1_sum, two_rdm=rdm2_sum_prqs)
                _ham_sf = ffsim.MolecularHamiltonian(
                    one_body_tensor=h1_avg,
                    two_body_tensor=chem_to_phys(h2_avg),
                    constant=nuc,
                )
                e_after_sf = float(_rdm_sf.rotated(U).expectation(_ham_sf))
                _iter_orbopt_e_before = float(_rdm_sf.expectation(_ham_sf))
                _iter_orbopt_e_after = e_after_sf
                _iter_orbrot_applied = True  # spin_free は常に適用
                _orbrot_histories.append([e_after_sf])
                _last_Ua = np.asarray(U)
                _last_Ub = np.asarray(U)
                log.info("  ||U^T U - I|| = %.2e", np.linalg.norm(U.T @ U - np.eye(norb)))
                h1_a  = rotate_one_body(h1_a, U)
                h1_b  = rotate_one_body(h1_b, U)
                h2_aa = rotate_two_body(h2_aa, U, U)
                h2_ab = rotate_two_body(h2_ab, U, U)
                h2_bb = rotate_two_body(h2_bb, U, U)

            else:
                # ── UHF α/β 独立回転 ─────────────────────────────────────────
                log.info("  [OrbRot] UHF independent alpha/beta rotation ...")
                Ua, Ub, _e_before_uhf, _e_after_uhf = optimize_orbitals_uhf(
                    rdm1_aa=best_rdm1_aa, rdm1_bb=best_rdm1_bb,
                    rdm2_aa=best_rdm2_aa, rdm2_ab=best_rdm2_ab, rdm2_bb=best_rdm2_bb,
                    h1_a=h1_a, h1_b=h1_b,
                    h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
                    nuclear_repulsion_energy=nuc,
                    use_jax=use_jax,
                )
                err_a = np.linalg.norm(Ua.T @ Ua - np.eye(norb))
                err_b = np.linalg.norm(Ub.T @ Ub - np.eye(norb))
                log.info("  ||Ua^T Ua - I||=%.2e  ||Ub^T Ub - I||=%.2e", err_a, err_b)
                _orbrot_histories.append([_e_after_uhf])
                _iter_orbopt_e_before = _e_before_uhf
                _iter_orbopt_e_after  = _e_after_uhf
                # spin-free エネルギーで適用可否を判断する
                # (solve_fermion が最小化するのは spin-free エネルギーのため)
                h1_a_new  = rotate_one_body(h1_a, Ua)
                h1_b_new  = rotate_one_body(h1_b, Ub)
                h2_aa_new = rotate_two_body(h2_aa, Ua, Ua)
                h2_ab_new = rotate_two_body(h2_ab, Ua, Ub)
                h2_bb_new = rotate_two_body(h2_bb, Ub, Ub)
                _e_sf_before = spin_free_energy_from_rdm(
                    best_rdm1_aa, best_rdm1_bb,
                    best_rdm2_aa, best_rdm2_ab, best_rdm2_bb,
                    h1_a, h1_b, h2_aa, h2_ab, h2_bb, nuc,
                )
                _e_sf_after = spin_free_energy_from_rdm(
                    best_rdm1_aa, best_rdm1_bb,
                    best_rdm2_aa, best_rdm2_ab, best_rdm2_bb,
                    h1_a_new, h1_b_new, h2_aa_new, h2_ab_new, h2_bb_new, nuc,
                )
                log.info(
                    "  SpinFree E before=%.8f  E after=%.8f  ΔE=%.2e Ha",
                    _e_sf_before, _e_sf_after, _e_sf_after - _e_sf_before,
                )
                if _e_sf_after < _e_sf_before:
                    _iter_orbrot_applied = True
                    _last_Ua = Ua
                    _last_Ub = Ub
                    h1_a, h1_b    = h1_a_new, h1_b_new
                    h2_aa, h2_ab, h2_bb = h2_aa_new, h2_ab_new, h2_bb_new
                else:
                    log.info(
                        "  OrbRot skipped: spin-free E after (%.8f) >= before (%.8f), "
                        "keeping current Hamiltonian.",
                        _e_sf_after, _e_sf_before,
                    )

            log.info("  Hamiltonian rotated.")

        # ── per-iter レポートデータを記録 ────────────────────────────────────
        _report_iters.append({
            "iteration": iteration + 1,
            "sci_energy": float(best_e),
            "spin_sq": float(best_sq),
            "orbrot_applied": _iter_orbrot_applied,
            "orbopt_e_before": _iter_orbopt_e_before,
            "orbopt_e_after":  _iter_orbopt_e_after,
            "orbopt_delta_e":  (
                float(_iter_orbopt_e_after - _iter_orbopt_e_before)
                if _iter_orbopt_e_before is not None and _iter_orbopt_e_after is not None
                else None
            ),
        })

    log.info("=== Finished ===")
    log.info("Final best SCI energy: %.8f Ha", best_energy)

    # ── JSON 出力（--optrot 指定時）────────────────────────────────────────────
    if optrot_path is not None:
        _norb = _norb_for_json or 0
        payload = {
            "mol": mol_key,
            "best_sci_energy": float(best_energy) if best_energy is not None else None,
            "orbrot_mode": "spin_free" if spin_free else ("uhf_indep" if apply_orbital_rotation else "none"),
            "norb": _norb,
            "iterations": len(_orbrot_histories),
            "histories": [[float(v) for v in h] for h in _orbrot_histories],
            "Ua": (_last_Ua.reshape(-1).tolist() if _last_Ua is not None
                   else np.eye(_norb).reshape(-1).tolist()),
            "Ub": (_last_Ub.reshape(-1).tolist() if _last_Ub is not None
                   else np.eye(_norb).reshape(-1).tolist()),
        }
        log.info("Writing orbital rotation result to %s", optrot_path)
        with open(optrot_path, "w", encoding="utf-8") as _f:
            json.dump(payload, _f, indent=2)
            _f.flush()
            os.fsync(_f.fileno())

    # ── レポート JSON 出力（--report 指定時）──────────────────────────────────
    if report_path is not None:
        _norb = _norb_for_json or 0
        _orbopt_mode = (
            "spin_free" if spin_free
            else ("uhf_indep" if apply_orbital_rotation else "none")
        )
        _history_sci    = [d["sci_energy"]      for d in _report_iters]
        _history_orbopt = [d["orbopt_e_after"]  for d in _report_iters]
        _orbrot_applied = [d["orbrot_applied"]  for d in _report_iters]
        report = {
            # ── 実験設定 ──────────────────────────────────────────────
            "mol": mol_key,
            "atom": mol_cfg["atom"],
            "basis": mol_cfg["basis"],
            "spin": mol_cfg["spin"],
            "norb": _norb,
            "nelec": list(nelec),
            "sqd_dim": sqd_dim,
            "num_walkers": num_walkers,
            "n_iters": n_iters,
            "orbopt_mode": _orbopt_mode,
            "seed": random_seed,
            # ── 参照エネルギー ────────────────────────────────────────
            "e_uhf": e_uhf,
            "e_fci": e_fci,            # None if compute_fci=False
            # ── 結果サマリー ──────────────────────────────────────────
            "best_sci_energy": float(best_energy) if best_energy is not None else None,
            "history_sci":    _history_sci,
            "history_orbopt": _history_orbopt,
            "orbrot_applied": _orbrot_applied,
            # ── per-iter 詳細 ─────────────────────────────────────────
            "iterations_detail": _report_iters,
        }
        log.info("Writing report to %s", report_path)
        with open(report_path, "w", encoding="utf-8") as _f:
            json.dump(report, _f, indent=2)
            _f.flush()
            os.fsync(_f.fileno())
        # サマリーをログにも出力
        log.info("─── Report Summary ─────────────────────────────────")
        if e_fci is not None:
            log.info("  FCI : %.8f Ha", e_fci)
        log.info("  UHF : %.8f Ha", e_uhf)
        log.info("  %-5s  %-14s  %-12s  %s",
                 "Iter", "SCI (Ha)", "OrbOpt ΔE (Ha)", "Applied")
        for d in _report_iters:
            delta = d["orbopt_delta_e"]
            delta_str = f"{delta:+.4e}" if delta is not None else "    —    "
            log.info("  %-5d  %.8f  %s  %s",
                     d["iteration"], d["sci_energy"], delta_str,
                     "YES" if d["orbrot_applied"] else "no ")
        log.info("  Best SCI: %.8f Ha", float(best_energy))
        if e_fci is not None:
            log.info("  FCI gap : %+.6f Ha", float(best_energy) - e_fci)
        log.info("────────────────────────────────────────────────────")

    return best_energy


# ─────────────────────────────────────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UHF SQD closed-loop with Orbital Rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OH radical, UHF alpha/beta rotation with JAX gradient (default)
  python run_local_uhf.py --mol OH --iters 5 --sqd_dim 8000 --fci

  # Force NumPy finite-difference gradient (disable JAX)
  python run_local_uhf.py --mol OH --iters 5 --sqd_dim 8000 --no_jax

  # spin-free rotation
  python run_local_uhf.py --mol OH --iters 5 --sqd_dim 8000 --spin_free

  # no orbital rotation (baseline)
  python run_local_uhf.py --mol OH --iters 5 --sqd_dim 8000 --no_orbopt
""",
    )
    parser.add_argument("--mol", default="OH", choices=list(MOLECULES))
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--sqd_dim", type=int, default=5000)
    parser.add_argument("--walkers", type=int, default=4)
    parser.add_argument("--no_orbopt", action="store_true",
                        help="Disable Orbital Rotation")
    parser.add_argument("--spin_free", action="store_true",
                        help="Use spin-free single rotation (ffsim) instead of UHF alpha/beta independent")
    parser.add_argument("--no_jax", action="store_true",
                        help="Force NumPy finite-difference gradient; "
                             "by default JAX analytical gradient is used when JAX is installed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optrot", default=None,
        help="Output JSON path for orbital rotation parameters "
             "(same schema as orbital_optimizer_rhf.py / orbital_optimizer_uhf.py)",
    )
    parser.add_argument(
        "--report", default=None, metavar="PATH",
        help="Output JSON path for full verification report "
             "(SCI history, OrbOpt effect, FCI/UHF reference). "
             "Compatible with work/verify_orbital_opt_uhf.py compare output.",
    )
    parser.add_argument(
        "--fci", action="store_true",
        help="Compute FCI reference energy and include it in the report "
             "(only practical for small systems, norb <= ~14).",
    )
    args = parser.parse_args()

    energy = run(
        mol_key=args.mol,
        n_iters=args.iters,
        sqd_dim=args.sqd_dim,
        num_walkers=args.walkers,
        apply_orbital_rotation=not args.no_orbopt,
        spin_free=args.spin_free,
        random_seed=args.seed,
        optrot_path=args.optrot,
        report_path=args.report,
        compute_fci=args.fci,
        use_jax=(False if args.no_jax else None),
    )
    print(f"\nFinal best SCI energy: {energy:.8f} Ha")


if __name__ == "__main__":
    main()
