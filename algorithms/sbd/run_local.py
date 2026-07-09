"""
ローカル実行スクリプト（Prefect サーバ・HPC・IBM Quantum 不要）

目標:
  - qcsc-prefect の SBD クローズドループを最小構成で動かす
  - 対角化(SBD)後に RDM から最適 Orbital Rotation を求め、
    その回転を Hamiltonian に反映してから次のループへ進む

実行方法:
  conda activate qcsc
  cd /Users/yutolt/Documents/qcsc-prefect
  python algorithms/sbd/run_local.py

引数（すべてオプション）:
  --fcidump    FCIDump ファイルのパス（デフォルト: N2 サンプル）
  --iters      SQD-DE ループ回数（デフォルト: 2）
  --sqd_dim    サブサンプリング次元（デフォルト: 10000、小さい値はデバッグ用）
  --no_orbopt  Orbital Rotation を無効にする（差分を確認するため）
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

# ────────────────────────────────────────────────────────────────────────────
# ロギング設定（Prefect なしで動かすために Python 標準ロギングを使用）
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_local")

# ────────────────────────────────────────────────────────────────────────────
# qcsc-prefect のインポート
# ────────────────────────────────────────────────────────────────────────────
import ffsim
import ffsim.linalg
from pyscf import tools as pyscf_tools
import prefect.logging.loggers as _prefect_loggers

# qcsc_workflow_utility は内部で Prefect get_run_logger() を呼ぶ。
# Prefect コンテキスト外で実行するため、標準 logger にフォールバックするよう差し替える。
_orig_get_run_logger = _prefect_loggers.get_run_logger

def _safe_get_run_logger(*args, **kwargs):
    try:
        return _orig_get_run_logger(*args, **kwargs)
    except Exception:
        return log

_prefect_loggers.get_run_logger = _safe_get_run_logger

# prefect.logging.get_run_logger もパッチ
import prefect.logging as _pl
_pl.get_run_logger = _safe_get_run_logger

# qcsc_workflow_utility.chem が直接 import した参照もパッチ
from qcsc_workflow_utility import chem as _chem_mod
_chem_mod.get_run_logger = _safe_get_run_logger

from qcsc_workflow_utility.chem import (
    compute_molecular_integrals_from_fcidump,
)

from qiskit_addon_sqd.configuration_recovery import (
    post_select_by_hamming_weight,
    recover_configurations,
)
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform

# SBD サブモジュール（Prefect @task を .fn で呼ぶ）
from sbd.lucj import create_lucj_circuit, initialize_ucj_parameters
from sbd.sqd import (
    postselect_bitstrings,
    subsample_close_shell,
)


# ────────────────────────────────────────────────────────────────────────────
# Orbital Rotation ヘルパー
# ────────────────────────────────────────────────────────────────────────────

def compute_orbital_rotation(
    one_rdm: np.ndarray,
    two_rdm: np.ndarray,
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    nuclear_repulsion_energy: float,
) -> np.ndarray:
    """
    RDM から ffsim.optimize_orbitals でエネルギーを最小化する
    最適直交変換行列 U を求めて返す。

    数学的意味:
        E = Tr[h U^T γ_1 U] + 0.5 Tr[g (U⊗U)^T γ_2 (U⊗U)]
    を最小化する直交行列 U を L-BFGS-B で探索する。

    SBD の Davidson 対角化で得られた 1-RDM, 2-RDM を渡すと、
    次のループでより低いエネルギーが期待できる軌道基底を返す。
    """
    norb = one_body_tensor.shape[0]

    rdm = ffsim.ReducedDensityMatrix(
        one_rdm=one_rdm,
        two_rdm=two_rdm,
    )
    hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        constant=nuclear_repulsion_energy,
    )

    orbital_rotation = ffsim.optimize_orbitals(
        rdm=rdm,
        hamiltonian=hamiltonian,
        real=True,          # 実数直交行列に限定（量子回路と整合）
    )

    rotated_energy = rdm.rotated(orbital_rotation).expectation(hamiltonian)
    log.info("  Orbital rotation energy: %.10f Ha", rotated_energy)
    return orbital_rotation


def rotate_hamiltonian(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    orbital_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    軌道回転 U を Hamiltonian の積分テンソルに反映する。

    h_new[p,q]       = sum_{rs} U[r,p] h[r,s] U[s,q]   = U^T h U
    g_new[p,q,r,s]   = sum_{tuvw} U[t,p] U[u,q] g[t,u,v,w] U[v,r] U[w,s]
    """
    U = orbital_rotation
    h1_new = U.T @ one_body_tensor @ U
    # einsum: p←t, q←u, r←v, s←w
    h2_new = np.einsum("tp,uq,tuvw,vr,ws->pqrs", U, U, two_body_tensor, U, U)
    return h1_new, h2_new


# ────────────────────────────────────────────────────────────────────────────
# SBD なしのダミーソルバー（C++ バイナリが不要な純 Python 版）
# ────────────────────────────────────────────────────────────────────────────

def dummy_sbd_solve(
    ci_strings: np.ndarray,
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    C++ diag バイナリの代替（純 Python 簡易ソルバー）。

    FCI は norb > 16 で指数的に重くなるため、ci_strings の行列式を使った
    独立粒子近似で RDM と Hamiltonian 期待値を計算する。
    本番では C++ diag (SBD バイナリ) が担う部分。

    返り値:
        energy        : Hamiltonian 期待値 (核反発除く)
        occ_a, occ_b  : α/β 軌道占有数 (norb,)
        rdm1          : スピン和 1-RDM (norb, norb)
        rdm2          : スピン和 2-RDM (norb, norb, norb, norb)
    """
    # pyscf の 2 電子積分は chemist's notation (pq|rs)
    # qcsc_workflow_utility は physicist's notation <pq|rs> で保持
    # h2_chem[p,q,r,s] = h2_phys[p,r,q,s]
    h2_chem = two_body_tensor.transpose(0, 2, 1, 3)

    # ci_strings はビット整数の 1D 配列。ビット i が立っていれば軌道 i が占有
    n_dets = min(len(ci_strings), 100)
    rdm1_a = np.zeros((norb, norb))
    rdm1_b = np.zeros((norb, norb))
    for ci_int in ci_strings[:n_dets]:
        for i in range(norb):
            if (int(ci_int) >> i) & 1:
                rdm1_a[i, i] += 1.0 / n_dets
                rdm1_b[i, i] += 1.0 / n_dets

    rdm1 = rdm1_a + rdm1_b

    # 2-RDM の独立粒子近似（交換相互作用込み）
    rdm2 = (
        np.einsum("ij,kl->ijkl", rdm1, rdm1)
        - 0.5 * np.einsum("il,kj->ijkl", rdm1_a, rdm1_a)
        - 0.5 * np.einsum("il,kj->ijkl", rdm1_b, rdm1_b)
    )

    # Hamiltonian 期待値
    e = (
        np.einsum("ij,ji->", one_body_tensor, rdm1)
        + 0.5 * np.einsum("ijkl,klij->", h2_chem, rdm2)
    )

    occ_a = np.diag(rdm1_a)
    occ_b = np.diag(rdm1_b)

    return float(e), occ_a, occ_b, rdm1, rdm2


# ────────────────────────────────────────────────────────────────────────────
# メインループ
# ────────────────────────────────────────────────────────────────────────────

def run(
    fcidump_path: str,
    n_iters: int = 2,
    sqd_dim: int = 10_000,
    n_lucj_layers: int = 2,
    num_walkers: int = 4,
    apply_orbital_rotation: bool = True,
    random_seed: int = 42,
):
    log.info("=== qcsc-prefect local run ===")
    log.info("FCIDump  : %s", fcidump_path)
    log.info("Iters    : %d", n_iters)
    log.info("SQD dim  : %d", sqd_dim)
    log.info("OrbRot   : %s", apply_orbital_rotation)

    # ── 1. 分子積分の計算（HF + CCSD）─────────────────────────────────────
    log.info("[1] Computing molecular integrals from FCIDump ...")
    # Prefect @task デコレータを外して直接呼ぶ
    elec_props = compute_molecular_integrals_from_fcidump.fn(fcidump_path)
    norb = elec_props.num_orbitals
    nelec = elec_props.num_electrons
    log.info("  norb=%d  nelec=%s", norb, nelec)

    # 現在の Hamiltonian テンソル（ループ中に更新される）
    h1 = elec_props.one_body_tensor.copy()
    h2 = elec_props.two_body_tensor.copy()
    nuc_rep = elec_props.nuclear_repulsion_energy

    # ── 2. UCJ パラメータ初期化 ─────────────────────────────────────────
    aa_indices = [(p, p + 1) for p in range(norb - 1)]
    ab_indices = [(p, p) for p in range(0, norb, 4)]

    log.info("[2] Initializing UCJ parameters (num_walkers=%d) ...", num_walkers)
    ucj_params_list = initialize_ucj_parameters.fn(
        elec_props=elec_props,
        aa_indices=aa_indices,
        ab_indices=ab_indices,
        num_walkers=num_walkers,
        randomization_factor=0.2,
        n_lucj_layers=n_lucj_layers,
    )

    carryover = np.full((0, norb), np.nan, dtype=bool)
    best_energy = None
    rng = np.random.default_rng(random_seed)

    for iteration in range(n_iters):
        log.info("── Iteration %d/%d ──────────────────────────────", iteration + 1, n_iters)
        walker_energies = []

        for w_idx, ucj_param in enumerate(ucj_params_list):
            log.info("  [Walker %d] random sampling & SQD ...", w_idx)

            # ── ランダムビット列サンプリング（IBM Quantum の代替）──────────
            seed = int(random_seed + iteration * num_walkers + w_idx)
            bit_array = generate_bit_array_uniform(
                num_samples=sqd_dim,
                num_bits=norb * 2,
                rand_seed=seed,
            )

            # ── Configuration recovery & ポストセレクション ────────────────
            raw_bs, raw_probs = bit_array_to_arrays(bit_array)
            bs, probs = recover_configurations(
                bitstring_matrix=raw_bs,
                probabilities=raw_probs,
                avg_occupancies=elec_props.initial_occupancy,
                num_elec_a=nelec[0],
                num_elec_b=nelec[1],
                rand_seed=rng,
            )
            bs_post, probs_post = postselect_bitstrings.fn(
                bitstring_matrix=bs,
                probabilities=probs,
                hamming_right=nelec[0],
                hamming_left=nelec[1],
            )
            ci_strings = subsample_close_shell.fn(
                bitstring_matrix=bs_post,
                probabilities=probs_post,
                carryover=carryover,
                subspace_dim=sqd_dim,
                norb=norb,
                num_elec_a=nelec[0],
            )
            log.info("    subspace dim = %d", len(ci_strings) ** 2)

            # ── SBD 対角化（ダミー FCI ソルバーを使用）────────────────────
            energy_sbd, occ_a, occ_b, rdm1, rdm2 = dummy_sbd_solve(
                ci_strings=ci_strings,
                one_body_tensor=h1,
                two_body_tensor=h2,
                norb=norb,
                nelec=nelec,
            )
            total_energy = energy_sbd + nuc_rep
            log.info("    SBD energy = %.10f Ha  (total = %.10f Ha)", energy_sbd, total_energy)
            walker_energies.append((total_energy, occ_a, occ_b, rdm1, rdm2))

        # ── 最良ウォーカーの選択 ─────────────────────────────────────────
        best_w = int(np.argmin([e for e, *_ in walker_energies]))
        iter_best_energy, best_occ_a, best_occ_b, best_rdm1, best_rdm2 = walker_energies[best_w]
        log.info("  Best walker: %d  energy = %.10f Ha", best_w, iter_best_energy)

        if best_energy is None or iter_best_energy < best_energy:
            best_energy = iter_best_energy
            log.info("  ★ New best energy: %.10f Ha", best_energy)

        # ── Orbital Rotation ──────────────────────────────────────────────
        # 対角化で得た RDM と Hamiltonian からエネルギーを最小にする
        # 直交行列 U を求め、次ループの Hamiltonian テンソルを回転する。
        if apply_orbital_rotation:
            log.info("  [OrbRot] Optimizing orbital rotation ...")
            U = compute_orbital_rotation(
                one_rdm=best_rdm1,
                two_rdm=best_rdm2,
                one_body_tensor=h1,
                two_body_tensor=h2,
                nuclear_repulsion_energy=nuc_rep,
            )
            # U の直交性を確認
            ortho_err = np.linalg.norm(U.T @ U - np.eye(norb))
            log.info("  Orthogonality error ||U^T U - I|| = %.2e", ortho_err)

            # Hamiltonian テンソルを回転して次ループへ
            h1, h2 = rotate_hamiltonian(h1, h2, U)
            log.info("  Hamiltonian rotated for next iteration.")

    log.info("=== Finished ===")
    log.info("Final best energy: %.10f Ha", best_energy)
    return best_energy


# ────────────────────────────────────────────────────────────────────────────
# エントリポイント
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="qcsc-prefect local run with Orbital Rotation")
    parser.add_argument(
        "--fcidump",
        default="algorithms/sbd/data/fcidump_N2_MO.txt",
        help="FCIDump file path (relative to qcsc-prefect root)",
    )
    parser.add_argument("--iters", type=int, default=2, help="Number of SQD loop iterations")
    parser.add_argument("--sqd_dim", type=int, default=10_000, help="SQD subspace dimension")
    parser.add_argument(
        "--no_orbopt",
        action="store_true",
        help="Disable Orbital Rotation (for comparison)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    energy = run(
        fcidump_path=args.fcidump,
        n_iters=args.iters,
        sqd_dim=args.sqd_dim,
        apply_orbital_rotation=not args.no_orbopt,
        random_seed=args.seed,
    )
    print(f"\nResult: {energy:.10f} Ha")


if __name__ == "__main__":
    main()
