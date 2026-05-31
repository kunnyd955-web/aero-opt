"""Phase 1 端到端低保真采样驱动。

流程: 采样设计点 (doe_lhs) -> CST 生成翼型 (cst_params) -> XFOIL 求解 (run_xfoil)
      -> 存 Cl/Cd 数组 (未收敛点写 NaN) -> 报告收敛率。

本脚本默认跑小批量 (n=20) 用于流程验证; 加 --full 跑满 1000 个样本 (方案 4.1)。

用法:
    conda run -n aero-opt python scripts/sample_lf_batch.py            # 20 点
    conda run -n aero-opt python scripts/sample_lf_batch.py --n 100
    conda run -n aero-opt python scripts/sample_lf_batch.py --full     # 1000 点
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doe.doe_lhs import sample_lf  # noqa: E402
from geometry.cst_params import check_geometry, generate_airfoil  # noqa: E402
from lf_solver.run_xfoil import BACKEND, run_xfoil  # noqa: E402

MA = 0.1
NCRIT = 9.0  # 对应来流湍流度 Tu≈0.1% (方案 Mack 换算表)


def run_batch(n: int, seed: int = 42) -> dict:
    if BACKEND == "none":
        raise RuntimeError("无可用 XFOIL 后端, 请先安装 xfoil 或 aerosandbox")

    doe_dir = ROOT / "doe"
    X = sample_lf(n=n, seed=seed, save_path=doe_dir / "X_lf.npy")

    Cl = np.full(n, np.nan)
    Cd = np.full(n, np.nan)
    geo_invalid = 0

    t0 = time.time()
    for i in range(n):
        A_up, A_lo = X[i, :5], X[i, 5:10]
        alpha, Re = float(X[i, 10]), float(X[i, 11])

        # 几何先过滤 (快), 非法几何不浪费 XFOIL 调用
        if not check_geometry(A_up, A_lo, t_min=0.10):
            geo_invalid += 1
            continue

        coords = generate_airfoil(A_up, A_lo)
        res = run_xfoil(coords, alpha=alpha, Re=Re, Ncrit=NCRIT, Ma=MA)
        if res["converged"]:
            Cl[i] = res["Cl"]
            Cd[i] = res["Cd"]
    dt = time.time() - t0

    out_dir = ROOT / "lf_solver" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "Cl_lf.npy", Cl)
    np.save(out_dir / "Cd_lf.npy", Cd)

    n_conv = int(np.sum(~np.isnan(Cl)))
    n_geo_ok = n - geo_invalid
    conv_rate = n_conv / n_geo_ok if n_geo_ok else 0.0
    return {
        "n": n, "geo_invalid": geo_invalid, "n_converged": n_conv,
        "conv_rate": conv_rate, "seconds": dt,
        "Cl_path": out_dir / "Cl_lf.npy", "Cd_path": out_dir / "Cd_lf.npy",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="样本数 (默认 20)")
    ap.add_argument("--full", action="store_true", help="跑满 1000 个样本")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = 1000 if args.full else args.n

    print(f"XFOIL 后端: {BACKEND} | 样本数: {n} | Ncrit={NCRIT} Ma={MA}")
    r = run_batch(n, seed=args.seed)
    print(f"几何非法点    : {r['geo_invalid']}/{r['n']}")
    print(f"XFOIL 收敛点  : {r['n_converged']} (收敛率 {r['conv_rate']*100:.1f}%, 目标 >85%)")
    print(f"耗时          : {r['seconds']:.1f}s ({r['seconds']/r['n']*1000:.0f} ms/点)")
    print(f"已保存        : {r['Cl_path'].name}, {r['Cd_path'].name}")
    if r["conv_rate"] < 0.85:
        print("⚠️  收敛率 <85%: 考虑收窄攻角上限或对失败点改用全湍流 RANS (方案 4.3)")


if __name__ == "__main__":
    main()
