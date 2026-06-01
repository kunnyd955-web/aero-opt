"""Phase 2 端到端高保真采样驱动 (方案 5.1 策略 B)。

流程: 高保真设计点 (sample_hf, 转捩活跃子区间) -> CST 生成翼型 (cst_params)
      -> gmsh 生成网格 (mesh_generator) -> SU2 转捩求解 (run_su2)
      -> 存 Cl/Cd 数组 (未收敛点写 NaN) -> 报告收敛率与耗时。

与 sample_lf_batch 对称, 但单点成本高两个数量级 (秒级 vs 分钟级), 故默认只跑
极小批量 (n=2) 做链路验证; 加 --full 跑满 50 个高保真样本 (方案 2 架构)。

用法:
    conda run -n aero-opt python scripts/sample_hf_batch.py            # 2 点
    conda run -n aero-opt python scripts/sample_hf_batch.py --n 5
    conda run -n aero-opt python scripts/sample_hf_batch.py --full     # 50 点
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doe.doe_lhs import sample_hf  # noqa: E402
from geometry.cst_params import check_geometry, generate_airfoil  # noqa: E402
from hf_solver.run_su2 import SU2_AVAILABLE, run_su2  # noqa: E402

MA = 0.1
TU = 0.002       # 入口湍流度 0.2% (补偿衰减后前缘约 0.1%, 对应 Ncrit=9, 方案 5.2)
TURB2LAM = 10.0  # 涡粘比初值


def run_batch(n: int, seed: int = 7, max_iter: int = 8000,
              timeout: float = 1800.0) -> dict:
    if not SU2_AVAILABLE:
        raise RuntimeError("SU2_CFD 不在 PATH, 请先 conda activate aero-opt")

    doe_dir = ROOT / "doe"
    X = sample_hf(n=n, seed=seed, save_path=doe_dir / "X_hf.npy")

    Cl = np.full(n, np.nan)
    Cd = np.full(n, np.nan)
    geo_invalid = 0
    residual_ok = 0

    t0 = time.time()
    for i in range(n):
        A_up, A_lo = X[i, :5], X[i, 5:10]
        alpha, Re = float(X[i, 10]), float(X[i, 11])

        # 几何先过滤 (快), 非法几何不浪费昂贵的网格+CFD 调用
        if not check_geometry(A_up, A_lo, t_min=0.10):
            geo_invalid += 1
            print(f"[{i+1}/{n}] 几何非法, 跳过", flush=True)
            continue

        coords = generate_airfoil(A_up, A_lo)
        ti = time.time()
        res = run_su2(coords, alpha=alpha, Re=Re, mach=MA, tu=TU,
                      turb2lam=TURB2LAM, max_iter=max_iter, timeout=timeout)
        dt_i = time.time() - ti
        if res["converged"]:
            Cl[i] = res["Cl"]
            Cd[i] = res["Cd"]
            residual_ok += int(res["residual_ok"])
            tag = "残差收敛" if res["residual_ok"] else "力收敛(残差未达标)"
            print(f"[{i+1}/{n}] α={alpha:.1f} Re={Re:.0e}  "
                  f"Cl={res['Cl']:.4f} Cd={res['Cd']:.5f} "
                  f"L/D={res['Cl']/res['Cd']:.1f}  {tag}  {dt_i:.0f}s", flush=True)
        else:
            print(f"[{i+1}/{n}] α={alpha:.1f} Re={Re:.0e}  失败: "
                  f"{res['reason'][:80]}  {dt_i:.0f}s", flush=True)
    dt = time.time() - t0

    out_dir = ROOT / "hf_solver" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "Cl_hf.npy", Cl)
    np.save(out_dir / "Cd_hf.npy", Cd)

    n_conv = int(np.sum(~np.isnan(Cl)))
    n_geo_ok = n - geo_invalid
    conv_rate = n_conv / n_geo_ok if n_geo_ok else 0.0
    return {
        "n": n, "geo_invalid": geo_invalid, "n_converged": n_conv,
        "residual_ok": residual_ok, "conv_rate": conv_rate, "seconds": dt,
        "Cl_path": out_dir / "Cl_hf.npy", "Cd_path": out_dir / "Cd_hf.npy",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="样本数 (默认 2)")
    ap.add_argument("--full", action="store_true", help="跑满 50 个高保真样本")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-iter", type=int, default=8000,
                    help="单点 SU2 最大迭代 (默认 8000)")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="单点墙钟超时秒数 (默认 1800)")
    args = ap.parse_args()
    n = 50 if args.full else args.n

    print(f"SU2 可用: {SU2_AVAILABLE} | 样本数: {n} | "
          f"Ma={MA} Tu={TU} max_iter={args.max_iter}", flush=True)
    r = run_batch(n, seed=args.seed, max_iter=args.max_iter,
                  timeout=args.timeout)
    print("-" * 60)
    print(f"几何非法点    : {r['geo_invalid']}/{r['n']}")
    print(f"SU2 收敛点    : {r['n_converged']} "
          f"(收敛率 {r['conv_rate']*100:.1f}%, 其中残差达标 {r['residual_ok']})")
    print(f"耗时          : {r['seconds']:.1f}s "
          f"({r['seconds']/max(r['n'],1):.0f} s/点)")
    print(f"已保存        : {r['Cl_path'].name}, {r['Cd_path'].name}")


if __name__ == "__main__":
    main()
