"""Phase 2 端到端高保真采样驱动 (方案 5.1 策略 B) —— 进程级并行版。

流程: 高保真设计点 (sample_hf, 转捩活跃子区间) -> CST 生成翼型 (cst_params)
      -> gmsh 生成网格 (mesh_generator) -> SU2 转捩求解 (run_su2)
      -> 存 Cl/Cd 数组 (未收敛点写 NaN) -> 报告收敛率与耗时。

加速 (方案外, 实施优化):
  1. 进程级并行: 每个 SU2 case 单进程单核、各自临时工作目录, 用 ProcessPoolExecutor
     并发 --workers 个 (默认 6)。gmsh 全局状态非线程安全, 故必须用"多进程"而非
     多线程隔离。50 点 ~100min(串行) -> ~15min(6 并发)。
  2. 力系数 Cauchy 早停 (见 templates/su2_transition.cfg): 力稳定即停, 单 case 再快 ~2x。

与 sample_lf_batch 对称, 但单点成本高两个数量级 (秒级 vs 分钟级), 故默认只跑
极小批量 (n=2) 做链路验证; 加 --full 跑满 50 个高保真样本 (方案 2 架构)。

用法:
    conda run -n aero-opt python scripts/sample_hf_batch.py                  # 2 点
    conda run -n aero-opt python scripts/sample_hf_batch.py --n 5 --workers 4
    conda run -n aero-opt python scripts/sample_hf_batch.py --full           # 50 点, 6 并发
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _run_one(task: dict) -> dict:
    """单点 worker (顶层函数以便 ProcessPoolExecutor 跨进程序列化)。

    几何先过滤 (快), 非法几何不浪费昂贵的网格+CFD 调用。run_su2 自身从不抛异常,
    内部已设 SU2_RUN/SU2_HOME 并用独立临时工作目录, 故各 worker 完全隔离。
    """
    i = task["idx"]
    A_up, A_lo = task["A_up"], task["A_lo"]
    alpha, Re = task["alpha"], task["Re"]

    if not check_geometry(A_up, A_lo, t_min=0.10):
        return {"idx": i, "geo_invalid": True, "alpha": alpha, "Re": Re}

    coords = generate_airfoil(A_up, A_lo)
    t0 = time.time()
    res = run_su2(coords, alpha=alpha, Re=Re, mach=MA, tu=TU,
                  turb2lam=TURB2LAM, max_iter=task["max_iter"],
                  timeout=task["timeout"])
    res.update({"idx": i, "geo_invalid": False, "alpha": alpha, "Re": Re,
                "dt": time.time() - t0})
    return res


def _run_points(X: np.ndarray, max_iter: int, timeout: float,
                workers: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """对给定设计点矩阵 X (n,12) 并行跑 SU2, 返回 (Cl, Cd, geo_invalid, residual_ok)。

    未收敛/几何非法点对应 Cl/Cd 写 NaN, 行序与 X 严格对齐 (按 idx 回填)。"""
    n = len(X)
    Cl = np.full(n, np.nan)
    Cd = np.full(n, np.nan)
    geo_invalid = 0
    residual_ok = 0

    tasks = [
        {"idx": i, "A_up": X[i, :5], "A_lo": X[i, 5:10],
         "alpha": float(X[i, 10]), "Re": float(X[i, 11]),
         "max_iter": max_iter, "timeout": timeout}
        for i in range(n)
    ]

    workers = max(1, min(workers, n))
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_one, t): t["idx"] for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            i = res["idx"]
            done += 1
            alpha, Re = res["alpha"], res["Re"]
            if res.get("geo_invalid"):
                geo_invalid += 1
                print(f"[{done}/{n}] #{i} 几何非法, 跳过", flush=True)
                continue
            if res.get("converged"):
                Cl[i] = res["Cl"]
                Cd[i] = res["Cd"]
                residual_ok += int(res["residual_ok"])
                tag = "残差收敛" if res["residual_ok"] else "力收敛(残差未达标)"
                print(f"[{done}/{n}] #{i} α={alpha:.1f} Re={Re:.0e}  "
                      f"Cl={res['Cl']:.4f} Cd={res['Cd']:.5f} "
                      f"L/D={res['Cl']/res['Cd']:.1f}  {tag}  "
                      f"{res['dt']:.0f}s", flush=True)
            else:
                print(f"[{done}/{n}] #{i} α={alpha:.1f} Re={Re:.0e}  失败: "
                      f"{res['reason'][:80]}  {res['dt']:.0f}s", flush=True)
    return Cl, Cd, geo_invalid, residual_ok


def run_batch(n: int, seed: int = 7, max_iter: int = 8000,
              timeout: float = 1800.0, workers: int = 6,
              append: bool = False) -> dict:
    if not SU2_AVAILABLE:
        raise RuntimeError("SU2_CFD 不在 PATH, 请先 conda activate aero-opt")

    doe_dir = ROOT / "doe"
    out_dir = ROOT / "hf_solver" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = doe_dir / "X_hf.npy"
    cl_path, cd_path = out_dir / "Cl_hf.npy", out_dir / "Cd_hf.npy"

    if append:
        # 追加模式: 续采一批新点 (新 seed), 只对新点跑 CFD, 再拼到已有结果后面。
        X_old = np.load(x_path)
        Cl_old = np.load(cl_path)
        Cd_old = np.load(cd_path)
        X_new = sample_hf(n=n, seed=seed)  # 不覆盖 X_hf, 稍后拼接后再存
        # 防呆: 新批与旧批若有点几乎重合 (同 seed 误用), 说明白跑, 直接拒绝。
        dmin = np.min(np.linalg.norm(
            X_new[:, None, :] - X_old[None, :, :], axis=-1))
        if dmin < 1e-9:
            raise SystemExit(
                f"追加批与已有点重复 (min dist={dmin:.2e}); 请用与历史不同的 --seed")
        print(f"追加模式: 已有 {len(X_old)} 点 (有效 {int(np.isfinite(Cl_old).sum())}), "
              f"续采 {n} 新点 (seed={seed}, 距旧批最近 {dmin:.3f})", flush=True)
    else:
        X_new = sample_hf(n=n, seed=seed, save_path=x_path)
        X_old = Cl_old = Cd_old = None

    t0 = time.time()
    Cl_new, Cd_new, geo_invalid, residual_ok = _run_points(
        X_new, max_iter, timeout, workers)
    dt = time.time() - t0

    if append:
        X = np.vstack([X_old, X_new])
        Cl = np.concatenate([Cl_old, Cl_new])
        Cd = np.concatenate([Cd_old, Cd_new])
        np.save(x_path, X)
    else:
        Cl, Cd = Cl_new, Cd_new
    np.save(cl_path, Cl)
    np.save(cd_path, Cd)

    n_conv_new = int(np.sum(~np.isnan(Cl_new)))
    n_geo_ok = n - geo_invalid
    conv_rate = n_conv_new / n_geo_ok if n_geo_ok else 0.0
    return {
        "n": n, "geo_invalid": geo_invalid, "n_converged": n_conv_new,
        "residual_ok": residual_ok, "conv_rate": conv_rate, "seconds": dt,
        "workers": max(1, min(workers, n)),
        "total_pts": len(Cl), "total_valid": int(np.isfinite(Cl).sum()),
        "Cl_path": cl_path, "Cd_path": cd_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="样本数 (默认 2)")
    ap.add_argument("--full", action="store_true", help="跑满 50 个高保真样本")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-iter", type=int, default=8000,
                    help="单点 SU2 最大迭代 (默认 8000, 力系数早停通常更早退出)")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="单点墙钟超时秒数 (默认 1800)")
    ap.add_argument("--workers", type=int, default=6,
                    help="并发进程数 (默认 6; 机器 12 核, 留余量给 OS)")
    ap.add_argument("--append", action="store_true",
                    help="追加模式: 续采新点拼到已有 HF 数据后 (需配不同 --seed)")
    args = ap.parse_args()
    n = 50 if args.full else args.n

    print(f"SU2 可用: {SU2_AVAILABLE} | 样本数: {n} | 并发: {args.workers} | "
          f"追加: {args.append} (seed={args.seed}) | "
          f"Ma={MA} Tu={TU} max_iter={args.max_iter} | CPU 核: {os.cpu_count()}",
          flush=True)
    r = run_batch(n, seed=args.seed, max_iter=args.max_iter,
                  timeout=args.timeout, workers=args.workers, append=args.append)
    print("-" * 60)
    print(f"本批几何非法点: {r['geo_invalid']}/{r['n']}")
    print(f"本批 SU2 收敛 : {r['n_converged']} "
          f"(收敛率 {r['conv_rate']*100:.1f}%, 其中残差达标 {r['residual_ok']})")
    print(f"耗时          : {r['seconds']:.1f}s "
          f"({r['seconds']/max(r['n'],1):.0f} s/点, {r['workers']} 并发)")
    print(f"累计 HF 数据  : {r['total_pts']} 点 (有效 {r['total_valid']})")
    print(f"已保存        : {r['Cl_path'].name}, {r['Cd_path'].name}")


if __name__ == "__main__":
    main()
