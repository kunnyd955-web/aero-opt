"""Phase 3 训练驱动: 用 LF(XFOIL)+HF(SU2) 数据拟合 Co-Kriging 气动代理。

流程: 载入 doe/X_{lf,hf}.npy 与 {lf,hf}_solver/results/C{l,d}_{lf,hf}.npy
      -> 拟合 AeroSurrogate(Cl, Cd) -> 留一交叉验证 (LOO-CV) 报告精度
      -> 与"仅高保真 Kriging"基线对比 (量化多保真增益)
      -> 保存模型到 surrogate/models/。

用法:
    conda run -n aero-opt python scripts/train_surrogate.py
    conda run -n aero-opt python scripts/train_surrogate.py --no-cv   # 跳过 LOO-CV (快)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doe.doe_lhs import LF_BOUNDS  # noqa: E402
from surrogate.cokriging import AeroSurrogate, CoKriging  # noqa: E402


def _load() -> dict:
    d = {}
    d["X_lf"] = np.load(ROOT / "doe" / "X_lf.npy")
    d["X_hf"] = np.load(ROOT / "doe" / "X_hf.npy")
    d["Cl_lf"] = np.load(ROOT / "lf_solver" / "results" / "Cl_lf.npy")
    d["Cd_lf"] = np.load(ROOT / "lf_solver" / "results" / "Cd_lf.npy")
    d["Cl_hf"] = np.load(ROOT / "hf_solver" / "results" / "Cl_hf.npy")
    d["Cd_hf"] = np.load(ROOT / "hf_solver" / "results" / "Cd_hf.npy")
    return d


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    e = y_pred - y_true
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": len(y_true),
    }


def _subsample_lf(X_lf, y_lf, cap, seed=0):
    """CV 重拟合时给低保真点采样封顶: MFK 训练成本随 LF 点数 O(n^3),
    全量 808 点 × 数十次重拟合会非常慢。CV 仅估误差, 用 cap 个点足够,
    每次重拟合便宜约 (808/cap)^3 倍。全量模型 (fit) 仍用所有 LF 点。"""
    X = np.atleast_2d(X_lf)
    y = np.asarray(y_lf, dtype=float).ravel()
    m = np.isfinite(y)
    X, y = X[m], y[m]
    if cap is None or len(X) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=cap, replace=False)
    return X[idx], y[idx]


def loo_cv_cokriging(X_lf, y_lf, X_hf, y_hf, bounds, lf_cap=250) -> dict:
    """对高保真点做留一交叉验证 (每次留 1 个 HF 点作测试)。

    LF 点采样封顶到 lf_cap 以控成本 (见 _subsample_lf)。"""
    Xl, yl = _subsample_lf(X_lf, y_lf, lf_cap)
    Xh = np.atleast_2d(X_hf)
    yh = np.asarray(y_hf, dtype=float).ravel()
    keep = np.isfinite(yh)
    Xh, yh = Xh[keep], yh[keep]
    preds = np.full(len(yh), np.nan)
    for i in range(len(yh)):
        tr = np.ones(len(yh), dtype=bool)
        tr[i] = False
        m = CoKriging(bounds).fit(Xl, yl, Xh[tr], yh[tr])
        preds[i], _ = m.predict(Xh[i:i + 1])
    return _metrics(yh, preds)


def loo_cv_hf_only(X_hf, y_hf, bounds) -> dict:
    """基线: 仅用高保真点的单保真 Kriging 做 LOO-CV (对照多保真增益)。"""
    from smt.surrogate_models import KRG

    Xh = np.atleast_2d(X_hf)
    yh = np.asarray(y_hf, dtype=float).ravel()
    keep = np.isfinite(yh)
    Xh, yh = Xh[keep], yh[keep]
    lo, span = bounds[:, 0], np.maximum(bounds[:, 1] - bounds[:, 0], 1e-12)
    Xn = (Xh - lo) / span
    preds = np.full(len(yh), np.nan)
    for i in range(len(yh)):
        tr = np.ones(len(yh), dtype=bool)
        tr[i] = False
        km = KRG(theta0=[1e-1] * Xn.shape[1], print_global=False)
        km.set_training_values(Xn[tr], yh[tr])
        km.train()
        preds[i] = float(km.predict_values(Xn[i:i + 1]).ravel()[0])
    return _metrics(yh, preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cv", action="store_true", help="跳过 LOO-CV (快)")
    ap.add_argument("--cv-lf-cap", type=int, default=250,
                    help="LOO-CV 重拟合时低保真点采样上限 (控成本, 默认 250)")
    args = ap.parse_args()

    d = _load()
    bounds = LF_BOUNDS
    n_lf_ok = int(np.isfinite(d["Cl_lf"]).sum())
    n_hf_ok = int(np.isfinite(d["Cl_hf"]).sum())
    print(f"数据: LF {d['X_lf'].shape[0]} 点 (有效 {n_lf_ok}) | "
          f"HF {d['X_hf'].shape[0]} 点 (有效 {n_hf_ok})", flush=True)
    if n_hf_ok < 3:
        raise SystemExit(f"高保真有效点过少 ({n_hf_ok}), 先跑 sample_hf_batch.py --full")

    # --- 拟合全量模型并保存 ---
    t0 = time.time()
    surr = AeroSurrogate(bounds).fit(
        d["X_lf"], d["Cl_lf"], d["Cd_lf"],
        d["X_hf"], d["Cl_hf"], d["Cd_hf"])
    out = ROOT / "surrogate" / "models" / "aero_cokriging.pkl"
    surr.save(out)
    print(f"模型已保存: {out}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  Co-Kriging 训练点: Cl 用 LF={surr.cl.n_lf} HF={surr.cl.n_hf}; "
          f"Cd 用 LF={surr.cd.n_lf} HF={surr.cd.n_hf}")

    if args.no_cv:
        return

    # --- LOO-CV: 多保真 vs 仅高保真基线 ---
    print("\n留一交叉验证 (LOO-CV, 高保真点)：", flush=True)
    print(f"{'输出':<6}{'方法':<18}{'RMSE':>10}{'MAE':>10}{'R²':>8}")
    for name, y_lf, y_hf in [("Cl", d["Cl_lf"], d["Cl_hf"]),
                             ("Cd", d["Cd_lf"], d["Cd_hf"])]:
        ck = loo_cv_cokriging(d["X_lf"], y_lf, d["X_hf"], y_hf, bounds,
                              lf_cap=args.cv_lf_cap)
        hf = loo_cv_hf_only(d["X_hf"], y_hf, bounds)
        print(f"{name:<6}{'Co-Kriging':<18}{ck['rmse']:>10.4f}"
              f"{ck['mae']:>10.4f}{ck['r2']:>8.3f}")
        print(f"{'':<6}{'仅高保真 KRG':<18}{hf['rmse']:>10.4f}"
              f"{hf['mae']:>10.4f}{hf['r2']:>8.3f}")
        gain = (hf["rmse"] - ck["rmse"]) / hf["rmse"] * 100 if hf["rmse"] else 0
        print(f"{'':<6}-> 多保真 RMSE 较单保真 {'降低' if gain>=0 else '升高'} "
              f"{abs(gain):.1f}%")


if __name__ == "__main__":
    main()
