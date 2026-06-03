"""Co-Kriging 多保真代理 (方案第 6 节)。

核心: smt 的 MFK (Multi-Fidelity Kriging)。自回归 1 阶模型
    y_hi(x) = rho * y_lo(x) + delta(x)
其中 y_lo 由低保真 (XFOIL) 大样本拟合的 Kriging 给出, delta 为高保真修正项,
由高保真 (SU2) 小样本拟合。rho 与超参由 MLE 联合估计。

设计变量量纲差异极大 (CST~0.1, alpha~0-12, Re~1e5), 必须先归一化到 [0,1],
否则各向同性核的相关长度无法同时适配。归一化用 LF 设计空间边界 (覆盖全域)。

非嵌套数据: LF/HF 采样点不要求重合 —— MFK 用 LF Kriging 在 HF 点处的预测值
参与构造修正项, 故只要 HF 点落在 LF 设计域内即可 (HF_BOUNDS ⊂ LF_BOUNDS, 满足)。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from smt.applications.mfk import MFK

__all__ = ["CoKriging", "AeroSurrogate"]


class CoKriging:
    """单输出 (标量) 的两级 Co-Kriging 代理。

    用法:
        m = CoKriging(bounds)            # bounds: (d,2) 输入归一化边界
        m.fit(X_lf, y_lf, X_hf, y_hf)    # y 含 NaN 的行 (未收敛点) 会被自动剔除
        mean, std = m.predict(X)         # std 为高保真预测标准差 (不确定度)
    """

    def __init__(self, bounds: np.ndarray, *, theta0: float = 1e-1):
        self.bounds = np.asarray(bounds, dtype=float)
        self.lo = self.bounds[:, 0]
        self.span = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12)
        self.theta0 = theta0
        self.sm: MFK | None = None
        self.n_lf = 0
        self.n_hf = 0

    # --- 输入归一化 (按设计空间边界, 不依赖训练数据统计, 预测时一致) ---
    def _norm(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return (X - self.lo) / self.span

    @staticmethod
    def _clean(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """剔除 y 为 NaN/Inf 的行 (未收敛采样点)。"""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        m = np.isfinite(y)
        return X[m], y[m]

    def fit(self, X_lf, y_lf, X_hf, y_hf) -> "CoKriging":
        Xl, yl = self._clean(X_lf, y_lf)
        Xh, yh = self._clean(X_hf, y_hf)
        if len(Xh) < 3:
            raise ValueError(f"高保真有效点过少 ({len(Xh)}), 无法拟合 Co-Kriging")
        if len(Xl) < len(Xh):
            raise ValueError(
                f"低保真有效点 ({len(Xl)}) 应远多于高保真 ({len(Xh)})")
        self.n_lf, self.n_hf = len(Xl), len(Xh)

        d = Xl.shape[1]
        self.sm = MFK(
            theta0=[self.theta0] * d,
            corr="squar_exp",
            print_global=False,
        )
        # name=0 为低保真; 缺省 (最高 level) 为高保真。
        self.sm.set_training_values(self._norm(Xl), yl, name=0)
        self.sm.set_training_values(self._norm(Xh), yh)
        self.sm.train()
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """返回 (mean, std)。std 为高保真预测的标准差 (sqrt 方差)。"""
        if self.sm is None:
            raise RuntimeError("模型未训练, 先调用 fit()")
        Xn = self._norm(X)
        mean = self.sm.predict_values(Xn).ravel()
        var = self.sm.predict_variances(Xn).ravel()
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "CoKriging":
        with Path(path).open("rb") as f:
            return pickle.load(f)


class AeroSurrogate:
    """气动系数代理: 同时持有 Cl、Cd 两个 Co-Kriging 模型, 派生 L/D。

    Phase 4 优化器只需调用 predict_ld(x) 取目标 (升阻比) 及其不确定度。
    """

    def __init__(self, bounds: np.ndarray, *, theta0: float = 1e-1):
        self.cl = CoKriging(bounds, theta0=theta0)
        self.cd = CoKriging(bounds, theta0=theta0)

    def fit(self, X_lf, Cl_lf, Cd_lf, X_hf, Cl_hf, Cd_hf) -> "AeroSurrogate":
        self.cl.fit(X_lf, Cl_lf, X_hf, Cl_hf)
        self.cd.fit(X_lf, Cd_lf, X_hf, Cd_hf)
        return self

    def predict(self, X) -> dict:
        cl_m, cl_s = self.cl.predict(X)
        cd_m, cd_s = self.cd.predict(X)
        return {"Cl": cl_m, "Cl_std": cl_s, "Cd": cd_m, "Cd_std": cd_s}

    def predict_ld(self, X) -> tuple[np.ndarray, np.ndarray]:
        """升阻比 L/D = Cl/Cd 及其一阶不确定度传播标准差。

        sigma_LD ≈ |L/D| * sqrt((s_cl/Cl)^2 + (s_cd/Cd)^2) (独立近似)。
        """
        r = self.predict(X)
        cl, cd = r["Cl"], np.maximum(r["Cd"], 1e-9)
        ld = cl / cd
        rel = np.sqrt((r["Cl_std"] / np.maximum(np.abs(cl), 1e-9)) ** 2
                      + (r["Cd_std"] / cd) ** 2)
        return ld, np.abs(ld) * rel

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "AeroSurrogate":
        with Path(path).open("rb") as f:
            return pickle.load(f)
