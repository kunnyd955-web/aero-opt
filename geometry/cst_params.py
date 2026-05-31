"""CST (Class-Shape-Transformation) 翼型参数化。

依据项目方案 v2.0 第 3 节实现。提供:
  - bernstein / cst_surface : CST 基础函数
  - generate_airfoil        : CST 系数 -> Selig 格式翼型坐标
  - check_geometry          : 几何合法性约束检查 (厚度/自交/后缘)
  - fit_cst                 : 把目标坐标拟合为 CST 系数 (Phase 0 验证用)

约定:
  - 弦长归一化为 1.0, x in [0, 1]
  - 钝后缘默认 yte_upper=+0.002, yte_lower=-0.002 (总后缘厚 4 per-mille 弦长),
    避免 SU2 在尖后缘处的边界层网格扭曲 (方案风险表)。
"""

from __future__ import annotations

from math import comb

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "bernstein",
    "cst_surface",
    "generate_airfoil",
    "check_geometry",
    "fit_cst",
]


def bernstein(n: int, k: int, x: np.ndarray) -> np.ndarray:
    """Bernstein 基函数 B_{k,n}(x), 向量化实现。"""
    return comb(n, k) * (x ** k) * ((1.0 - x) ** (n - k))


def cst_surface(
    A: np.ndarray,
    x: np.ndarray,
    N1: float = 0.5,
    N2: float = 1.0,
    yte: float = 0.0,
) -> np.ndarray:
    """CST 翼面纵坐标 y(x)。

    y(x) = x^N1 (1-x)^N2 * sum_i A_i B_{i,N}(x) + x * yte

    参数
    ----
    A   : Bernstein 系数向量, 长度 N+1。
    x   : 弦向坐标, in [0, 1]。
    N1, N2 : 类函数指数。标准翼型 N1=0.5 (前缘圆钝), N2=1.0 (后缘收敛)。
    yte : 后缘偏移量 (半厚度)。y(x=1)=yte, 总后缘厚度 = yte_upper - yte_lower。
    """
    A = np.asarray(A, dtype=float)
    # x 定义域为 [0,1]; 裁剪掉因几何旋转 (弯度翼型前缘) 产生的微小越界,
    # 避免 x**0.5 / (1-x)**N2 出现 nan。越界点位于前后缘, y≈0, 物理无害。
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    N = len(A) - 1
    C = x ** N1 * (1.0 - x) ** N2
    S = np.zeros_like(x)
    for i in range(N + 1):
        S += A[i] * bernstein(N, i, x)
    return C * S + x * yte


def generate_airfoil(
    A_upper: np.ndarray,
    A_lower: np.ndarray,
    yte_upper: float = 0.002,
    yte_lower: float = -0.002,
    n_pts: int = 201,
) -> np.ndarray:
    """生成翼型坐标 (Selig 格式: 上翼面后缘 -> 前缘 -> 下翼面后缘)。

    返回
    ----
    coords : (2*n_pts-1, 2) ndarray, 列为 (x, y)。
             前缘点 (x=0) 只出现一次。
    """
    t = np.linspace(0.0, 1.0, n_pts)
    x = 0.5 * (1.0 - np.cos(np.pi * t))      # 余弦加密, 前后缘点密

    y_upper = cst_surface(A_upper, x, yte=yte_upper)
    y_lower = cst_surface(A_lower, x, yte=yte_lower)

    # Selig: 上翼面后缘->前缘, 再下翼面前缘->后缘 (跳过重复前缘点)
    coords = np.vstack([
        np.column_stack([x[::-1], y_upper[::-1]]),
        np.column_stack([x[1:], y_lower[1:]]),
    ])
    return coords


def check_geometry(
    A_upper: np.ndarray,
    A_lower: np.ndarray,
    yte_upper: float = 0.002,
    yte_lower: float = -0.002,
    t_min: float = 0.10,
) -> bool:
    """几何约束检查 (方案 3.1 / 7.1)。

    1. 无自交: 全弦向厚度 > 0
    2. 最大厚度比 >= t_min (结构下限)
    3. 后缘厚度 >= 0 (下翼面不得高于上翼面)
    """
    x = np.linspace(0.005, 0.995, 300)
    y_up = cst_surface(A_upper, x, yte=yte_upper)
    y_lo = cst_surface(A_lower, x, yte=yte_lower)
    thickness = y_up - y_lo
    te_thick = yte_upper - yte_lower
    return bool(
        np.all(thickness > 0.0)
        and np.max(thickness) >= t_min
        and te_thick >= 0.0
    )


def fit_cst(
    x_target: np.ndarray,
    y_target: np.ndarray,
    n_coef: int = 5,
    N1: float = 0.5,
    N2: float = 1.0,
    yte: float = 0.0,
) -> tuple[np.ndarray, float]:
    """把一条翼面 (x_target, y_target) 拟合为 CST 系数。

    用于 Phase 0 验证: 对已知翼型 (NACA) 拟合后比较重建精度 (方案 3.2)。

    返回
    ----
    (A, mse) : 拟合得到的系数向量与均方误差。
    """
    x_target = np.asarray(x_target, dtype=float)
    y_target = np.asarray(y_target, dtype=float)

    def cost(A: np.ndarray) -> float:
        y_fit = cst_surface(A, x_target, N1=N1, N2=N2, yte=yte)
        return float(np.sum((y_fit - y_target) ** 2))

    A0 = np.zeros(n_coef)
    res = minimize(cost, A0, method="L-BFGS-B")
    mse = res.fun / len(x_target)
    return res.x, mse
