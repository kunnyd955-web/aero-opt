"""NACA 4 位翼型解析坐标生成。

纯解析公式 (Jacobs et al. 1933), 无外部依赖。用作 Phase 0 中 CST 拟合的
"真值" 翼型 (方案第 8 节验证: 对 NACA 0012 / 2412 拟合, MSE < 1e-5)。
"""

from __future__ import annotations

import numpy as np

__all__ = ["naca4"]


def _thickness(x: np.ndarray, t: float, closed_te: bool = True) -> np.ndarray:
    """NACA 半厚度分布 yt(x)。closed_te=True 用 -0.1036 使后缘闭合。"""
    a4 = -0.1036 if closed_te else -0.1015
    return 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        + a4 * x ** 4
    )


def naca4(
    code: str,
    n_pts: int = 201,
    cosine_spacing: bool = True,
    closed_te: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成 NACA 4 位翼型上下表面坐标。

    参数
    ----
    code : 4 位字符串, 如 "0012", "2412"。
           digit1=最大弯度 m (%c), digit2=弯度位置 p (1/10 c), digit3-4=厚度 t (%c)。
    n_pts : 每个表面的点数。
    cosine_spacing : 余弦加密 (前后缘点密), 与 CST generate_airfoil 一致。

    返回
    ----
    (x_upper, y_upper, x_lower, y_lower)
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"NACA 4 位代码须为 4 位数字, 收到 {code!r}")

    m = int(code[0]) / 100.0          # 最大弯度
    p = int(code[1]) / 10.0           # 弯度位置
    t = int(code[2:]) / 100.0         # 厚度比

    if cosine_spacing:
        beta = np.linspace(0.0, np.pi, n_pts)
        x = 0.5 * (1.0 - np.cos(beta))
    else:
        x = np.linspace(0.0, 1.0, n_pts)

    yt = _thickness(x, t, closed_te=closed_te)

    if m == 0.0 or p == 0.0:
        # 对称翼型: 无弯度
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p ** 2 * (2 * p * x - x ** 2),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2),
        )
        dyc = np.where(
            x < p,
            2 * m / p ** 2 * (p - x),
            2 * m / (1 - p) ** 2 * (p - x),
        )

    theta = np.arctan(dyc)
    x_upper = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)

    return x_upper, y_upper, x_lower, y_lower
