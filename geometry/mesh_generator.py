"""网格生成 (Phase 2 占位)。

本次 (Phase 0-1) 仅实现并验证 first_layer_height —— 自适应 y+ 第一层网格高度
(方案 5.0), 因为它是纯函数、可独立测试, 且 Co-Kriging 之前不依赖完整网格。

Gmsh 边界层网格生成 (generate_mesh) 留待 Phase 2 实现, 届时按实际安装的
gmsh 4.x Python API 填充。
"""

from __future__ import annotations

__all__ = ["first_layer_height"]


def first_layer_height(Re: float, chord: float = 1.0,
                       yplus_target: float = 0.5) -> float:
    """按雷诺数计算壁面第一层网格高度, 目标 y+ = yplus_target (方案 5.0)。

    γ-Reθ 转捩模型要求 y+ < 1 (理想 0.1~0.5)。优化中翼型跨两个数量级的 Re,
    固定第一层高度无法通用, 必须写成 Re 的函数。

    推导 (Blasius 平板层流近似):
        Cf ≈ 0.664 / sqrt(Re)
        u_tau / U_inf = sqrt(Cf / 2)
        Δy = y+ * nu / u_tau = y+ * chord / (Re * sqrt(Cf/2))

    参考值: Re=5e4 -> ~5.3e-4 c; Re=2e5 -> ~1.8e-4 c; Re=5e5 -> ~9.0e-5 c。
    """
    Cf = 0.664 / Re ** 0.5
    u_tau_ratio = (Cf / 2.0) ** 0.5
    return yplus_target * chord / (Re * u_tau_ratio)


def generate_mesh(*args, **kwargs):  # noqa: D401
    """[Phase 2 占位] Gmsh 边界层网格生成。

    计划: 调用 first_layer_height(Re) 设置 hwall_n, 25 层边界层, ratio 1.20,
    输出 .su2 网格 (方案 5.0)。本次未实现。
    """
    raise NotImplementedError("Gmsh 网格生成在 Phase 2 实现")
