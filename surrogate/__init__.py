"""Co-Kriging 多保真代理模型 (Phase 3)。

用廉价的 XFOIL 低保真大样本 + 昂贵的 SU2 高保真小样本, 经 Co-Kriging
(自回归 1 阶: y_hi(x) = rho * y_lo(x) + delta(x)) 桥接, 给出对真实 (高保真)
气动系数的廉价预测与不确定度, 供 Phase 4 全局优化调用。
"""

from surrogate.cokriging import CoKriging, AeroSurrogate

__all__ = ["CoKriging", "AeroSurrogate"]
