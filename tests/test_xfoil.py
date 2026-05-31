"""Phase 1 smoke test: XFOIL 流水线在 NACA 0012 上跑通。

这是流程验证 (后端可用、极曲线解析正确、量级合理), 不是精度认证。
NACA 0012 在 Re=2e5、alpha=4 deg 时 Cl 约 0.40-0.50。

运行: conda run -n aero-opt python -m pytest tests/test_xfoil.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry.cst_params import fit_cst, generate_airfoil  # noqa: E402
from geometry.naca import naca4  # noqa: E402
from lf_solver.run_xfoil import BACKEND, run_xfoil_polar  # noqa: E402

pytestmark = pytest.mark.skipif(
    BACKEND == "none", reason="无可用 XFOIL 后端 (xfoil / aerosandbox)"
)


def _naca0012_coords():
    """用 CST 拟合 NACA 0012 后生成干净的 Selig 坐标 (XFOIL 友好)。"""
    xu, yu, xl, yl = naca4("0012", n_pts=160)
    A_up, _ = fit_cst(xu, yu, n_coef=5, yte=0.0)
    A_lo, _ = fit_cst(xl, yl, n_coef=5, yte=0.0)
    return generate_airfoil(A_up, A_lo, yte_upper=0.0, yte_lower=0.0, n_pts=160)


def test_xfoil_polar_naca0012():
    coords = _naca0012_coords()
    polar = run_xfoil_polar(coords, -2.0, 10.0, 2.0,
                            Re=2e5, Ncrit=9.0, Ma=0.1)
    assert len(polar) > 0

    conv = [p for p in polar if p["converged"]]
    assert len(conv) >= len(polar) // 2, "收敛点过少, 后端或几何异常"

    # alpha≈4 deg 应收敛, Cl 量级合理
    p4 = min(conv, key=lambda p: abs(p["alpha"] - 4.0))
    assert abs(p4["alpha"] - 4.0) <= 2.0
    assert 0.2 < p4["Cl"] < 0.7, f"Cl={p4['Cl']} 超出 NACA0012 量级"
    assert p4["Cd"] > 0, f"Cd={p4['Cd']} 非正"
    assert p4["Cd"] < 0.05, f"Cd={p4['Cd']} 偏高 (层流翼型应 <0.02)"


def test_run_xfoil_single_point():
    from lf_solver.run_xfoil import run_xfoil
    coords = _naca0012_coords()
    r = run_xfoil(coords, alpha=4.0, Re=2e5, Ncrit=9.0, Ma=0.1)
    assert set(r) >= {"Cl", "Cd", "Cm", "converged"}
    if r["converged"]:
        assert 0.2 < r["Cl"] < 0.7
