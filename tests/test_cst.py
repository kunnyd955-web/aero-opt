"""Phase 0 验证: CST 参数化对已知 NACA 翼型的重建精度。

验收标准 (方案第 8 节): 对 NACA 0012 / 2412 拟合, 每个翼面 MSE < 1e-5。

运行: conda run -n aero-opt python -m pytest tests/test_cst.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry.cst_params import (  # noqa: E402
    cst_surface,
    check_geometry,
    fit_cst,
    generate_airfoil,
)
from geometry.naca import naca4  # noqa: E402

MSE_THRESHOLD = 1e-5


@pytest.mark.parametrize("code", ["0012", "2412"])
def test_cst_fit_naca(code):
    """对 NACA 上下表面分别拟合, MSE 均应 < 1e-5。"""
    xu, yu, xl, yl = naca4(code, n_pts=201)

    # 尖后缘 NACA, 用 yte=0 拟合 (闭合后缘)
    A_up, mse_up = fit_cst(xu, yu, n_coef=5, yte=0.0)
    A_lo, mse_lo = fit_cst(xl, yl, n_coef=5, yte=0.0)

    assert mse_up < MSE_THRESHOLD, f"NACA {code} 上翼面 MSE={mse_up:.2e}"
    assert mse_lo < MSE_THRESHOLD, f"NACA {code} 下翼面 MSE={mse_lo:.2e}"


def test_generate_airfoil_shape():
    """generate_airfoil 输出形状与前缘唯一性。"""
    A_up = np.array([0.2, 0.17, 0.19, 0.11, 0.10])
    A_lo = -A_up
    coords = generate_airfoil(A_up, A_lo, n_pts=201)
    assert coords.shape == (2 * 201 - 1, 2)
    # x 介于 [0,1], 前缘点 x≈0 只出现一次
    assert coords[:, 0].min() >= -1e-9
    assert coords[:, 0].max() <= 1.0 + 1e-9
    assert np.sum(np.isclose(coords[:, 0], 0.0)) == 1


def test_check_geometry_valid():
    """对称合法翼型应通过几何检查 (厚度 ~12%)。"""
    A_up = np.array([0.18, 0.16, 0.18, 0.13, 0.12])
    A_lo = -A_up
    assert check_geometry(A_up, A_lo, t_min=0.10) is True


def test_check_geometry_selfintersect():
    """人为制造下翼面高于上翼面 (负厚度) 应被判非法。"""
    A_up = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    A_lo = np.array([0.20, 0.20, 0.20, 0.20, 0.20])  # 下翼面更高 -> 自交
    assert check_geometry(A_up, A_lo, t_min=0.10) is False


def test_check_geometry_too_thin():
    """厚度不足 10% 应被判非法。"""
    A_up = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
    A_lo = -A_up
    assert check_geometry(A_up, A_lo, t_min=0.10) is False


def test_cst_surface_endpoints():
    """类函数保证前缘 y(0)=0; 后缘 y(1)=yte。"""
    A = np.array([0.2, 0.15, 0.18, 0.1, 0.1])
    assert np.isclose(cst_surface(A, np.array([0.0]))[0], 0.0)
    assert np.isclose(cst_surface(A, np.array([1.0]), yte=0.003)[0], 0.003)
