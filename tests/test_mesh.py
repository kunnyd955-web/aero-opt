"""Phase 2 验证: gmsh 网格生成与自适应 y+ 第一层高度 (方案 5.0)。

运行: conda run -n aero-opt python -m pytest tests/test_mesh.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry.cst_params import fit_cst, generate_airfoil  # noqa: E402
from geometry.mesh_generator import (  # noqa: E402
    MARKER_AIRFOIL,
    MARKER_FARFIELD,
    first_layer_height,
    generate_mesh,
)
from geometry.naca import naca4  # noqa: E402

gmsh = pytest.importorskip("gmsh")


@pytest.mark.parametrize("Re,expected", [
    (5e4, 2.6e-4), (2e5, 9.2e-5), (5e5, 4.6e-5),  # y+=0.5 默认值
])
def test_first_layer_height_order(Re, expected):
    """第一层高度量级应与 Blasius 推导一致 (容差 20%)。"""
    h = first_layer_height(Re)
    assert h == pytest.approx(expected, rel=0.2)


def test_first_layer_height_monotonic():
    """Re 越大, 边界层越薄, 第一层高度应单调下降。"""
    h = [first_layer_height(Re) for Re in (5e4, 1e5, 2e5, 5e5)]
    assert all(h[i] > h[i + 1] for i in range(len(h) - 1))


def _naca0012_coords():
    xu, yu, xl, yl = naca4("0012", n_pts=120)
    Au, _ = fit_cst(xu, yu, n_coef=5)
    Al, _ = fit_cst(xl, yl, n_coef=5)
    return generate_airfoil(Au, Al)


def _parse_su2(path: Path) -> dict:
    """极简 SU2 .su2 解析: 取 NDIME / NPOIN / NELEM 及各 MARKER_TAG。"""
    info = {"markers": []}
    with path.open() as f:
        lines = f.readlines()
    for ln in lines:
        s = ln.strip()
        if s.startswith("NDIME="):
            info["ndime"] = int(s.split("=")[1])
        elif s.startswith("NPOIN="):
            info["npoin"] = int(s.split("=")[1].split()[0])
        elif s.startswith("NELEM="):
            info["nelem"] = int(s.split("=")[1])
        elif s.startswith("NMARK="):
            info["nmark"] = int(s.split("=")[1])
        elif s.startswith("MARKER_TAG="):
            info["markers"].append(s.split("=")[1].strip())
    return info


def test_generate_mesh_su2_valid(tmp_path):
    """生成 NACA0012 网格, .su2 应为 2D、含 airfoil/farfield 两个标记、单元非空。"""
    coords = _naca0012_coords()
    out = tmp_path / "naca0012.su2"
    info = generate_mesh(coords, Re=2e5, out_path=out)

    assert out.exists()
    assert info["n_elements"] > 1000
    assert info["n_nodes"] > 500
    assert info["h_wall"] == pytest.approx(first_layer_height(2e5))

    parsed = _parse_su2(out)
    assert parsed["ndime"] == 2
    assert parsed["nelem"] == info["n_elements"]
    assert parsed["nmark"] == 2
    assert MARKER_AIRFOIL in parsed["markers"]
    assert MARKER_FARFIELD in parsed["markers"]


def test_generate_mesh_rejects_bad_coords(tmp_path):
    """坐标形状非法应抛 ValueError。"""
    with pytest.raises(ValueError):
        generate_mesh(np.zeros((3, 2)), Re=2e5, out_path=tmp_path / "x.su2")


def test_mesh_hwall_scales_with_Re(tmp_path):
    """高 Re 网格第一层更薄 (自适应 y+)。"""
    coords = _naca0012_coords()
    lo = generate_mesh(coords, Re=5e4, out_path=tmp_path / "lo.su2")
    hi = generate_mesh(coords, Re=5e5, out_path=tmp_path / "hi.su2")
    assert hi["h_wall"] < lo["h_wall"]
