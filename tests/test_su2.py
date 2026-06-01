"""Phase 2 验证: SU2 配置填充与 history 解析 (方案 5.2)。

默认只测纯函数 (fill_config / parse_history), 不调 SU2_CFD, 故快且无外部依赖。
设环境变量 RUN_SU2_INTEGRATION=1 时额外跑一个极短的端到端 SU2 求解 (慢, 需 SU2)。

运行: conda run -n aero-opt python -m pytest tests/test_su2.py -v
端到端: RUN_SU2_INTEGRATION=1 conda run -n aero-opt python -m pytest tests/test_su2.py -v
"""

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hf_solver.run_su2 import (  # noqa: E402
    SU2_AVAILABLE,
    fill_config,
    parse_history,
)


def test_fill_config_no_placeholder_left():
    """所有 {{占位符}} 必须被替换。"""
    cfg = fill_config("mesh.su2", alpha=4.0, Re=2e5)
    assert "{{" not in cfg and "}}" not in cfg


def test_fill_config_values_present():
    """关键工况值与转捩模型关键字应出现在配置中。"""
    cfg = fill_config("airfoil_hf.su2", alpha=6.5, Re=1.5e5, mach=0.1)
    assert "SOLVER= RANS" in cfg
    assert "KIND_TURB_MODEL= SST" in cfg
    assert "KIND_TRANS_MODEL= LM" in cfg
    assert "airfoil_hf.su2" in cfg
    assert "6.5" in cfg                       # AOA
    assert "MARKER_HEATFLUX= ( airfoil" in cfg
    assert "MARKER_FAR= ( farfield )" in cfg
    # JST 中心格式与 MUSCL_FLOW= NO 的约束 (SU2 8.5 要求)
    assert "MUSCL_FLOW= NO" in cfg


def test_parse_history_basic(tmp_path):
    """解析含引号/空格表头的 SU2 history.csv, 取末行 CL/CD/CMz。"""
    hist = tmp_path / "history.csv"
    hist.write_text(
        '"Inner_Iter",    "rms[Rho]"   ,       "CD"       ,'
        '       "CL"       ,       "CMz"      \n'
        "          0,     -3.10,      0.030,     -0.10,      0.01\n"
        "        500,     -7.50,      0.0125,     0.4500,    -0.0200\n"
    )
    r = parse_history(hist)
    assert r["Cd"] == pytest.approx(0.0125)
    assert r["Cl"] == pytest.approx(0.45)
    assert r["Cm"] == pytest.approx(-0.02)
    assert r["rms_rho"] == pytest.approx(-7.5)
    assert r["inner_iter"] == 500


def test_parse_history_missing_file(tmp_path):
    """文件不存在时返回 nan, 不抛异常 (批量层依赖此行为)。"""
    r = parse_history(tmp_path / "nope.csv")
    assert math.isnan(r["Cl"]) and math.isnan(r["Cd"])
    assert r["inner_iter"] == -1


def test_su2_available_is_bool():
    assert isinstance(SU2_AVAILABLE, bool)


@pytest.mark.skipif(
    not (SU2_AVAILABLE and os.environ.get("RUN_SU2_INTEGRATION") == "1"),
    reason="需 SU2_CFD 且设 RUN_SU2_INTEGRATION=1 (慢)",
)
def test_su2_end_to_end_short(tmp_path):
    """极短端到端: NACA0012 求解少量迭代, 确认链路连通且能解析出力系数。

    迭代极少不要求收敛, 只验证 run_su2 返回结构 (residual/iter 已被读取)。
    """
    import numpy as np

    from geometry.cst_params import fit_cst, generate_airfoil
    from geometry.naca import naca4
    from hf_solver.run_su2 import run_su2

    xu, yu, xl, yl = naca4("0012", n_pts=120)
    Au, _ = fit_cst(xu, yu, n_coef=5)
    Al, _ = fit_cst(xl, yl, n_coef=5)
    coords = generate_airfoil(Au, Al)

    r = run_su2(coords, alpha=4.0, Re=2e5, max_iter=20,
                workdir=tmp_path / "case", timeout=600)
    # 20 步通常不收敛, 但残差与迭代数应被读到 (链路连通的证据)
    assert r["inner_iter"] >= 0
    assert not math.isnan(r["rms_rho"])
    assert r["returncode"] == 0
