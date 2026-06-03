"""Phase 3 验证: Co-Kriging 多保真代理 (方案第 6 节)。

用经典多保真 Forrester 函数构造合成数据 (1D), 不依赖昂贵的 SU2 采样, 故快且独立:
    y_hi(x) = (6x-2)^2 sin(12x-4)
    y_lo(x) = 0.5 y_hi(x) + 10(x-0.5) - 5
低保真与高保真强相关, Co-Kriging 应能用稀疏高保真点 + 密集低保真点逼近真值。

运行: conda run -n aero-opt python -m pytest tests/test_surrogate.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surrogate.cokriging import AeroSurrogate, CoKriging  # noqa: E402


def _y_hi(x):
    x = np.asarray(x, dtype=float).ravel()
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def _y_lo(x):
    return 0.5 * _y_hi(x) + 10 * (np.asarray(x, dtype=float).ravel() - 0.5) - 5


BOUNDS = np.array([[0.0, 1.0]])


def _data(n_lf=21, n_hf=6):
    x_lf = np.linspace(0, 1, n_lf).reshape(-1, 1)
    x_hf = np.linspace(0, 1, n_hf).reshape(-1, 1)
    return x_lf, _y_lo(x_lf), x_hf, _y_hi(x_hf)


def test_fit_predict_shapes():
    X_lf, y_lf, X_hf, y_hf = _data()
    m = CoKriging(BOUNDS).fit(X_lf, y_lf, X_hf, y_hf)
    xs = np.linspace(0, 1, 13).reshape(-1, 1)
    mean, std = m.predict(xs)
    assert mean.shape == (13,)
    assert std.shape == (13,)
    assert np.all(np.isfinite(mean))
    assert np.all(std >= 0)


def test_cokriging_beats_lf_at_hf_points():
    """高保真训练点处, Co-Kriging 预测应贴近真值 (远好于纯低保真)。"""
    X_lf, y_lf, X_hf, y_hf = _data(n_hf=8)
    m = CoKriging(BOUNDS).fit(X_lf, y_lf, X_hf, y_hf)
    mean, _ = m.predict(X_hf)
    # 训练点处近插值, 误差应很小
    assert np.max(np.abs(mean - y_hf.ravel())) < 0.5
    # 同点纯低保真值偏差大得多 (证明多保真修正起效)
    assert np.max(np.abs(_y_lo(X_hf) - y_hf.ravel())) > 1.0


def test_nan_rows_filtered():
    """y 含 NaN (未收敛点) 的行应被自动剔除, 不报错。"""
    X_lf, y_lf, X_hf, y_hf = _data()
    y_hf = y_hf.copy()
    y_hf[2] = np.nan
    y_lf = y_lf.copy()
    y_lf[5] = np.nan
    m = CoKriging(BOUNDS).fit(X_lf, y_lf, X_hf, y_hf)
    assert m.n_hf == len(y_hf) - 1
    assert m.n_lf == len(y_lf) - 1


def test_too_few_hf_raises():
    X_lf, y_lf, _, _ = _data()
    with pytest.raises(ValueError):
        CoKriging(BOUNDS).fit(X_lf, y_lf, np.array([[0.5]]), np.array([1.0]))


def test_save_load_roundtrip(tmp_path):
    X_lf, y_lf, X_hf, y_hf = _data()
    m = CoKriging(BOUNDS).fit(X_lf, y_lf, X_hf, y_hf)
    p = tmp_path / "m.pkl"
    m.save(p)
    m2 = CoKriging.load(p)
    xs = np.linspace(0, 1, 7).reshape(-1, 1)
    a, _ = m.predict(xs)
    b, _ = m2.predict(xs)
    assert np.allclose(a, b)


def test_aero_surrogate_ld():
    """AeroSurrogate 同时建 Cl/Cd 并派生 L/D 及不确定度。"""
    X_lf, _, X_hf, _ = _data(n_lf=21, n_hf=8)
    # 构造随 x 变化的正 Cl、Cd 合成场 (恒正避免除零; 非常数避免 Kriging 病态)
    cl_lf = 0.4 + 0.5 * X_lf.ravel()
    cl_hf = 0.45 + 0.5 * X_hf.ravel()
    cd_lf = 0.02 + 0.005 * X_lf.ravel()
    cd_hf = 0.025 + 0.005 * X_hf.ravel()
    s = AeroSurrogate(BOUNDS).fit(X_lf, cl_lf, cd_lf, X_hf, cl_hf, cd_hf)
    r = s.predict(np.array([[0.4]]))
    assert {"Cl", "Cl_std", "Cd", "Cd_std"} <= set(r)
    ld, ld_std = s.predict_ld(np.array([[0.4], [0.6]]))
    assert ld.shape == (2,)
    assert np.all(ld > 0) and np.all(ld_std >= 0)
