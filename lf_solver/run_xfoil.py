"""XFOIL 自动化封装 (Phase 1 低保真求解器)。

方案 4.2 关键要求: 不要用 subprocess 裸调 XFOIL (Fortran 死循环会挂起进程)。
本模块对上层隐藏后端差异, 按可用性自动选择:

  后端 A: xfoil (DARcorporation, 内存绑定共享库) —— 首选, 不起子进程。
  后端 B: aerosandbox —— 回退方案, 暴露转捩位置 xtr。

不收敛 / 异常一律返回 {'converged': False}, 绝不抛出, 以便批量采样跳过坏点。
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np

__all__ = ["BACKEND", "run_xfoil", "run_xfoil_polar"]


@contextlib.contextmanager
def _silence_stdout():
    """OS 级屏蔽 fd=1。XFOIL 是 Fortran, 直接写 fd 1, xf.print=False 压不住,
    批量跑上千点时会刷屏。pytest 捕获等场景下 fileno 不可用则降级为不屏蔽。"""
    try:
        target_fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    saved = os.dup(target_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, target_fd)
        yield
    finally:
        os.dup2(saved, target_fd)
        os.close(devnull)
        os.close(saved)


def _detect_backend() -> str:
    try:
        import xfoil  # noqa: F401
        return "xfoil"
    except Exception:
        try:
            import aerosandbox  # noqa: F401
            return "aerosandbox"
        except Exception:
            return "none"


BACKEND = _detect_backend()


# ---------------------------------------------------------------------------
# 后端 A: xfoil-python (DARcorporation)
# ---------------------------------------------------------------------------
def _polar_xfoil(coords, a0, a1, da, Re, Ncrit, Ma, max_iter):
    from xfoil import XFoil
    from xfoil.model import Airfoil as XFAirfoil

    with _silence_stdout():
        xf = XFoil()
        xf.airfoil = XFAirfoil(x=coords[:, 0], y=coords[:, 1])
        xf.Re = Re
        xf.M = Ma
        xf.n_crit = Ncrit
        xf.max_iter = max_iter
        xf.print = False

        a, cl, cd, cm, _cp = xf.aseq(a0, a1, da)
    out = []
    for i in range(len(a)):
        ok = not (np.isnan(cl[i]) or np.isnan(cd[i]) or cd[i] <= 0)
        out.append({
            "alpha": float(a[i]),
            "Cl": float(cl[i]) if ok else None,
            "Cd": float(cd[i]) if ok else None,
            "Cm": float(cm[i]) if ok else None,
            "converged": bool(ok),
        })
    return out


# ---------------------------------------------------------------------------
# 后端 B: aerosandbox (subprocess, 但内部有超时与异常管理)
# ---------------------------------------------------------------------------
def _polar_aerosandbox(coords, a0, a1, da, Re, Ncrit, Ma, max_iter):
    import aerosandbox as asb

    af = asb.Airfoil(coordinates=coords)
    xf = asb.XFoil(airfoil=af, Re=Re, mach=Ma, n_crit=Ncrit,
                   max_iter=max_iter, verbose=False)
    alphas = np.arange(a0, a1 + 0.5 * da, da)
    res = xf.alpha(alphas)  # dict: alpha/CL/CD/CM (仅含收敛点)

    converged_a = {round(float(av), 6): k for k, av in enumerate(res["alpha"])}
    out = []
    for a in alphas:
        key = round(float(a), 6)
        if key in converged_a:
            k = converged_a[key]
            cd = float(res["CD"][k])
            ok = cd > 0
            out.append({
                "alpha": float(a),
                "Cl": float(res["CL"][k]) if ok else None,
                "Cd": cd if ok else None,
                "Cm": float(res["CM"][k]) if ok else None,
                "converged": ok,
            })
        else:
            out.append({"alpha": float(a), "Cl": None, "Cd": None,
                        "Cm": None, "converged": False})
    return out


# ---------------------------------------------------------------------------
# 公共接口
# ---------------------------------------------------------------------------
def run_xfoil_polar(coords: np.ndarray,
                    alpha_start: float, alpha_end: float, alpha_step: float,
                    Re: float, Ncrit: float = 9.0, Ma: float = 0.1,
                    max_iter: int = 100) -> list[dict]:
    """扫描攻角范围, 返回极曲线 [{alpha, Cl, Cd, Cm, converged}, ...]。

    方案 4.2: 整段 aseq 延续上一步收敛解, 比逐点调用快约 2x 且失速附近更稳。
    任何后端异常都被吞掉, 整段返回 converged=False 的占位结果。
    """
    coords = np.asarray(coords, dtype=float)
    try:
        if BACKEND == "xfoil":
            return _polar_xfoil(coords, alpha_start, alpha_end, alpha_step,
                                Re, Ncrit, Ma, max_iter)
        if BACKEND == "aerosandbox":
            return _polar_aerosandbox(coords, alpha_start, alpha_end,
                                      alpha_step, Re, Ncrit, Ma, max_iter)
        raise RuntimeError("无可用 XFOIL 后端: 请安装 xfoil 或 aerosandbox")
    except RuntimeError:
        raise
    except Exception:
        alphas = np.arange(alpha_start, alpha_end + 0.5 * alpha_step, alpha_step)
        return [{"alpha": float(a), "Cl": None, "Cd": None, "Cm": None,
                 "converged": False} for a in alphas]


def _point_xfoil(coords, alpha, Re, Ncrit, Ma, max_iter):
    from xfoil import XFoil
    from xfoil.model import Airfoil as XFAirfoil

    with _silence_stdout():
        xf = XFoil()
        xf.airfoil = XFAirfoil(x=coords[:, 0], y=coords[:, 1])
        xf.Re = Re
        xf.M = Ma
        xf.n_crit = Ncrit
        xf.max_iter = max_iter
        xf.print = False
        cl, cd, cm, _cp = xf.a(alpha)   # 单点直解, 不走退化 aseq
    ok = not (np.isnan(cl) or np.isnan(cd) or cd <= 0)
    return {"Cl": float(cl) if ok else None,
            "Cd": float(cd) if ok else None,
            "Cm": float(cm) if ok else None,
            "converged": bool(ok)}


def _point_aerosandbox(coords, alpha, Re, Ncrit, Ma, max_iter):
    res = _polar_aerosandbox(coords, alpha, alpha, 1.0, Re, Ncrit, Ma, max_iter)
    r = res[0]
    return {"Cl": r["Cl"], "Cd": r["Cd"], "Cm": r["Cm"],
            "converged": r["converged"]}


def run_xfoil(coords: np.ndarray, alpha: float, Re: float,
              Ncrit: float = 9.0, Ma: float = 0.1,
              max_iter: int = 100) -> dict:
    """单工况 (alpha, Re) 计算, 返回 {Cl, Cd, Cm, converged}。

    单点用 xf.a(alpha) 直解 (退化的 aseq(a,a,da) 会返回空, 不可用)。
    异常 / 不收敛一律返回 converged=False。
    """
    coords = np.asarray(coords, dtype=float)
    try:
        if BACKEND == "xfoil":
            return _point_xfoil(coords, alpha, Re, Ncrit, Ma, max_iter)
        if BACKEND == "aerosandbox":
            return _point_aerosandbox(coords, alpha, Re, Ncrit, Ma, max_iter)
        raise RuntimeError("无可用 XFOIL 后端: 请安装 xfoil 或 aerosandbox")
    except RuntimeError:
        raise
    except Exception:
        return {"Cl": None, "Cd": None, "Cm": None, "converged": False}
