"""SU2 高保真求解封装 (Phase 2 高保真求解器, 方案 5.2)。

链路: CST 翼型坐标 -> gmsh 网格 (.su2) -> 填充 SU2 8.5 配置 -> 子进程跑 SU2_CFD
      -> 解析 history.csv 取 Cl/Cd/Cm。

设计与 lf_solver.run_xfoil 对齐: 单工况入口 run_su2(coords, alpha, Re, ...),
返回 {Cl, Cd, Cm, converged, ...}; 任何失败 (网格/求解器异常/超时/不收敛) 一律
返回 converged=False 且不抛出, 以便批量采样跳过坏点。

依赖外部命令 SU2_CFD (本环境为 SU2 v8.5.0)。须在已 source SU2 环境的 shell 中
运行 (conda activate aero-opt; 见 README 的 SU2_RUN/SU2_HOME 说明)。
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from geometry.mesh_generator import generate_mesh

__all__ = ["SU2_AVAILABLE", "run_su2", "fill_config", "parse_history"]

_HERE = Path(__file__).resolve().parent
TEMPLATE = _HERE / "templates" / "su2_transition.cfg"


def _detect_su2() -> str | None:
    return shutil.which("SU2_CFD")


SU2_BIN = _detect_su2()
SU2_AVAILABLE = SU2_BIN is not None


def fill_config(mesh_file: str, alpha: float, Re: float, *,
                mach: float = 0.1, tu: float = 0.002, turb2lam: float = 10.0,
                max_iter: int = 8000, conv_name: str = "history",
                restart_name: str = "restart_flow.dat") -> str:
    """把模板里的 {{占位符}} 用工况值替换, 返回完整配置文本。"""
    text = TEMPLATE.read_text()
    repl = {
        "{{MACH}}": repr(float(mach)),
        "{{AOA}}": repr(float(alpha)),
        "{{RE}}": repr(float(Re)),
        "{{TU}}": repr(float(tu)),
        "{{TURB2LAM}}": repr(float(turb2lam)),
        "{{ITER}}": str(int(max_iter)),
        "{{MESH}}": mesh_file,
        "{{CONV}}": conv_name,
        "{{RESTART}}": restart_name,
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    if "{{" in text:
        leftover = text[text.index("{{"):text.index("{{") + 20]
        raise RuntimeError(f"配置模板存在未填充占位符: {leftover!r}")
    return text


def _to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def parse_history(history_path: str | Path) -> dict:
    """读 SU2 history.csv, 返回末行的 {Cl, Cd, Cm, rms_rho, inner_iter}。

    SU2 表头形如 `"Inner_Iter","rms[Rho]",...,"CL","CD","CMz"`, 列名带引号和
    空格, 这里统一去引号去空格后按规范名匹配。某列缺失则取 nan。
    """
    history_path = Path(history_path)
    if not history_path.exists():
        # SU2 可能写成 history.csv / .dat, 容错探测同名其他后缀
        for alt in (history_path.with_suffix(".csv"),
                    history_path.with_suffix(".dat")):
            if alt.exists():
                history_path = alt
                break
        else:
            return {"Cl": math.nan, "Cd": math.nan, "Cm": math.nan,
                    "rms_rho": math.nan, "inner_iter": -1}

    with history_path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {"Cl": math.nan, "Cd": math.nan, "Cm": math.nan,
                "rms_rho": math.nan, "inner_iter": -1}

    def norm(name: str) -> str:
        return name.strip().strip('"').strip().lower()

    header = [norm(h) for h in rows[0]]
    last = rows[-1]
    col = {h: i for i, h in enumerate(header)}

    def get(*names: str) -> float:
        for n in names:
            if n in col and col[n] < len(last):
                return _to_float(last[col[n]])
        return math.nan

    return {
        "Cl": get("cl"),
        "Cd": get("cd"),
        "Cm": get("cmz", "cm"),
        "rms_rho": get("rms[rho]", "rms_rho"),
        "inner_iter": int(get("inner_iter")) if not math.isnan(
            get("inner_iter")) else -1,
    }


def run_su2(coords: np.ndarray, alpha: float, Re: float, *,
            mach: float = 0.1, tu: float = 0.002, turb2lam: float = 10.0,
            max_iter: int = 8000, mesh_path: str | Path | None = None,
            workdir: str | Path | None = None, timeout: float = 1800.0,
            rms_converged: float = -6.0, keep_files: bool = False,
            verbose: bool = False) -> dict:
    """单工况 (alpha, Re) SU2 转捩求解, 返回 {Cl, Cd, Cm, converged, ...}。

    参数
    ----
    coords    : (M,2) 翼型坐标 (generate_airfoil 输出)。
    mesh_path : 若给定则复用现成网格, 否则按 Re 现生成。
    timeout   : SU2 子进程墙钟超时 (秒); 超时记为不收敛。
    rms_converged : 判定收敛的密度残差阈值 (末步 rms[Rho] <= 此值视为残差收敛)。
                    无论残差是否达标, 只要力系数有限即返回数值。
    keep_files : True 则保留工作目录 (调试用)。

    返回除 Cl/Cd/Cm/converged 外还含: residual_ok(残差是否达标),
    rms_rho, inner_iter, returncode, workdir, reason(失败原因)。
    """
    coords = np.asarray(coords, dtype=float)
    base = {"Cl": None, "Cd": None, "Cm": None, "converged": False,
            "residual_ok": False, "rms_rho": math.nan, "inner_iter": -1,
            "returncode": None, "workdir": None, "reason": ""}

    if not SU2_AVAILABLE:
        return {**base, "reason": "SU2_CFD 不在 PATH (需 conda activate aero-opt)"}

    tmp = workdir is None
    wd = Path(tempfile.mkdtemp(prefix="su2_")) if tmp else Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    base["workdir"] = str(wd)

    try:
        # --- 网格 ---
        if mesh_path is not None:
            mesh_file = Path(mesh_path)
            if not mesh_file.exists():
                return {**base, "reason": f"网格不存在: {mesh_file}"}
        else:
            mesh_file = wd / "airfoil_hf.su2"
            generate_mesh(coords, Re=Re, out_path=mesh_file, verbose=verbose)

        # --- 配置 ---
        cfg_text = fill_config(
            mesh_file=str(mesh_file), alpha=alpha, Re=Re, mach=mach,
            tu=tu, turb2lam=turb2lam, max_iter=max_iter,
            conv_name=str(wd / "history"),
            restart_name=str(wd / "restart_flow.dat"))
        cfg_path = wd / "case.cfg"
        cfg_path.write_text(cfg_text)

        # --- 求解 ---
        env = dict(os.environ)
        prefix = env.get("CONDA_PREFIX")
        if prefix:                       # README 所述 SU2 运行期环境变量
            env.setdefault("SU2_RUN", str(Path(prefix) / "bin"))
            env.setdefault("SU2_HOME", prefix)
        log_path = wd / "su2.log"
        with log_path.open("w") as log:
            try:
                proc = subprocess.run(
                    [SU2_BIN, str(cfg_path)], cwd=str(wd), env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                return {**base, "reason": f"SU2 超时 (>{timeout:.0f}s)",
                        "returncode": -9}
        base["returncode"] = rc

        # --- 解析 (即使 rc!=0 也尝试读已写出的 history) ---
        hist = parse_history(wd / "history")
        cl, cd, cm = hist["Cl"], hist["Cd"], hist["Cm"]
        forces_ok = all(map(lambda v: not math.isnan(v), (cl, cd))) and cd > 0
        residual_ok = (not math.isnan(hist["rms_rho"])
                       and hist["rms_rho"] <= rms_converged)

        if not forces_ok:
            tail = log_path.read_text()[-400:] if log_path.exists() else ""
            return {**base, "rms_rho": hist["rms_rho"],
                    "inner_iter": hist["inner_iter"],
                    "reason": f"力系数无效 (rc={rc}). 日志尾: ...{tail}"}

        return {
            "Cl": float(cl), "Cd": float(cd),
            "Cm": float(cm) if not math.isnan(cm) else None,
            "converged": True,           # 力系数有限即视为可用数据点
            "residual_ok": bool(residual_ok),
            "rms_rho": float(hist["rms_rho"]),
            "inner_iter": hist["inner_iter"],
            "returncode": rc, "workdir": str(wd), "reason": "",
        }
    except Exception as e:               # noqa: BLE001 — 批量层需要永不抛出
        return {**base, "reason": f"{type(e).__name__}: {e}"}
    finally:
        if tmp and not keep_files:
            shutil.rmtree(wd, ignore_errors=True)
