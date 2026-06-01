"""网格生成 (方案 5.0)。

Phase 0-1 仅实现并验证 first_layer_height (自适应 y+ 第一层高度)。
Phase 2 用 gmsh 4.x Python API 实现 generate_mesh: 围绕翼型生成 O 型混合网格
(壁面附近结构化边界层四边形 + 外部非结构三角形), 输出 SU2 格式, 供 SU2 +
gamma-Re_theta 转捩模型使用。

转捩模型对 y+ 极敏感, 第一层高度由 first_layer_height(Re) 自适应给出, 不硬编码。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["first_layer_height", "generate_mesh"]


def first_layer_height(Re: float, chord: float = 1.0,
                       yplus_target: float = 0.5) -> float:
    """按雷诺数计算壁面第一层网格高度, 目标 y+ = yplus_target (方案 5.0)。

    γ-Reθ 转捩模型要求 y+ < 1 (理想 0.1~0.5)。优化中翼型跨两个数量级的 Re,
    固定第一层高度无法通用, 必须写成 Re 的函数。

    推导 (Blasius 平板层流近似):
        Cf ≈ 0.664 / sqrt(Re)
        u_tau / U_inf = sqrt(Cf / 2)
        Δy = y+ * nu / u_tau = y+ * chord / (Re * sqrt(Cf/2))

    参考值 (默认 y+=0.5): Re=5e4 -> ~2.6e-4 c; Re=2e5 -> ~9.2e-5 c;
    Re=5e5 -> ~4.6e-5 c。(方案文档列的是 y+=1.0 的两倍值, 此处以代码默认为准。)
    """
    Cf = 0.664 / Re ** 0.5
    u_tau_ratio = (Cf / 2.0) ** 0.5
    return yplus_target * chord / (Re * u_tau_ratio)


# 物理组标记名: SU2 配置里以同名引用 (MARKER_HEATFLUX / MARKER_FAR)。
MARKER_AIRFOIL = "airfoil"
MARKER_FARFIELD = "farfield"


def generate_mesh(
    coords: np.ndarray,
    Re: float,
    out_path: str | Path,
    *,
    chord: float = 1.0,
    yplus_target: float = 0.5,
    n_bl_layers: int = 25,
    bl_ratio: float = 1.20,
    farfield_radius: float = 20.0,
    size_le: float | None = None,
    size_te: float | None = None,
    verbose: bool = False,
) -> dict:
    """围绕翼型生成 O 型混合网格, 输出 SU2 .su2 文件 (方案 5.0)。

    结构:
      - 翼型上下表面 + 钝后缘底边构成内边界 (MARKER_HEATFLUX 壁面);
      - 半径 farfield_radius 的圆为远场外边界 (MARKER_FAR);
      - 壁面附近用 BoundaryLayer 场拉出 n_bl_layers 层四边形 (第一层
        = first_layer_height(Re), 增长比 bl_ratio), 外部为三角形;
      - Distance+Threshold 背景场控制由壁面到远场的网格尺寸过渡。

    参数
    ----
    coords : (M, 2) 翼型坐标 (Selig 格式, generate_airfoil 输出)。钝后缘:
             首点 = 上翼面后缘, 末点 = 下翼面后缘, 二者不重合。
    Re     : 当前工况雷诺数, 决定第一层高度。
    out_path : 输出 .su2 路径。
    size_le / size_te : 前/后缘附近目标网格尺寸 (弦长归一)。默认按弦长推。

    返回
    ----
    dict: {path, n_nodes, n_elements, h_wall, n_bl_layers, markers}。
          失败抛 RuntimeError (批量层捕获后跳过该点)。
    """
    import gmsh

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 10:
        raise ValueError(f"翼型坐标形状非法: {coords.shape}")

    h_wall = first_layer_height(Re, chord=chord, yplus_target=yplus_target)
    size_le = size_le if size_le is not None else 0.004 * chord
    size_te = size_te if size_te is not None else 0.008 * chord
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 去掉数值上重合的相邻点 (样条不允许重复点); 保持首末后缘点。
    keep = [0]
    for i in range(1, len(coords)):
        if np.hypot(*(coords[i] - coords[keep[-1]])) > 1e-9:
            keep.append(i)
    pts = coords[keep]
    # 若首末点几乎重合 (尖后缘), 钝后缘底边退化, 去掉末点闭合成环。
    blunt_te = np.hypot(*(pts[0] - pts[-1])) > 1e-6

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("airfoil")
        geo = gmsh.model.geo

        # --- 翼型表面点 ---
        # 后缘附近尺寸用 size_te, 前缘 (x 最小处) 用 size_le, 其余线性插值。
        x = pts[:, 0]
        x_min, x_max = x.min(), x.max()
        span = max(x_max - x_min, 1e-9)
        af_tags = []
        for px, py in pts:
            frac = (px - x_min) / span          # 0=前缘, 1=后缘
            lc = size_le + (size_te - size_le) * frac
            af_tags.append(geo.addPoint(px, py, 0.0, lc))

        # 翼型样条 (上后缘 -> 前缘 -> 下后缘)
        spline = geo.addSpline(af_tags)
        bl_curves = [spline]
        if blunt_te:
            te_base = geo.addLine(af_tags[-1], af_tags[0])  # 钝后缘底边
            bl_curves.append(te_base)
            airfoil_loop = geo.addCurveLoop([spline, te_base])
        else:
            airfoil_loop = geo.addCurveLoop([spline])

        # --- 远场圆 (以翼型 1/4 弦附近为心) ---
        cx, cy = x_min + 0.25 * span, float(np.mean(pts[:, 1]))
        R = farfield_radius * chord
        # 远场约 40 段: 太粗 (个位数段) 会让 SU2 远场边界条件失真。
        size_far = 0.15 * R
        cpt = geo.addPoint(cx, cy, 0.0, size_far)
        fpts = [
            geo.addPoint(cx + R, cy, 0.0, size_far),
            geo.addPoint(cx, cy + R, 0.0, size_far),
            geo.addPoint(cx - R, cy, 0.0, size_far),
            geo.addPoint(cx, cy - R, 0.0, size_far),
        ]
        arcs = [
            geo.addCircleArc(fpts[0], cpt, fpts[1]),
            geo.addCircleArc(fpts[1], cpt, fpts[2]),
            geo.addCircleArc(fpts[2], cpt, fpts[3]),
            geo.addCircleArc(fpts[3], cpt, fpts[0]),
        ]
        far_loop = geo.addCurveLoop(arcs)

        # 流体域 = 远场环 - 翼型孔
        surf = geo.addPlaneSurface([far_loop, airfoil_loop])
        geo.synchronize()

        # --- 边界层场: 壁面拉结构化四边形 ---
        f_bl = gmsh.model.mesh.field.add("BoundaryLayer")
        gmsh.model.mesh.field.setNumbers(f_bl, "CurvesList", bl_curves)
        gmsh.model.mesh.field.setNumber(f_bl, "Size", h_wall)
        gmsh.model.mesh.field.setNumber(f_bl, "Ratio", bl_ratio)
        gmsh.model.mesh.field.setNumber(f_bl, "NbLayers", n_bl_layers)
        gmsh.model.mesh.field.setNumber(f_bl, "Quads", 1)
        # 后缘两个尖角处放射状过渡, 避免边界层单元扭曲。
        fan_pts = [af_tags[0], af_tags[-1]] if blunt_te else [af_tags[0]]
        gmsh.model.mesh.field.setNumbers(f_bl, "FanPointsList", fan_pts)
        gmsh.model.mesh.field.setAsBoundaryLayer(f_bl)

        # --- 背景尺寸场: 壁面附近细, 远场粗 ---
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", bl_curves)
        gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 400)
        f_th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMin", size_te)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMax", size_far)
        gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0.05 * chord)
        gmsh.model.mesh.field.setNumber(f_th, "DistMax", 0.5 * R)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_th)

        # 关掉基于点 lc 的尺寸来源, 完全交给背景场 (BL 区除外)。
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay, 对 BL 友好

        # --- 物理组 (SU2 标记) ---
        gmsh.model.addPhysicalGroup(1, bl_curves, name=MARKER_AIRFOIL)
        gmsh.model.addPhysicalGroup(1, arcs, name=MARKER_FARFIELD)
        gmsh.model.addPhysicalGroup(2, [surf], name="fluid")

        gmsh.model.mesh.generate(2)

        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        etypes, etags, _ = gmsh.model.mesh.getElements(2)
        n_elements = int(sum(len(t) for t in etags))
        if n_elements == 0:
            raise RuntimeError("网格生成失败: 0 个面单元")

        gmsh.write(str(out_path))
    finally:
        gmsh.finalize()

    return {
        "path": out_path,
        "n_nodes": int(n_nodes),
        "n_elements": n_elements,
        "h_wall": h_wall,
        "n_bl_layers": n_bl_layers,
        "markers": [MARKER_AIRFOIL, MARKER_FARFIELD],
    }
