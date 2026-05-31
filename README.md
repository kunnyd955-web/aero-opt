# aero-opt

**2D Airfoil Multi-Fidelity Aerodynamic Shape Optimization**
**二维翼型多保真度气动形状优化**

CST parameterization · XFOIL (eᴺ) / SU2 (γ–Reθ) multi-fidelity sampling · Co-Kriging surrogate · CMA-ES / NSGA-II optimization

CST 参数化 · XFOIL (eᴺ) / SU2 (γ–Reθ) 多保真度采样 · Co-Kriging 代理模型 · CMA-ES / NSGA-II 优化

---

## Overview / 项目简介

**EN:** Maximize lift-to-drag ratio (L/D) for 2D airfoils at low-to-medium Reynolds numbers (Re = 10⁴–10⁶), where laminar-turbulent transition is the dominant physical mechanism. A multi-fidelity pipeline combines cheap XFOIL panel-method evaluations with expensive SU2 transition-model CFD, bridged by a Co-Kriging surrogate to enable gradient-free global optimization.

**中文：** 在低-中雷诺数（Re = 10⁴–10⁶）条件下，层流-湍流转捩是影响翼型阻力的核心因素。本项目构建多保真度优化流水线：以廉价的 XFOIL 面板法大规模采样，结合昂贵的 SU2 转捩 CFD 计算，通过 Co-Kriging 代理模型桥接，最终用无梯度全局优化器最大化翼型升阻比 L/D。

Full technical plan: [`airfoil_opt_project_plan.md`](airfoil_opt_project_plan.md)

---

## Pipeline Architecture / 流水线架构

```
Design Space (CST, 10 params)
        │
        ▼
  LHS Sampling (doe/)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
XFOIL LF   SU2 HF        ← multi-fidelity solvers
(lf_solver) (hf_solver)
   │         │
   └────┬────┘
        ▼
  Co-Kriging Surrogate (surrogate/)
        │
        ▼
  CMA-ES / NSGA-II (optimization/)
        │
        ▼
  Optimal Airfoil Geometry
```

---

## Progress / 当前进度

| Phase | Status | Description / 说明 |
|-------|--------|---------------------|
| Phase 0 — CST Parameterization | ✅ Done | NACA 0012/2412 fit MSE < 1e-5 |
| Phase 1 — XFOIL LF Sampling | ✅ Done | End-to-end batch sampling, in-memory backend |
| Phase 2 — SU2 Transition CFD | ⬜ Todo | Template placeholder in `hf_solver/templates/` |
| Phase 3 — Co-Kriging Surrogate | ⬜ Todo | Module stub in `surrogate/` |
| Phase 4 — Global Optimization | ⬜ Todo | — |

---

## Setup / 环境配置

```bash
conda env create -f environment.yml
conda activate aero-opt
```

**Included / 已含：** SU2 8.5 · gmsh · openmpi / mpi4py · numpy / scipy / scikit-learn · smt · cma · pymoo · xfoil-python (f2py in-memory backend, not subprocess)

> **SU2 environment variables (Phase 2 only / 仅 Phase 2 需要):**
> ```bash
> export SU2_RUN=$CONDA_PREFIX/bin
> export SU2_HOME=$CONDA_PREFIX
> export PATH=$SU2_RUN:$PATH
> ```

---

## Quick Start / 快速验证

```bash
conda activate aero-opt

# Run tests / 运行测试
python -m pytest tests/ -v

# Small batch end-to-end sampling / 小批量端到端采样
python scripts/sample_lf_batch.py --n 20

# Generate full LHS design points / 生成全量 LHS 采样点
python doe/doe_lhs.py
```

---

## Repository Structure / 目录结构

```
aero-opt/
├── geometry/        CST parameterization, NACA reference, mesh generation (Phase 2)
│                    CST 参数化、NACA 真值、网格生成
├── doe/             Latin Hypercube Sampling
│                    拉丁超立方采样
├── lf_solver/       Low-fidelity solver: XFOIL wrapper
│                    低保真求解器：XFOIL 封装
├── hf_solver/       High-fidelity solver: SU2 transition model (Phase 2)
│                    高保真求解器：SU2 转捩模型
├── surrogate/       Co-Kriging surrogate model (Phase 3)
│                    Co-Kriging 代理模型
├── optimization/    CMA-ES / NSGA-II optimizer (Phase 4)
│                    全局优化器
├── scripts/         Batch sampling drivers / 采样驱动脚本
├── tests/           Unit & integration tests / 验证测试
├── notebooks/       Analysis & visualization / 分析与可视化
├── environment.yml  Conda environment spec
└── environment.lock.yml  Locked dependencies
```

---

## Implementation Notes / 实施注记

Deviations from the original plan discovered during implementation:
实施中发现并修正的原方案技术问题：

1. **Design space bounds** (`doe/doe_lhs.py`): Original plan set all 10 CST coefficients to the symmetric range `[-0.3, 0.3]`, yielding only ~1.4% geometrically valid airfoils. Changed to upper-surface `[0.05, 0.30]` and lower-surface `[-0.30, -0.05]`; validity rate rose to ~97%.
   **设计空间边界**：原方案对称区间导致合法率仅 1.4%，改为上/下翼面分区后升至约 97%。

2. **XFOIL backend**: The plan described "xfoil-python calls system XFOIL" — incorrect. The package compiles XFOIL as a shared library via f2py (in-memory, no subprocess). PyPI sdist is incomplete; install from GitHub source.
   **XFOIL 后端**：该包通过 f2py 将 XFOIL 编译为共享库，非子进程调用；须从 GitHub 源码安装。

3. **CST x-domain clipping** (`geometry/cst_params.py`): Cambered airfoil leading-edge coordinates can slightly exceed `[0, 1]` after geometric rotation, causing `x**0.5` NaN. Fixed by clamping inside `cst_surface`.
   **CST x 域裁剪**：弯度翼型前缘坐标旋转后可能微小越界，导致 NaN，已在函数内裁剪。

4. **XFOIL single-point solve**: Degenerate `aseq(a, a, da)` returns empty; single-point calls must use `xf.a(alpha)`.
   **XFOIL 单点求解**：退化序列返回空，单点必须用 `xf.a(alpha)`。

---

## Known Limitations / 已知限制

- **XFOIL convergence**: Viscous solver convergence is inherently difficult for thick airfoils (>12%) at medium Re. Small-batch convergence rate ~84%. Mitigation: narrow AoA upper bound; fall back to fully-turbulent RANS for failed points (see plan §4.3).
  **XFOIL 收敛**：厚翼型中雷诺数下固有收敛困难，小批量收敛率约 84%。缓解措施：收窄攻角上限，失败点回退全湍流 RANS。

- **SU2 config syntax**: Templates use legacy `PHYSICAL_PROBLEM=` syntax; SU2 8.x changed to `SOLVER=`. Calibrate against placeholder notes in `hf_solver/templates/` when implementing Phase 2.
  **SU2 配置语法**：模板使用旧版语法，SU2 8.x 已改为 `SOLVER=`，Phase 2 实施时需校准。

---

## License / 许可证

MIT
