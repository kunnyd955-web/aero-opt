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
| Phase 2 — SU2 Transition CFD | ✅ Done | gmsh BL mesh + SU2 8.5 SST/γ-Reθ; NACA0012 Re=2e5 → CL 0.446 / Cd 0.021 |
| Phase 3 — Co-Kriging Surrogate | ✅ Done | MFK Cl/Cd surrogate over LF(808)+HF(**55**, expanded from 26 via `--append`); 55-pt LOO-CV: **Cd R²≈0.88** via Co-Kriging (RMSE −12% vs HF-only), **Cl R²≈0.67** via HF-only KRG (multi-fidelity hurts Cl, +23% RMSE); 6 tests pass |
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

# Small batch low-fidelity (XFOIL) sampling / 小批量低保真采样
python scripts/sample_lf_batch.py --n 20

# Small batch high-fidelity (SU2 transition) sampling / 小批量高保真采样
# ~2 min/point: mesh + SST/γ-Reθ CFD. Needs SU2 env vars (see Setup).
python scripts/sample_hf_batch.py --n 2

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

- **SU2 transition drag bias**: SST/γ-Reθ RANS predicts higher Cd than XFOIL's panel + eᴺ (NACA0012 Re=2e5: Cd≈0.021 vs XFOIL ≈0.013), partly from the blunt-TE base drag (0.4%c gap). This is the *systematic LF→HF bias* Co-Kriging (Phase 3) is built to correct, not an error. Symmetric-airfoil Cm≈0 confirms solver correctness.
  **SU2 转捩阻力偏置**：SST/γ-Reθ RANS 的 Cd 系统性高于 XFOIL（含钝后缘基阻），这正是 Phase 3 Co-Kriging 要校正的低-高保真系统偏差；对称翼型 Cm≈0 佐证求解正确。

- **HF turbulence-intensity calibration**: Inlet `FREESTREAM_TURBULENCEINTENSITY=0.002` (0.2%) compensates convective decay so leading-edge Tu≈0.1% (Ncrit=9 equivalent). The exact factor should be back-calculated from a measured LE Tu once volume output is post-processed (plan §5.2).
  **HF 湍流度校准**：入口 0.2% 补偿衰减使前缘约 0.1%，精确因子待用前缘实测值反推。

- **Per-output fidelity choice (HF-sample-driven)**: Expanding the HF set 26→55 valid points (`sample_hf_batch.py --append --seed <new>`) sharply improved both surrogates, but the 55-point LOO-CV reveals the best model differs per output. **Cd:** Co-Kriging wins (R²≈0.88, RMSE 12% below single-fidelity KRG) — the LF→HF bias is systematic and learnable, so multi-fidelity transfer helps. **Cl:** single-fidelity HF-only Kriging now wins (R²≈0.67) while Co-Kriging is worse (R²≈0.50, +23% RMSE) — the XFOIL→SU2 Cl discrepancy is not consistent enough to transfer, so borrowing LF data injects noise. Takeaway for Phase 4: drive Cd from the Co-Kriging model and Cl from HF-only Kriging. Both outputs are now in a usable range (vs the earlier 26-point CV: Cd R²≈0.61, Cl R²≈0.20). Training-side mitigations in place: input normalization by `LF_BOUNDS`, LF-point caps (`--fit-lf-cap`/`--cv-lf-cap`) to bound MFK's O(n³) cost.
  **按输出选择保真度（取决于 HF 样本量）**：把有效 HF 点从 26 扩到 55 后两个代理都大幅改善，但 55 点 LOO-CV 显示二者最优模型不同。**Cd**：Co-Kriging 最优（R²≈0.88，RMSE 较单保真低 12%），LF→HF 偏差系统且可学，多保真有效。**Cl**：单保真纯 HF Kriging 反而最优（R²≈0.67），Co-Kriging 偏弱（R²≈0.50，RMSE 高 23%），XFOIL→SU2 的 Cl 差异不够一致，借 LF 反添噪。Phase 4 据此：Cd 用 Co-Kriging、Cl 用纯 HF Kriging。两者均已可用（对比 26 点：Cd 0.61、Cl 0.20）。

---

## License / 许可证

MIT
