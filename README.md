# aero-opt — 2D 翼型多保真度气动形状优化

CST 参数化 + XFOIL($e^N$)/SU2($\gamma$-$Re_\theta$) 多保真度采样 + Co-Kriging 代理模型
+ CMA-ES/NSGA-II 优化，最大化低-中雷诺数翼型升阻比 $L/D$。

完整方案见 [`airfoil_opt_project_plan.md`](airfoil_opt_project_plan.md)。

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0 CST 参数化 | ✅ 完成 | NACA 0012/2412 拟合 MSE < 1e-5 |
| Phase 1 XFOIL 采样 | ✅ 打通 | 小批量端到端跑通，内存绑定后端 |
| Phase 2 SU2 转捩 | ⬜ 待做 | 模板占位 (`hf_solver/templates/`) |
| Phase 3 Co-Kriging | ⬜ 待做 | 脚本占位 (`surrogate/`) |
| Phase 4 优化 | ⬜ 待做 | — |

## 环境配置

```bash
conda env create -f environment.yml      # 创建 aero-opt 环境
conda activate aero-opt
```

环境已含: SU2 8.5、gmsh、openmpi/mpi4py、numpy/scipy/sklearn、smt、cma、pymoo，
以及 **xfoil-python**（从 GitHub 源码编译的内存绑定后端，非 subprocess）。

> SU2 环境变量（仅 Phase 2 需要）：
> ```bash
> export SU2_RUN=$CONDA_PREFIX/bin
> export SU2_HOME=$CONDA_PREFIX
> export PATH=$SU2_RUN:$PATH
> ```

## 快速验证

```bash
conda activate aero-opt
python -m pytest tests/ -v                              # Phase 0 + Phase 1 测试
python scripts/sample_lf_batch.py --n 20                # 端到端小批量采样
python doe/doe_lhs.py                                   # 生成全量 LHS 采样点
```

## 目录

```
geometry/   CST 参数化 (cst_params)、NACA 真值 (naca)、网格生成 (mesh_generator, Phase 2)
doe/        拉丁超立方采样 (doe_lhs)
lf_solver/  低保真求解器: XFOIL 封装 (run_xfoil)
hf_solver/  高保真求解器: SU2 转捩模型 (Phase 2)
surrogate/  Co-Kriging 代理模型 (Phase 3)
optimization/ CMA-ES / NSGA-II (Phase 4)
scripts/    采样驱动脚本
tests/      验证测试
notebooks/  分析与可视化
```

## 对方案文档的修正

实施中发现并修正了原方案的若干技术问题，记录于此以便对照：

1. **设计空间边界**（`doe/doe_lhs.py`）：方案原文把 10 个 CST 系数全设为对称区间
   `[-0.3, 0.3]`，实测仅约 **1.4%** 的随机翼型几何合法。改为"上翼面系数取正
   `[0.05, 0.30]`、下翼面取负 `[-0.30, -0.05]`"后合法率升至 **~97%**。
2. **XFOIL 后端**：方案称"xfoil-python 调用系统 XFOIL"不准确；该包用 f2py 把
   XFOIL 编译为共享库（内存绑定，不起子进程）。PyPI sdist 残缺，须从 GitHub 源码装。
3. **CST x 域裁剪**（`geometry/cst_params.py`）：弯度翼型前缘 `x` 经几何旋转会微小
   越界，导致 `x**0.5` 出现 NaN；已在 `cst_surface` 内裁剪到 `[0, 1]`。
4. **XFOIL 单点求解**：退化的 `aseq(a, a, da)` 返回空，单点必须用 `xf.a(alpha)`。

## 已知限制

- XFOIL 对**厚翼型（>12%）在中雷诺数下的黏性求解**存在固有收敛困难，小批量收敛率
  约 84%。方案缓解措施：收窄攻角上限、对失败点回退全湍流 RANS（见方案 4.3）。
- SU2 配置使用旧版 `PHYSICAL_PROBLEM=` 语法；SU2 8.x 已改为 `SOLVER=`，Phase 2
  实施时按 `hf_solver/templates/` 中的占位说明校准。
