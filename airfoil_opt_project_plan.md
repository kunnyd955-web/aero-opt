# 2D 翼型气动形状优化项目计划

**项目性质**：计算流体力学 × 代理模型优化，带转捩模型的多保真度气动形状优化

**运行环境**：WSL-Ubuntu + conda 环境（SU2、XFOIL）；Windows 侧保留 Fluent/PyFluent 现有工作流

---

## 目录

1. [物理背景与问题定义](#1-物理背景与问题定义)
2. [整体架构](#2-整体架构)
3. [Phase 0 — 翼型参数化（CST / XFOIL eN）](#3-phase-0--翼型参数化cst--xfoil-en)
4. [Phase 1 — 低保真采样（XFOIL / 全湍流 RANS）](#4-phase-1--低保真采样xfoil--全湍流-rans)
5. [Phase 2 — 高保真采样（带转捩模型的 CFD / LES）](#5-phase-2--高保真采样带转捩模型的-cfd--les)
6. [Phase 3 — Co-Kriging 多保真代理模型](#6-phase-3--co-kriging-多保真代理模型)
7. [Phase 4 — 气动优化](#7-phase-4--气动优化)
8. [验证策略](#8-验证策略)
9. [工具链与环境配置](#9-工具链与环境配置)
10. [目录结构](#10-目录结构)
11. [里程碑与时间规划](#11-里程碑与时间规划)
12. [已知风险与缓解措施](#12-已知风险与缓解措施)

---

## 1. 物理背景与问题定义

### 1.1 为什么转捩是核心问题

低到中雷诺数翼型（$Re = 10^4 \sim 10^6$）的边界层在大部分翼面上保持**层流**，并在某个位置发生**层流→湍流转捩**。这一过程对阻力有决定性影响：

- **层流边界层**：摩擦阻力极低，但对逆压梯度脆弱，容易分离形成**层流分离泡**
- **湍流边界层**：摩擦阻力高但不易分离，附着性好
- **转捩位置**随攻角 $\alpha$、雷诺数 $Re$、翼型形状剧烈变化

如果 CFD 模型错误地假设流场全为湍流（**全湍流 RANS**），会严重高估摩擦阻力，导致 $C_d$ 偏高、$L/D$ 偏低，优化出的"最优翼型"在真实流动中并不是真正最优的。

### 1.2 转捩判据：$e^N$ 方法

$e^N$ 方法（Mack 1956，Smith & Gamberoni 1956）是工程上最成熟的转捩预测方法：

$$
N = \int_{x_0}^{x_{tr}} \sigma \, dx
$$

其中 $\sigma$ 是 Tollmien-Schlichting 波的空间增长率，$x_0$ 是不稳定性起始点，$x_{tr}$ 是转捩点。当 N 值积累到阈值 $N_{cr}$ 时判定转捩发生：

| 来流湍流度 $T_u$ | $N_{cr}$ |
|:---:|:---:|
| $< 0.1\%$（极低，如飞行条件）| 9–11 |
| $0.1\%$（低湍流风洞） | 8–9 |
| $0.3\%$（普通风洞） | 6–7 |
| $> 1\%$（高湍流度） | 2–4 |

**XFOIL** 内置了 $e^N$ 方法，计算极快（毫秒级），是低保真层的首选。**SU2** 通过 $\gamma$-$Re_\theta$ 转捩模型（Langtry-Menter）在 CFD 层面实现类似功能，精度更高但计算成本大两个数量级。

> **⚠️ 关键一致性要求：** XFOIL 的 $N_{cr}$ 和 SU2 的 `FREESTREAM_TURBULENCEINTENSITY` 必须对应同一物理来流湍流度，否则两个求解器预测的转捩位置在系统上不可比，Co-Kriging 的误差修正项会混入参数不一致噪声。换算关系（Mack 经验公式）：
>
> $$N_{cr} = -8.43 - 2.4 \ln\!\left(\frac{T_u}{100}\right), \quad T_u \text{ 单位为 \%}$$
>
> | $N_{cr}$（XFOIL） | $T_u$（SU2 `FREESTREAM_TURBULENCEINTENSITY`） | 对应环境 |
> |:---:|:---:|:---|
> | 11 | 0.0003（0.03%） | 飞行条件、极低湍流度 |
> | 9 | 0.001（0.1%） | 低湍流风洞（本项目默认） |
> | 7 | 0.004（0.4%） | 普通风洞 |
> | 5 | 0.014（1.4%） | 高湍流度环境 |
>
> 本项目统一取 $N_{cr} = 9$，SU2 设 `FREESTREAM_TURBULENCEINTENSITY= 0.001`。

### 1.3 设计空间定义

| 变量 | 范围 | 说明 |
|------|------|------|
| CST 形状参数 $\mathbf{A}$ | $[-0.3, +0.3]^{10}$ | 上下翼面各 5 个 Bernstein 系数 |
| 攻角 $\alpha$ | $-2°\sim 12°$ | 覆盖正常工作区间，避开失速 |
| 雷诺数 $Re$ | $5\times10^4 \sim 5\times10^5$ | 无人机/小型飞机典型区间 |
| 马赫数 $Ma$ | $0.05 \sim 0.2$ | 低速不可压，视为固定（$Ma = 0.1$）或参数化 |

**优化目标**：最大化升阻比 $L/D = C_l / C_d$

**约束**：
- 翼型最大厚度比 $t/c \geq 10\%$（结构强度下限）
- $C_l \geq 0.8$（保证足够升力）
- 翼型轮廓不能自交、不能出现负厚度

---

## 2. 整体架构

```
Phase 0: 参数化几何
  CST 系数 → 翼型坐标 (.dat) → Gmsh 生成网格 (.su2)
        ↓
Phase 1: 低保真采样（LF, 1000 个样本）
  XFOIL [eN 转捩]  ──────────────────────────┐
  或 SU2 全湍流 RANS                          │
        ↓                                     │
  (x, α, Re) → (Cl_LF, Cd_LF)               │
        ↓                                     │
Phase 2: 高保真采样（HF, 50 个样本）          │
  SU2 + γ-Reθ 转捩模型 / LES                 │
        ↓                                     │
  (x, α, Re) → (Cl_HF, Cd_HF)               │
        ↓                                     ↓
Phase 3: Co-Kriging 代理模型（分量独立建模）
  融合 LF（趋势）+ HF（误差修正）
  → Ĉl(x, α, Re)  ← 独立 Co-Kriging
  → Ĉd(x, α, Re)  ← 独立 Co-Kriging
  [禁止直接拟合 L/D：Cd→0 时 L/D 有理函数奇异]
        ↓
Phase 4: 优化搜索
  目标函数：Ĉl(x) / Ĉd(x) 在代理模型上实时计算
  NSGA-II 或 CMA-ES 运行
  → Pareto 最优翼型集合
        ↓
验证：高保真 CFD 重跑候选翼型，对比代理模型预测
```

---

## 3. Phase 0 — 翼型参数化（CST / XFOIL $e^N$）

### 3.1 CST（Class-Shape-Transformation）方法

#### 原理

翼型上下表面的型函数用**Bernstein 多项式**展开，乘以保证前缘圆钝、后缘闭合的类函数：

$$
y(x) = \underbrace{x^{N_1}(1-x)^{N_2}}_{C(x)} \cdot \underbrace{\sum_{i=0}^{N} A_i B_{i,N}(x)}_{S(x)} + x \cdot y_{te}
$$

其中 $B_{i,N}(x) = \binom{N}{i} x^i (1-x)^{N-i}$ 是 Bernstein 基函数，$y_{te}$ 是后缘偏移量。对于标准翼型取 $N_1 = 0.5$，$N_2 = 1.0$。

选取 $N=4$（5 个系数）分别拟合上下翼面，共 **10 个设计变量**，可覆盖从对称翼型到高弯度翼型的广泛形态。

#### CST 实现模板

```python
# cst_params.py
import numpy as np
from math import comb

def bernstein(n: int, k: int, x: np.ndarray) -> np.ndarray:
    """Bernstein 基函数 B_{k,n}(x)，向量化实现。"""
    return comb(n, k) * (x ** k) * ((1 - x) ** (n - k))

def cst_surface(A: np.ndarray, x: np.ndarray,
                N1: float = 0.5, N2: float = 1.0,
                yte: float = 0.0) -> np.ndarray:
    """
    CST 翼面纵坐标。

    参数
    ----
    A   : Bernstein 系数向量，长度 N+1
    x   : 弦向坐标，∈ [0, 1]
    yte : 后缘偏移量（半厚度）。
          y(x=1) = yte，因此总后缘厚度 = yte_upper - yte_lower。
          yte=0 → 绝对尖后缘（⚠️ SU2 中易引发边界层网格扭曲，慎用）。
          推荐取 yte_upper = +0.002, yte_lower = -0.002（总厚度 4‰ 弦长）。
    """
    N = len(A) - 1
    C = x ** N1 * (1 - x) ** N2            # 类函数：保证前缘圆钝、后缘趋近 yte
    S = sum(A[i] * bernstein(N, i, x) for i in range(N + 1))
    return C * S + x * yte

def generate_airfoil(A_upper: np.ndarray, A_lower: np.ndarray,
                     yte_upper: float = 0.002, yte_lower: float = -0.002,
                     n_pts: int = 201) -> np.ndarray:
    """
    生成翼型坐标（Selig 格式：下翼面后缘 → 前缘 → 上翼面后缘）。

    yte_upper / yte_lower 默认值 ±0.002 对应总后缘厚度 4‰ 弦长，
    足以避免 SU2 网格划分在后缘点出现极度扭曲的边界层单元。
    """
    t = np.linspace(0, 1, n_pts)
    x = 0.5 * (1 - np.cos(np.pi * t))      # 余弦加密

    y_upper = cst_surface(A_upper, x, yte=yte_upper)
    y_lower = cst_surface(A_lower, x, yte=yte_lower)

    coords = np.vstack([
        np.column_stack([x[::-1], y_upper[::-1]]),
        np.column_stack([x[1:],   y_lower[1:]]),
    ])
    return coords

def check_geometry(A_upper: np.ndarray, A_lower: np.ndarray,
                   yte_upper: float = 0.002, yte_lower: float = -0.002,
                   t_min: float = 0.10) -> bool:
    """
    几何约束检查：
      1. 最大厚度比 ≥ t_min（结构下限）
      2. 无自交（全弦向厚度 > 0）
      3. 后缘厚度 ≥ 0（不允许下翼面高于上翼面）
    """
    x = np.linspace(0.005, 0.995, 300)
    y_up = cst_surface(A_upper, x, yte=yte_upper)
    y_lo = cst_surface(A_lower, x, yte=yte_lower)
    thickness = y_up - y_lo
    te_thick = yte_upper - yte_lower        # 后缘厚度
    return (
        np.all(thickness > 0)               # 无自交
        and np.max(thickness) >= t_min      # 最大厚度满足结构要求
        and te_thick >= 0                   # 后缘合法
    )
```

### 3.2 备用方案：XFOIL $e^N$ 直接驱动

如果某些翼型形态 CST 拟合残差 > 0.005c（超过弦长的 0.5%），退回到在 XFOIL 的几何扰动框架下直接优化翼型坐标点，用 $e^N$ 作为转捩判据而非 SU2 $\gamma$-$Re_\theta$ 模型。

判断标准：对基础翼型（如 NACA 2412）做 CST 拟合后，对比原始坐标和重建坐标的均方误差，若 MSE > 1e-5 则考虑切换。

```python
# 验证 CST 拟合精度（以 NACA 2412 为例）
from scipy.optimize import minimize

def cst_fit_error(A, x_target, y_target, **kwargs):
    y_fit = cst_surface(A, x_target, **kwargs)
    return np.sum((y_fit - y_target)**2)

# 初始猜测
A0 = np.zeros(5)
res = minimize(cst_fit_error, A0, args=(x_upper, y_upper),
               method='L-BFGS-B')
mse = res.fun / len(x_upper)
print(f"CST 拟合 MSE = {mse:.2e}（阈值 1e-5）")
```

---

## 4. Phase 1 — 低保真采样（XFOIL / 全湍流 RANS）

低保真层的目标是用**极低成本**覆盖尽可能广的设计空间，为 Co-Kriging 提供全局趋势信息。

### 4.1 设计空间采样：拉丁超立方

```python
# doe_lhs.py
from smt.sampling_methods import LHS
import numpy as np

# 设计变量边界：10 个 CST 系数 + α + Re
n_A = 10          # CST 系数（上下翼面各 5）
bounds = np.array([
    *[[-0.3, 0.3]] * n_A,     # CST 形状参数
    [-2.0, 12.0],              # 攻角 α (度)
    [5e4, 5e5],                # 雷诺数 Re
])

sampling = LHS(xlimits=bounds, criterion='ese')  # 最大熵采样
X_lf = sampling(1000)   # 1000 个低保真样本点
np.save('data/X_lf.npy', X_lf)
```

### 4.2 XFOIL 自动化（推荐首选）

XFOIL 将 $e^N$ 转捩模型内置，单次运行耗时约 50–200ms，1000 个样本约 10–30 分钟。

> **⚠️ 不要用 `subprocess` 调 XFOIL。** XFOIL 底层是 Fortran 代码，在大攻角分离或奇异几何时极易陷入死循环，导致 Python 进程永久挂起，1000 个样本的批量任务会因此卡死。应改用 **`xfoil-python`**（DARcorporation，内存绑定，不启动子进程），或 **AeroSandbox**。

安装：

```bash
pip install xfoil           # xfoil-python (DARcorporation)
# 或
pip install aerosandbox     # 备选，API 更现代
```

```python
# run_xfoil.py — 使用 xfoil-python 内存绑定
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil
import numpy as np

def run_xfoil(coords: np.ndarray, alpha: float, Re: float,
              Ncrit: float = 9.0, Ma: float = 0.1,
              max_iter: int = 100) -> dict:
    """
    通过内存绑定调用 XFOIL，计算单个 (α, Re) 工况的气动系数。

    相比 subprocess 的优势：
      - 速度快约 5× （无文件 I/O）
      - 内置异常捕获：Fortran 死循环会返回 NaN 而非挂起进程
      - 返回 None 而不是崩溃，1000 个批量样本中的少数失败点
        可以直接标记为无效并跳过
    """
    xf = XFoil()
    xf.airfoil = XFAirfoil(x=coords[:, 0], y=coords[:, 1])
    xf.Re      = Re
    xf.M       = Ma
    xf.n_crit  = Ncrit      # 与 SU2 Tu_inlet 通过 Mack 公式保持一致
    xf.max_iter = max_iter
    xf.print   = False      # 关闭 Fortran 控制台输出

    try:
        cl, cd, cm, cp = xf.a(alpha)
        if np.isnan(cl) or np.isnan(cd) or cd <= 0:
            return {'Cl': None, 'Cd': None, 'converged': False}
        return {
            'Cl': float(cl), 'Cd': float(cd), 'Cm': float(cm),
            'converged': True,
        }
    except Exception as e:
        return {'Cl': None, 'Cd': None, 'converged': False, 'error': str(e)}


def run_xfoil_polar(coords: np.ndarray,
                    alpha_start: float, alpha_end: float, alpha_step: float,
                    Re: float, Ncrit: float = 9.0, Ma: float = 0.1) -> list:
    """
    扫描一段攻角范围，返回极曲线列表。
    内部延续上一步收敛解，比逐点调用更高效，且失速点前后的收敛更稳定。

    建议每个采样翼型都跑完整极曲线（如 α = -2° → 12°，步长 1°），
    后处理时从极曲线中提取各攻角的 Cl/Cd，节省重复调用开销。
    """
    xf = XFoil()
    xf.airfoil  = XFAirfoil(x=coords[:, 0], y=coords[:, 1])
    xf.Re       = Re
    xf.M        = Ma
    xf.n_crit   = Ncrit
    xf.max_iter = 100
    xf.print    = False

    alphas, cls, cds, cms, _ = xf.aseq(alpha_start, alpha_end, alpha_step)

    results = []
    for i, a in enumerate(alphas):
        if np.isnan(cls[i]) or np.isnan(cds[i]) or cds[i] <= 0:
            results.append({'alpha': float(a), 'Cl': None, 'Cd': None,
                            'converged': False})
        else:
            results.append({'alpha': float(a), 'Cl': float(cls[i]),
                            'Cd': float(cds[i]), 'Cm': float(cms[i]),
                            'converged': True})
    return results
```

**注意事项**：
- 转捩位置（`xtr_top`/`xtr_bot`）通过 `xfoil-python` 暂不直接暴露；若需要转捩位置用于后处理对比，在 `xf.a(alpha)` 后访问 `xf.airfoil` 的边界层结构，或改用 AeroSandbox（有 `xtr_upper` 输出）
- 大攻角（$\alpha > 10°$）不收敛是正常现象（失速），收敛率 < 85% 时缩小攻角上限
- 完整极曲线策略（每翼型一次 `aseq`）优于逐点单次调用：前者计算时间约为后者的 50%

### 4.3 备用：全湍流 RANS（SU2，不加转捩模型）

当 XFOIL 收敛率不满足要求（<85%）或需要跨声速数据（$Ma > 0.3$）时，使用 SU2 全湍流 SA 模型作为低保真层：

```cfg
# su2_fullturbulent.cfg（关键差异段）
PHYSICAL_PROBLEM= NAVIER_STOKES
KIND_TURB_MODEL= SA          # Spalart-Allmaras，全湍流，不加转捩
SA_OPTIONS= NONE             # 明确不使用 SA-Transition 变体
MACH_NUMBER= 0.1
AOA= 4.0
REYNOLDS_NUMBER= 1E6
```

全湍流 RANS 单次约 5 分钟（4 核），1000 样本需要约 83 小时，因此建议并行化（见工具链一节）或缩减到 500 样本后依赖 Co-Kriging 的外推能力。

---

## 5. Phase 2 — 高保真采样（带转捩模型的 CFD / LES）

高保真层的目标是在**转捩最敏感的区域**提供精确的物理数据，用于校正低保真代理模型的系统误差。

### 5.0 网格专项：自适应 $y^+$ 控制

$\gamma$-$Re_\theta$ 转捩模型对壁面第一层网格高度极其敏感，必须严格保证 $y^+ < 1$（理想值 $0.1 \sim 0.5$）。优化中 1000 种不同翼型跨越两个数量级的 Re，固定的第一层高度无法在所有工况下满足要求。第一层高度必须写成关于 Re 的自适应函数：

```python
# 在 mesh_generator.py 中调用，不要硬编码 h_wall
def first_layer_height(Re: float, chord: float = 1.0,
                        yplus_target: float = 0.5) -> float:
    """
    根据雷诺数计算第一层网格高度，目标 y+ = yplus_target。

    推导（Blasius 平板层流近似）：
      Cf ≈ 0.664 / sqrt(Re_c)          (层流摩擦系数)
      u_tau = U_inf * sqrt(Cf / 2)     (摩擦速度)
      nu = U_inf * chord / Re          (运动粘度)
      Delta_y = y+ * nu / u_tau
             = y+ * chord / (Re * sqrt(Cf/2))

    取 y+ = 0.5（留足余量，实际 y+ 因前缘 Cf 更大而偏低）。
    """
    Cf = 0.664 / Re ** 0.5
    u_tau_ratio = (Cf / 2) ** 0.5          # u_tau / U_inf
    delta_y = yplus_target * chord / (Re * u_tau_ratio)
    return delta_y

# 验证：
# Re=5e4 → delta_y ≈ 5.3e-4 c
# Re=2e5 → delta_y ≈ 1.8e-4 c
# Re=5e5 → delta_y ≈ 9.0e-5 c
for Re in [5e4, 2e5, 5e5]:
    print(f"Re={Re:.0e}  h_wall = {first_layer_height(Re):.2e} c")
```

生成网格时传入当前工况的 Re：

```python
# mesh_generator.py 中
h_wall = first_layer_height(Re=current_Re)
gmsh_field_bl.NbLayers    = 25
gmsh_field_bl.Ratio       = 1.20
gmsh_field_bl.Thickness   = 15 * h_wall   # 边界层总厚度约 15h
gmsh_field_bl.hwall_n     = h_wall        # 第一层厚度自适应
```

### 5.1 样本点选取策略

不能随机均匀地选 50 个高保真点，这样会浪费在代理模型已经预测准确的区域。推荐两种策略：

**策略 A — 最大预期改进（EI-based，在代理模型训练后迭代加点）**：

先用初始 LF 数据训练一个粗代理模型，再通过 Expected Improvement（EI）准则选择下一个高保真仿真点，交替进行，称为 **Efficient Global Optimization（EGO）**。这会把高保真点自动集中在转捩最敏感的区域。

**策略 B — 先验知识驱动（项目早期无代理模型时使用）**：

转捩最活跃的参数区间先验已知：$Re = 10^5 \sim 3\times10^5$，$\alpha = 4°\sim 10°$，前缘圆钝度中等的翼型。在这个子空间内做拉丁超立方采样 50 个点：

```python
# 高保真采样区间缩减
bounds_hf = np.array([
    *[[-0.2, 0.2]] * 10,      # CST 稍微限制范围
    [4.0, 10.0],               # 攻角限制在转捩活跃区
    [1e5, 3e5],                # 转捩最敏感的 Re 区间
])
X_hf = LHS(xlimits=bounds_hf, criterion='ese')(50)
```

### 5.2 SU2 + $\gamma$-$Re_\theta$ 转捩模型

$\gamma$-$Re_\theta$ 模型（Langtry-Menter 2009）是目前工程 CFD 中使用最广泛的转捩模型，SU2 从 7.x 版本开始内置。它通过求解两个额外的输运方程（间歇因子 $\gamma$ 和转捩动量厚度 Reynolds 数 $\widetilde{Re}_{\theta t}$）来预测转捩位置。

```cfg
# su2_transition.cfg

%=========================== 物理模型 =============================%
PHYSICAL_PROBLEM= NAVIER_STOKES
KIND_TURB_MODEL= SST                    # 底层湍流模型用 SST（γ-Reθ 需要）
KIND_TRANS_MODEL= LM                    # Langtry-Menter 转捩模型

% ── 来流湍流度设置（注意衰减补偿，见下方说明）──────────────────────
% 目标：翼型前缘处有效湍流度 Tu_LE ≈ 0.1%（对应 Ncrit = 9）
% 问题：Tu 从入口（距翼型 ~15c）对流到前缘过程中会衰减，
%       若直接设 0.001（0.1%），到达前缘时实际 Tu 可能已降至 0.02%，
%       相当于 Ncrit ≈ 11，与 XFOIL 设置不一致，Co-Kriging 误差修正失效。
% 解决：首次运行后提取前缘上游 0.5c 处的实际 Tu，按比例反推入口值。
%       典型修正因子约 1.5–3.0（取决于计算域大小和涡粘比）。
FREESTREAM_TURBULENCEINTENSITY= 0.0020  # 入口设 0.2%，补偿衰减后 LE 约 0.1%
FREESTREAM_TURB2LAMTIMERATIOEDDY= 10.0  # 涡粘比：较低值减缓湍流衰减速率
                                         # 初始取 10，首次验证后调整

%=========================== 来流条件 =============================%
MACH_NUMBER= 0.1
AOA= 4.0                                % 攻角（每个样本覆盖）
REYNOLDS_NUMBER= 2E5                    % 雷诺数（每个样本覆盖）
REYNOLDS_LENGTH= 1.0                    % 参考长度（弦长 = 1m）

%=========================== 网格 =================================%
MESH_FILENAME= airfoil_hf.su2
MESH_FORMAT= SU2

%=========================== 求解器 ===============================%
LINEAR_SOLVER= FGMRES
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 2.0                         % 转捩模型对初始条件敏感，从低 CFL 开始
CFL_ADAPT= YES                          % 自适应 CFL：收敛后自动提升到 5
CFL_ADAPT_PARAM= ( 0.1, 2.0, 2.0, 10.0 )
MAX_ITER= 8000                          % 比全湍流 RANS 多留余量
CONV_RESIDUAL_MINVAL= -12

%=========================== 输出 =================================%
SOLUTION_FILENAME= solution_hf.dat
CONV_FILENAME= convergence_hf
BREAKDOWN_FILENAME= forces_hf.dat
```

**关键参数说明**：

- **湍流度衰减问题（重要）**：RANS 中湍流变量从远场入口对流到翼型前缘会发生物理耗散，大计算域（> 30c）尤为明显。应在第一个验证工况（NACA 0012，Re=2e5）运行后，提取翼型前缘上游 0.5c 处的实际 $T_u$，按目标值反推入口设定值。具体做法：
  ```python
  # 提取前缘上游点的 Tu（通过 SU2 的 VOLUME_OUTPUT 和场数据读取）
  # Tu_local = sqrt(2/3 * k) / U_inf，其中 k 是湍动能
  target_Tu_LE = 0.001   # 目标 0.1%
  measured_Tu_LE = 0.0004  # 实测值（示例）
  correction_factor = target_Tu_LE / measured_Tu_LE
  new_inlet_Tu = 0.0020 * correction_factor  # 调整入口设定值
  ```

- **SST 必须项**：$\gamma$-$Re_\theta$ 模型只能与 SST 配合使用，不能用 SA

- **CFL 自适应**：从低 CFL（2.0）开始，让转捩模型有充足迭代时间建立稳定的间歇因子场

### 5.3 备用：LES（大涡模拟）

LES 适用于需要解析层流分离泡精细结构的场景（$Re < 10^5$，分离泡主导气动性能）。

**限制条件**：二维 LES 在物理上不自洽（湍流本质是三维的），仅作为参考。完整 3D LES 计算成本极高（单个工况 > 100 核小时），建议在 Phase 2 中仅用于 5–10 个最关键的验证点，不作为 50 个高保真样本的主要来源。

```cfg
# su2_les.cfg（仅用于极少数关键验证点）
PHYSICAL_PROBLEM= NAVIER_STOKES
KIND_TURB_MODEL= NONE                   % 无湍流模型：直接数值模拟
NAVIER_STOKES_RUNGE_KUTTA= YES          % 显式时间积分
TIME_DOMAIN= YES
TIME_STEP= 1e-4
MAX_TIME= 5.0                           % 需跑足够多周期后取时均
```

---

## 6. Phase 3 — Co-Kriging 多保真代理模型

### 6.1 为什么用 Co-Kriging，为什么分量独立建模

**Co-Kriging（Multi-fidelity Kriging）** 的核心方程来自 Kennedy & O'Hagan (2000)：

$$
f_{HF}(\mathbf{x}) = \rho \cdot f_{LF}(\mathbf{x}) + \delta(\mathbf{x})
$$

$\rho$ 是缩放系数，$\delta(\mathbf{x})$ 专门学习高保真与低保真之间的**系统性偏差**（例如全湍流 RANS 在转捩区高估 $C_d$ 的偏差），由此以 50 个高保真点的代价获得接近高保真的全局精度。

**为什么必须分别对 $C_l$ 和 $C_d$ 建模，不能直接拟合 $L/D$**：

升阻比 $L/D = C_l / C_d$ 在 $C_d$ 接近零时（翼型抬升失速的临界点）具有有理函数奇异性——局部区域 $C_d$ 的极小变化会引发 $L/D$ 数量级跳变。Kriging 的高斯过程假设要求响应面是"平滑连续"的；直接对 $L/D$ 建模时，这种奇异性会在极值附近引入大量伪震荡，严重污染优化搜索。

正确做法：分别训练 $\hat{C}_l(\mathbf{x})$ 和 $\hat{C}_d(\mathbf{x})$ 两个 Co-Kriging 模型，在优化目标函数中实时计算 $\hat{C}_l / \hat{C}_d$。

### 6.2 实现：SMT 库，双模型方案

```python
# train_cokriging.py
import numpy as np
import pickle
from smt.applications import MFK
from sklearn.preprocessing import MinMaxScaler

# === 加载数据（Cl 和 Cd 分开保存）===
X_lf   = np.load('data/X_lf.npy')    # (1000, 12)  [CST×10, α, Re]
Cl_lf  = np.load('data/Cl_lf.npy')   # (1000,)
Cd_lf  = np.load('data/Cd_lf.npy')   # (1000,)

X_hf   = np.load('data/X_hf.npy')    # (50, 12)
Cl_hf  = np.load('data/Cl_hf.npy')   # (50,)
Cd_hf  = np.load('data/Cd_hf.npy')   # (50,)

# Re 量级（~1e5）与 CST 系数（~0.1）相差 6 个数量级，必须归一化
scaler  = MinMaxScaler()
X_lf_n = scaler.fit_transform(X_lf)
X_hf_n = scaler.transform(X_hf)

def train_mfk(y_lf: np.ndarray, y_hf: np.ndarray, label: str) -> MFK:
    """训练一个 Co-Kriging 模型（两层：LF → HF）。"""
    sm = MFK(
        theta0      = [1e-2] * X_lf.shape[1],  # 各维相关长度初始值
        rho_regr    = 'constant',               # ρ 的回归形式
        eval_noise  = True,                     # 允许 HF 数据有少量噪声
        noise0      = [1e-6, 1e-6],             # LF/HF 噪声方差初始值
    )
    sm.set_training_values(X_lf_n, y_lf.reshape(-1, 1), name=0)  # LF 层
    sm.set_training_values(X_hf_n, y_hf.reshape(-1, 1), name=1)  # HF 层
    sm.train()
    rho = float(sm.rho)
    print(f"  [{label}] ρ = {rho:.4f}  "
          f"({'LF/HF 高度相关' if abs(rho) > 0.8 else '相关性弱，检查数据质量'})")
    return sm

print("训练 Cl Co-Kriging 模型...")
sm_Cl = train_mfk(Cl_lf, Cl_hf, 'Cl')

print("训练 Cd Co-Kriging 模型...")
sm_Cd = train_mfk(Cd_lf, Cd_hf, 'Cd')

# 序列化：分开保存，便于单独更新
with open('surrogate/cokriging_Cl.pkl', 'wb') as f:
    pickle.dump(sm_Cl, f)
with open('surrogate/cokriging_Cd.pkl', 'wb') as f:
    pickle.dump(sm_Cd, f)
with open('surrogate/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("模型已保存。")
```

### 6.3 模型质量验证（LOO-CV，分量独立评估）

```python
# validate_surrogate.py
from sklearn.model_selection import LeaveOneOut
import numpy as np, pickle

with open('surrogate/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_lf_n = scaler.transform(np.load('data/X_lf.npy'))
X_hf_n = scaler.transform(np.load('data/X_hf.npy'))
Cl_lf  = np.load('data/Cl_lf.npy')
Cd_lf  = np.load('data/Cd_lf.npy')

for label, y_lf, y_hf_path in [
    ('Cl', Cl_lf, 'data/Cl_hf.npy'),
    ('Cd', Cd_lf, 'data/Cd_hf.npy'),
]:
    y_hf = np.load(y_hf_path)
    errors = []

    for train_idx, test_idx in LeaveOneOut().split(X_hf_n):
        sm_loo = MFK(theta0=[1e-2] * X_lf_n.shape[1],
                     rho_regr='constant', eval_noise=True)
        sm_loo.set_training_values(X_lf_n, y_lf.reshape(-1,1), name=0)
        sm_loo.set_training_values(X_hf_n[train_idx],
                                   y_hf[train_idx].reshape(-1,1), name=1)
        sm_loo.train()
        pred = float(sm_loo.predict_values(X_hf_n[test_idx]))
        errors.append(pred - float(y_hf[test_idx]))

    rmse  = np.sqrt(np.mean(np.array(errors) ** 2))
    rel   = rmse / np.mean(y_hf) * 100
    print(f"LOO-CV RMSE ({label:2s}) = {rmse:.5f}  "
          f"(相对误差 {rel:.2f}%，目标 < 1.5%)")

# 额外检验：代理模型预测 L/D 与真实 HF 的散点图（见 notebooks/03_cokriging_viz.ipynb）
```

验收标准：$C_l$ 和 $C_d$ 的 LOO-CV RMSE 均 < 1.5%（相对各自高保真均值）。

---

## 7. Phase 4 — 气动优化

### 7.1 优化问题形式化

$$
\max_{\mathbf{A}, \alpha} \; \hat{f}(\mathbf{A}, \alpha, Re^*)
$$

$$
\text{s.t.} \quad
\begin{cases}
t_{\max}(\mathbf{A}) \geq 0.10 & \text{(最小厚度约束)} \\
C_l(\mathbf{A}, \alpha, Re^*) \geq 0.8 & \text{(最小升力约束)} \\
\text{geometry valid}(\mathbf{A}) & \text{(无自交，非负厚度)} \\
\mathbf{A} \in [-0.3, 0.3]^{10} & \text{(设计空间边界)} \\
\alpha \in [-2°, 12°]
\end{cases}
$$

$Re^*$ 为设计点（目标雷诺数，先固定后可参数化）。

### 7.2 优化算法

**第一轮：CMA-ES（快速全局搜索）**

CMA-ES（协方差矩阵自适应进化策略）在 10–15 维设计空间上表现优秀，不需要梯度信息：

```python
# run_opt.py
import cma
import pickle, numpy as np
from cst_params import generate_airfoil, check_geometry

with open('surrogate/cokriging_Cl.pkl', 'rb') as f:
    sm_Cl = pickle.load(f)
with open('surrogate/cokriging_Cd.pkl', 'rb') as f:
    sm_Cd = pickle.load(f)
with open('surrogate/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

RE_STAR    = 2e5    # 设计雷诺数
ALPHA_FREE = 6.0    # 设计攻角（也可纳入优化变量）
YTE_U, YTE_L = 0.002, -0.002   # 与生成网格时保持一致

def objective(A):
    """
    目标函数：最小化 -(Cl/Cd)。

    关键：Cl 和 Cd 由各自的代理模型独立预测，
    在目标函数内部实时计算比值。
    禁止用对 L/D 直接建模的代理模型预测 L/D，
    原因：Cd→0 时 L/D 的有理函数奇异性破坏 Kriging 平滑假设。
    """
    A_up, A_lo = A[:5], A[5:10]

    # 1. 几何约束（快速，不调用代理模型）
    if not check_geometry(A_up, A_lo, yte_upper=YTE_U, yte_lower=YTE_L,
                          t_min=0.10):
        return 1e6   # 几何违约，返回极大惩罚值

    # 2. 查询两个独立代理模型
    x_q = scaler.transform(
        np.array([*A, ALPHA_FREE, RE_STAR]).reshape(1, -1)
    )
    Cl_pred = float(sm_Cl.predict_values(x_q))
    Cd_pred = float(sm_Cd.predict_values(x_q))

    # 3. 气动约束（依赖代理模型）
    if Cl_pred < 0.8:      # 最小升力约束
        return 1e6
    if Cd_pred <= 1e-4:    # 防止 Cd 预测值非物理地接近零（奇异性防护）
        return 1e6

    # 4. 目标：最大化 L/D
    return -(Cl_pred / Cd_pred)

# 初始点：NACA 2412 的 CST 拟合系数
A0 = np.array([0.2273, 0.1700, 0.1920, 0.1100, 0.0972,
               -0.1145, -0.1103, -0.1315, -0.0620, -0.0662])
sigma0 = 0.05

es = cma.CMAEvolutionStrategy(A0, sigma0, {
    'bounds':   [[-0.3] * 10, [0.3] * 10],
    'maxiter':  500,
    'tolx':     1e-5,
    'popsize':  20,
    'verbose':  -9,
})
es.optimize(objective)
A_opt = es.result.xbest

# 输出验证
x_opt = scaler.transform(np.array([*A_opt, ALPHA_FREE, RE_STAR]).reshape(1,-1))
Cl_opt = float(sm_Cl.predict_values(x_opt))
Cd_opt = float(sm_Cd.predict_values(x_opt))
print(f"最优 Cl = {Cl_opt:.4f}，Cd = {Cd_opt:.5f}，"
      f"L/D = {Cl_opt/Cd_opt:.2f}（代理模型预测）")
```

**第二轮（可选）：NSGA-II 多目标**

如果同时优化多个设计雷诺数或多个攻角（最大化最小 $L/D$），使用 pymoo：

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize

class AirfoilProblem(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, n_ieq_constr=2,
                         xl=-0.3, xu=0.3)
    def _evaluate(self, X, out, *args, **kwargs):
        # 目标 1：Re=1e5 下的 -L/D；目标 2：Re=3e5 下的 -L/D
        ld1 = [predict_ld(A, alpha=6, Re=1e5) for A in X]
        ld2 = [predict_ld(A, alpha=6, Re=3e5) for A in X]
        out['F'] = np.column_stack([-np.array(ld1), -np.array(ld2)])
        # 约束：厚度 ≥ 10%，Cl ≥ 0.8
        out['G'] = np.column_stack([thickness_constraint(X),
                                    cl_constraint(X)])

res = pymoo_minimize(AirfoilProblem(), NSGA2(pop_size=50),
                     ('n_gen', 200), seed=42)
# res.X 是 Pareto 最优翼型集合
```

### 7.3 最终验证

对优化出的翼型，用高保真 CFD 重新计算，验证代理模型预测精度：

```python
# 验证：代理模型预测 vs 高保真 CFD
result_hf = run_su2_hf(A_opt, alpha=ALPHA_FREE, Re=RE_STAR)
Cl_cfd, Cd_cfd = result_hf['Cl'], result_hf['Cd']
LD_cfd = Cl_cfd / Cd_cfd

x_opt = scaler.transform(np.array([*A_opt, ALPHA_FREE, RE_STAR]).reshape(1,-1))
Cl_pred = float(sm_Cl.predict_values(x_opt))
Cd_pred = float(sm_Cd.predict_values(x_opt))
LD_pred = Cl_pred / Cd_pred

print(f"代理模型预测：Cl={Cl_pred:.4f}  Cd={Cd_pred:.5f}  L/D={LD_pred:.2f}")
print(f"高保真 CFD：  Cl={Cl_cfd:.4f}  Cd={Cd_cfd:.5f}  L/D={LD_cfd:.2f}")
print(f"L/D 相对误差：{abs(LD_pred-LD_cfd)/LD_cfd*100:.2f}%（目标 < 5%）")
```

---

## 8. 验证策略

每个阶段独立验证，防止错误向下游传播：

| 阶段 | 验证内容 | 验收标准 |
|------|----------|----------|
| Phase 0 | CST 对已知翼型（NACA 0012、NACA 2412）的重建精度 | MSE < 1e-5 |
| Phase 1（XFOIL） | 对已知翼型对比 XFOIL 文献结果 / 风洞数据 | $C_l$ 误差 < 5%，$C_d$ 误差 < 10% |
| Phase 1（RANS） | 对 Re=1e6 的 NACA 0012 对比文献 $C_d$ | $C_d$ 误差 < 8% |
| Phase 2 | 对 NACA 0012 转捩位置对比风洞实验 | 转捩 $x/c$ 误差 < 0.05 |
| Phase 3 | Co-Kriging LOO-CV | RMSE < 1.5% × 均值 |
| Phase 4 | 最优翼型 CFD 验证 | 代理模型误差 < 5% |

---

## 9. 工具链与环境配置

### 9.1 WSL 侧（主要计算环境）

```bash
# 创建独立 conda 环境
conda create -n aero-opt python=3.11 -y
conda activate aero-opt

# 基础科学栈
conda install -c conda-forge numpy scipy matplotlib pandas \
              jupyter ipykernel scikit-learn -y

# SU2（含 MPI 并行）
conda install -c conda-forge su2 openmpi mpi4py -y

# 网格工具
conda install -c conda-forge gmsh meshio -y

# 代理模型和优化
pip install smt
conda install -c conda-forge pymoo -y
pip install cma       # CMA-ES
pip install xfoil     # DARcorporation xfoil-python（内存绑定，替代 subprocess）

# XFOIL 二进制（仍需安装，xfoil-python 会调用系统 XFOIL）
sudo apt update && sudo apt install -y xfoil
```

### 9.2 环境变量配置

```bash
# 加入 ~/.bashrc
export SU2_RUN=$CONDA_PREFIX/bin
export SU2_HOME=$CONDA_PREFIX
export PATH=$SU2_RUN:$PATH
```

### 9.3 并行化策略

低保真 RANS 样本（若选 SU2 全湍流方案）并行化：

```python
# 用 joblib 并行跑 SU2（注意每个工况内部也可以 MPI 并行）
from joblib import Parallel, delayed

results = Parallel(n_jobs=4)(   # 同时跑 4 个 SU2 进程
    delayed(run_su2_lf)(X_lf[i]) for i in range(len(X_lf))
)
```

高保真工况单个任务使用 MPI 多核：

```bash
mpirun -n 8 SU2_CFD su2_transition.cfg   # 8 核跑单个高保真工况
```

---

## 10. 目录结构

```
~/aero-opt/
├── environment.yml              # conda 环境定义
├── geometry/
│   ├── cst_params.py            # CST 参数化（含钝后缘 yte 参数）
│   ├── mesh_generator.py        # Gmsh 脚本，自适应 y+ 网格生成
│   └── meshes/                  # 按哈希命名的网格缓存（含 Re 信息）
├── doe/
│   ├── doe_lhs.py               # 拉丁超立方采样
│   ├── X_lf.npy                 # 低保真采样点 (1000×12)
│   └── X_hf.npy                 # 高保真采样点 (50×12)
├── lf_solver/
│   ├── run_xfoil.py             # XFOIL（xfoil-python 内存绑定，非 subprocess）
│   ├── run_su2_lf.py            # 全湍流 RANS（备用）
│   ├── templates/
│   │   └── su2_fullturbulent.cfg
│   └── results/
│       ├── Cl_lf.npy            # 低保真 Cl (1000,)
│       └── Cd_lf.npy            # 低保真 Cd (1000,)
├── hf_solver/
│   ├── run_su2_hf.py            # 带转捩模型的 SU2
│   ├── templates/
│   │   └── su2_transition.cfg   # 含 Tu 衰减补偿和自适应 CFL
│   └── results/
│       ├── Cl_hf.npy            # 高保真 Cl (50,)
│       └── Cd_hf.npy            # 高保真 Cd (50,)
├── surrogate/
│   ├── train_cokriging.py       # 训练 Cl/Cd 双 Co-Kriging 模型
│   ├── validate_surrogate.py    # LOO-CV（Cl 和 Cd 分别验证）
│   └── models/
│       ├── cokriging_Cl.pkl     # Cl 代理模型
│       ├── cokriging_Cd.pkl     # Cd 代理模型
│       └── scaler.pkl           # 输入归一化器（共用）
├── optimization/
│   ├── run_cmaes.py             # CMA-ES（目标函数内实时计算 Cl/Cd）
│   ├── run_nsga2.py             # NSGA-II 多目标优化（可选）
│   └── results/
│       ├── optimal_airfoil.dat  # 最优翼型坐标
│       └── pareto_front.csv     # （多目标时）Pareto 前沿
└── notebooks/
    ├── 01_cst_fitting.ipynb     # CST 拟合验证（含钝后缘对比）
    ├── 02_lf_analysis.ipynb     # 低保真数据 EDA（Cl/Cd 分开展示）
    ├── 03_cokriging_viz.ipynb   # Cl/Cd 代理模型可视化 + LOO-CV 散点图
    └── 04_optimization.ipynb    # 优化结果分析
```

---

## 11. 里程碑与时间规划

| 周次 | 里程碑 | 产出 |
|------|--------|------|
| **Week 1** | Phase 0 完成 | CST 实现 + 对 NACA 系列翼型拟合验证通过 |
| **Week 2** | XFOIL 流水线打通 | 100 个样本跑通，极曲线解析正确 |
| **Week 3** | 低保真采样完成 | 1000 个 XFOIL 样本，收敛率 > 85% |
| **Week 4** | SU2 转捩模型验证 | 单个工况（NACA 0012，Re=2e5）转捩位置对比文献 |
| **Week 5** | 高保真采样完成 | 50 个 HF 样本，与 LF 系统误差可见 |
| **Week 6** | Co-Kriging 训练 | LOO-CV RMSE 达标，代理模型可视化合理 |
| **Week 7** | 优化完成 | CMA-ES 找到候选最优翼型 |
| **Week 8** | 验证 + 总结 | 高保真 CFD 验证，撰写分析报告 |

---

## 12. 已知风险与缓解措施

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| **XFOIL 进程假死**（subprocess 死循环） | ~~高~~ → **已消除** | 改用 `xfoil-python` 内存绑定，Fortran 死循环返回 NaN 而非挂起进程 |
| **SU2 尖后缘网格扭曲**（`yte=0`） | ~~高~~ → **已消除** | `cst_params.py` 默认 `yte_upper=+0.002, yte_lower=-0.002`，总后缘厚度 4‰ 弦长 |
| **代理模型直接拟合 L/D**（有理奇异性） | ~~高~~ → **已消除** | 独立训练 `sm_Cl` 和 `sm_Cd`；优化目标函数内实时计算比值 |
| **SU2 y+ 不满足**（固定 h_wall 跨 Re 失效） | ~~高~~ → **已消除** | `mesh_generator.py` 调用 `first_layer_height(Re)` 自适应计算第一层高度 |
| **SU2 湍流度衰减**（入口 Tu ≠ 前缘 Tu） | ~~中~~ → **已缓解** | 入口设 0.2% 补偿衰减；首次验证工况后实测前缘 Tu 并反推修正入口值 |
| XFOIL 在设计空间边界收敛率 < 80% | 中 | 攻角上限收窄到 10°；不收敛点补全湍流 RANS |
| SU2 $\gamma$-$Re_\theta$ 对低 Re（< 1e5）不收敛 | 中高 | CFL 从 2.0 自适应增长；Re 下限收窄到 2e5 |
| Co-Kriging $\rho$ 参数退化（LF/HF 相关性弱） | 低中 | 检查 $\rho$ 是否远离 1：若 $\|\rho\| < 0.5$，增加 HF 样本数或缩小设计空间 |
| 最优翼型 CFD 验证误差 > 5% | 低中 | 触发加点：在最优翼型附近补 5 个 HF 样本，重训 `sm_Cl` 和 `sm_Cd` |
| CST 设计空间无法覆盖真正最优形态 | 低 | 验证阶段与 NACA 6 系翼型（层流翼型）手动对比；差距 > 10% 时升阶至 N=6 |
| SU2 Student 版 512K 单元限制 | 高（如用 Fluent 验证） | 高保真计算全在 WSL 侧 SU2 运行，不受此限制 |

---

*文档版本：v2.0 | 环境：WSL-Ubuntu + conda `aero-opt` | SU2 7.x + XFOIL 6.99 + xfoil-python*

*v2.0 变更：① 钝后缘 yte 参数默认化；② XFOIL subprocess → xfoil-python；③ 自适应 y+ 网格公式；④ Co-Kriging 改为 Cl/Cd 双模型；⑤ Tu 衰减补偿说明；⑥ Ncrit ↔ Tu 一致性换算表*
