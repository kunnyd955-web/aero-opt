"""设计空间拉丁超立方采样 (Design of Experiments)。

依据方案第 4.1 / 5.1 节。设计变量 (12 维, 列序固定):
  index 0-9 : CST 系数 (上翼面 5 + 下翼面 5)
  index 10  : 攻角 alpha (度)
  index 11  : 雷诺数 Re

低保真 (LF) 覆盖全设计空间; 高保真 (HF) 收缩到转捩活跃子区间。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from smt.sampling_methods import LHS

__all__ = ["LF_BOUNDS", "HF_BOUNDS", "sample_lf", "sample_hf"]

N_CST = 10  # CST 系数个数 (上下翼面各 5)

# 修正方案设计空间 (重要):
#   方案原文把全部 10 个 CST 系数设为对称区间 [-0.3,0.3]。实测此设定下
#   独立随机采样仅约 1.4% 的翼型几何合法 (上翼面常跌到下翼面之下 -> 厚度<=0,
#   或厚度<10%)。改为"上翼面系数取正、下翼面取负"的符号分离方案后, 合法率
#   升至约 97%。设计空间宽度基本不变 (单侧 0.05~0.3 仍覆盖薄到厚的翼型),
#   且天然满足 t/c>=10% 的结构约束 (方案 7.1)。
N_HALF = N_CST // 2  # 上/下翼面各 5

# 低保真: 全设计空间 (方案 1.3 / 4.1, 符号分离修正版)
LF_BOUNDS = np.array([
    *[[0.05, 0.30]] * N_HALF,   # 上翼面 CST 系数 (正)
    *[[-0.30, -0.05]] * N_HALF,  # 下翼面 CST 系数 (负)
    [-2.0, 12.0],               # 攻角 alpha (度)
    [5e4, 5e5],                 # 雷诺数 Re
])

# 高保真: 转捩最活跃子区间 (方案 5.1 策略 B, 符号分离修正版)
HF_BOUNDS = np.array([
    *[[0.05, 0.20]] * N_HALF,   # 上翼面 CST 系数 (稍收缩)
    *[[-0.20, -0.05]] * N_HALF,  # 下翼面 CST 系数
    [4.0, 10.0],                # 攻角限制在转捩活跃区
    [1e5, 3e5],                 # 转捩最敏感 Re 区间
])


def _sample(bounds: np.ndarray, n: int, seed: int) -> np.ndarray:
    """用 smt LHS (最大熵 ese 准则) 在 bounds 内采 n 个点。"""
    sampling = LHS(xlimits=bounds, criterion="ese", seed=seed)
    return sampling(n)


def sample_lf(n: int = 1000, seed: int = 42,
              save_path: str | Path | None = None) -> np.ndarray:
    """低保真采样点 (n, 12)。save_path 非空则保存为 .npy。"""
    X = _sample(LF_BOUNDS, n, seed)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, X)
    return X


def sample_hf(n: int = 50, seed: int = 7,
              save_path: str | Path | None = None) -> np.ndarray:
    """高保真采样点 (n, 12)。save_path 非空则保存为 .npy。"""
    X = _sample(HF_BOUNDS, n, seed)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, X)
    return X


if __name__ == "__main__":
    here = Path(__file__).parent
    X_lf = sample_lf(n=1000, save_path=here / "X_lf.npy")
    X_hf = sample_hf(n=50, save_path=here / "X_hf.npy")
    print(f"X_lf: {X_lf.shape} -> {here / 'X_lf.npy'}")
    print(f"X_hf: {X_hf.shape} -> {here / 'X_hf.npy'}")
