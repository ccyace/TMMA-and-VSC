"""
激活统计收集和期望值计算模块
用于基于统计的激活量化截断
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import defaultdict
from scipy import integrate
from scipy.special import erf
import math

logger = logging.getLogger(__name__)


def collect_activation_statistics_by_group(
    model, cali_data: Tuple[torch.Tensor, torch.Tensor],
    num_samples_per_timestep: int = 100,
    num_groups: int = 2,
    total_timesteps: int = 1000,
    device: torch.device = None
) -> Dict[int, Dict[str, float]]:
    """
    按组收集所有时间步的激活统计信息（均值μ和方差σ）
    对每个组，收集组内所有时间步的激活值，然后整体计算统计信息

    Args:
        model: 量化模型
        cali_data: 校准数据 (xs, ts)
        num_samples_per_timestep: 每个时间步收集的样本数（默认100）
        num_groups: 分组数量（默认2）
        total_timesteps: 总时间步数（默认1000）
        device: 设备

    Returns:
        statistics: 字典 {group_id: {'mu': float, 'sigma': float, 'samples': int}}
    """
    if device is None:
        device = next(model.parameters()).device

    cali_xs, cali_ts = cali_data
    timesteps_per_group = total_timesteps // num_groups

    logger.info(f"开始收集激活统计信息（{num_groups}组，每组时间步范围{timesteps_per_group}）")
    logger.info(f"每个时间步收集{num_samples_per_timestep}个样本")
    logger.info("按组收集：组内所有时间步的激活值将合并计算统计信息")

    # 按组收集所有激活值（不区分时间步）
    activations_by_group = defaultdict(list)

    # 收集每个时间步的样本索引
    timestep_indices = defaultdict(list)
    for i in range(len(cali_ts)):
        ts = cali_ts[i].item() if torch.is_tensor(cali_ts[i]) else cali_ts[i]
        timestep_indices[ts].append(i)

    logger.info(f"校准数据中的唯一时间步数量: {len(timestep_indices)}")

    model.eval()
    with torch.no_grad():
        from tqdm import tqdm

        # 对每个时间步收集激活
        for ts, indices in tqdm(timestep_indices.items(), desc="收集激活统计"):
            group_id = min(int(ts / timesteps_per_group), num_groups - 1)

            # 限制每个时间步的样本数
            sample_indices = indices[:num_samples_per_timestep]

            # 收集该时间步的激活值
            for idx in sample_indices:
                x = cali_xs[idx:idx+1].to(device)
                t = torch.tensor([ts], device=device)

                # 使用hook收集所有激活层的输出
                # 收集实际的激活值（不是最大值），用于计算μ和σ
                # 然后通过公式计算n个激活值的最大值的期望E(Y)
                sample_activations = []

                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        # 收集实际的激活值（绝对值）
                        # 由于激活值可能很大（例如 [batch, channels, H, W]），
                        # 我们采样一些激活值以避免内存问题
                        act_abs = output.abs()

                        # 方法1: 如果是小张量，收集所有值
                        if act_abs.numel() <= 10000:  # 少于10000个元素
                            act_values = act_abs.flatten().cpu().numpy().tolist()
                            sample_activations.extend(act_values)
                        else:
                            # 方法2: 对于大张量，随机采样一些激活值
                            # 或者使用统计方法：收集均值、最大值等代表性值
                            # 这里我们采样最多1000个值
                            act_flat = act_abs.flatten()
                            num_samples = min(1000, act_flat.numel())
                            if num_samples == act_flat.numel():
                                # 如果总数小于1000，收集所有值
                                act_values = act_flat.cpu().numpy().tolist()
                            else:
                                # 随机采样
                                indices = torch.randperm(act_flat.numel(), device=act_flat.device)[:num_samples]
                                act_values = act_flat[indices].cpu().numpy().tolist()
                            sample_activations.extend(act_values)

                hooks = []
                for name, module in model.named_modules():
                    # 收集激活量化层的输出
                    if hasattr(module, 'act_quantizer') or hasattr(module, 'act_quantizer_w'):
                        hooks.append(module.register_forward_hook(hook_fn))

                try:
                    _ = model(x, t)
                except Exception as e:
                    logger.warning(f"处理时间步{ts}样本{idx}时出错: {e}")
                    continue

                # 移除hooks
                for hook in hooks:
                    hook.remove()

                # 收集该样本的所有激活值到组中
                # 这些激活值将用于计算该组的μ和σ
                # 然后用公式计算n个激活值的最大值的期望E(Y)
                if sample_activations:
                    activations_by_group[group_id].extend(sample_activations)

    # 计算每个组的统计信息（均值μ和方差σ）
    # 组内所有时间步的激活值合并计算
    statistics = {}
    for group_id in range(num_groups):
        activations = activations_by_group[group_id]

        if len(activations) >= 10:  # 至少需要10个样本
            mu = np.mean(activations)
            sigma = np.std(activations, ddof=1)  # 使用样本标准差
            min_act = np.min(activations)
            max_act = np.max(activations)
            median_act = np.median(activations)
            statistics[group_id] = {
                'mu': float(mu),
                'sigma': float(sigma),
                'samples': len(activations)
            }
            logger.info(f"组{group_id}: μ={mu:.4f}, σ={sigma:.4f}, min={min_act:.4f}, max={max_act:.4f}, median={median_act:.4f}, 总样本数={len(activations)}")
        else:
            logger.warning(f"组{group_id}: 样本数不足（{len(activations)} < 10），跳过")
            statistics[group_id] = {
                'mu': 1.0,  # 默认值
                'sigma': 0.5,  # 默认值
                'samples': len(activations)
            }

    logger.info(f"收集完成：共{len(statistics)}个组的统计信息")
    return statistics


def compute_f_Y(y: float, mu: float, sigma: float, n: int) -> float:
    """
    计算概率密度函数 f_Y(y)

    公式：f_Y(y) = n * (exp(-(y-μ)^2 / (2σ^2)) / sqrt(2πσ^2)) * (1/2 + 1/2 * erf((y-μ) / sqrt(2σ^2)))^(n-1)

    Args:
        y: 变量值
        mu: 均值μ
        sigma: 标准差σ
        n: 样本数量

    Returns:
        f_Y(y): 概率密度函数值
    """
    if sigma <= 0:
        return 0.0

    # 标准化
    z = (y - mu) / sigma

    # 计算正态分布PDF: exp(-(y-μ)^2 / (2σ^2)) / sqrt(2πσ^2)
    normal_pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi * sigma ** 2)

    # 计算CDF: 1/2 + 1/2 * erf((y-μ) / sqrt(2σ^2))
    # 注意：erf(x) 是标准误差函数，标准正态分布的CDF = 1/2 * (1 + erf(x/sqrt(2)))
    # 这里使用 erf((y-μ) / sqrt(2σ^2)) = erf(z / sqrt(2))
    normal_cdf = 0.5 + 0.5 * erf(z / np.sqrt(2))

    # 计算 f_Y(y) = n * PDF * CDF^(n-1)
    f_y = n * normal_pdf * (normal_cdf ** (n - 1))

    return float(f_y)


def compute_activation_maximum_expectation(
    mu: float, sigma: float, n: int = 100
) -> float:
    """
    计算n个独立同分布正态随机变量的最大值的期望 E(Y)

    使用数值积分计算：
    E[Y] = ∫_{-∞}^{∞} y * f_Y(y) dy

    其中 f_Y(y) = n * (exp(-(y-μ)^2 / (2σ^2)) / sqrt(2πσ^2)) * (1/2 + 1/2 * erf((y-μ) / sqrt(2σ^2)))^(n-1)

    Args:
        mu: 均值μ
        sigma: 标准差σ
        n: 样本数量（默认100）

    Returns:
        E(Y): 最大值期望
    """
    if sigma <= 0:
        return mu

    # 定义被积函数：y * f_Y(y)
    def integrand(y):
        """被积函数：y * f_Y(y)"""
        return y * compute_f_Y(y, mu, sigma, n)

    # 确定积分范围
    # 由于是正态分布，主要概率质量在 [μ - 5σ, μ + 5σ] 范围内
    # 但由于是最大值分布，可能向右偏，所以上界需要更大
    lower = mu - 5 * sigma
    upper = mu + 10 * sigma  # 使用更大的上界以覆盖最大值分布的尾部

    # 确保下界非负（激活值通常非负）
    if lower < 0:
        lower = max(0, mu - 5 * sigma)

    try:
        # 使用数值积分计算 E[Y] = ∫ y * f_Y(y) dy
        # 使用 scipy.integrate.quad 进行自适应数值积分
        result, error = integrate.quad(
            integrand,
            lower,
            upper,
            limit=1000,  # 增加积分点数量以提高精度
            epsabs=1e-6,
            epsrel=1e-6
        )

        E_Y = result

        # 验证结果合理性
        if np.isnan(E_Y) or np.isinf(E_Y) or E_Y < mu:
            logger.warning(f"数值积分结果异常: E_Y={E_Y}, 使用近似公式")
            # 使用近似公式：E[Y] ≈ μ + σ * √(2*log(n))
            E_Y = mu + sigma * np.sqrt(2 * np.log(n))

        logger.debug(f"计算E[Y]: μ={mu:.4f}, σ={sigma:.4f}, n={n}, E[Y]={E_Y:.4f}, 积分误差={error:.2e}")

    except Exception as e:
        logger.warning(f"数值积分失败: {e}，使用近似公式")
        # 备用方案：使用近似公式
        # 对于正态分布的最大值，E[Y] ≈ μ + σ * √(2*log(n))
        E_Y = mu + sigma * np.sqrt(2 * np.log(n))

    return float(max(E_Y, mu))  # 确保E(Y) >= μ


def compute_group_truncation_values(
    statistics: Dict[int, Dict[str, float]],
    num_groups: int = 2,
    n: int = 100
) -> Dict[int, float]:
    """
    为每个组计算截断值（基于统计的E(Y)）

    对每个组，使用组内所有时间步的激活统计（μ和σ）计算 E(Y)

    Args:
        statistics: 统计信息字典 {group_id: {'mu': float, 'sigma': float, 'samples': int}}
        num_groups: 分组数量
        n: 样本数量（默认100，用于计算E(Y)）

    Returns:
        truncation_values: 字典 {group_id: E(Y)}
    """
    truncation_values = {}

    for group_id in range(num_groups):
        group_stats = statistics.get(group_id)

        if not group_stats:
            logger.warning(f"组{group_id}没有统计信息，使用默认值")
            truncation_values[group_id] = 1.0
            continue

        # 使用该组的整体统计信息（μ和σ）计算E(Y)
        mu = group_stats['mu']
        sigma = group_stats['sigma']
        num_samples = group_stats.get('samples', n)

        # 使用实际收集的样本数量n来计算E(Y)
        # 如果收集的样本数足够多，使用实际样本数；否则使用默认值100
        effective_n = min(num_samples, n) if num_samples >= 10 else n

        E_Y = compute_activation_maximum_expectation(mu, sigma, effective_n)
        truncation_values[group_id] = E_Y

        # 计算近似公式的值用于对比
        approx_e_y = mu + sigma * np.sqrt(2 * np.log(effective_n))
        logger.info(f"组{group_id}截断值: E(Y)={E_Y:.4f} (基于μ={mu:.4f}, σ={sigma:.4f}, n={effective_n}, 总样本数={num_samples})")
        logger.info(f"  近似公式: E(Y)≈{approx_e_y:.4f} (μ + σ*√(2*ln(n)))")
        logger.info(f"  说明: E(Y)是n={effective_n}个激活值的最大值的期望，会向右偏移")

    return truncation_values

