"""
按层收集激活统计信息并计算E(Y)
使用全精度模型生成100张图像，收集每一层在前50步和后50步的激活值
"""
import sys
sys.path.append(r'C:\Users\wsh\Desktop\q-diffusion-master')
import argparse
import os
import json
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from scipy import integrate
from scipy.special import erf
from pytorch_lightning import seed_everything

from ddim.models.diffusion import Model
from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import generalized_steps, compute_alpha
import yaml


def dict2namespace(config):
    """递归地将dict转换为namespace对象"""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    计算beta schedule

    Args:
        beta_schedule: beta schedule类型（'linear', 'quad', 'const', 'jsd', 'sigmoid'）
        beta_start: 起始beta值
        beta_end: 结束beta值
        num_diffusion_timesteps: 扩散时间步数

    Returns:
        betas: numpy数组
    """
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")

    assert betas.shape == (num_diffusion_timesteps,)
    return betas

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def compute_f_Y(y: float, mu: float, sigma: float, n: int) -> float:
    """
    计算概率密度函数 f_Y(y)

    公式：f_Y(y) = n * (exp(-(y-μ)^2 / (2σ^2)) / sqrt(2πσ^2)) * (1/2 + 1/2 * erf((y-μ) / sqrt(2σ^2)))^(n-1)
    """
    if sigma <= 0:
        return 0.0

    z = (y - mu) / sigma
    normal_pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
    normal_cdf = 0.5 + 0.5 * erf(z / np.sqrt(2))
    f_y = n * normal_pdf * (normal_cdf ** (n - 1))

    return float(f_y)


def compute_activation_maximum_expectation(mu: float, sigma: float, n: int = 5000) -> float:
    """
    计算n个独立同分布正态随机变量的最大值的期望 E(Y)

    E[Y] = ∫_{-∞}^{∞} y * f_Y(y) dy
    """
    if sigma <= 0:
        return mu

    def integrand(y):
        return y * compute_f_Y(y, mu, sigma, n)

    lower = max(0, mu - 5 * sigma)
    upper = mu + 10 * sigma

    try:
        result, error = integrate.quad(
            integrand,
            lower,
            upper,
            limit=1000,
            epsabs=1e-6,
            epsrel=1e-6
        )

        E_Y = result

        if np.isnan(E_Y) or np.isinf(E_Y) or E_Y < mu:
            logger.warning(f"数值积分结果异常: E_Y={E_Y}, 使用近似公式")
            E_Y = mu + sigma * np.sqrt(2 * np.log(n))

    except Exception as e:
        logger.warning(f"数值积分失败: {e}，使用近似公式")
        E_Y = mu + sigma * np.sqrt(2 * np.log(n))

    return float(max(E_Y, mu))


def collect_layerwise_activations_during_sampling(
    model: nn.Module,
    betas: torch.Tensor,
    num_images: int = 100,
    num_timesteps: int = 100,
    total_timesteps: int = 1000,
    device: torch.device = None,
    batch_size: int = 1,
    skip_type: str = "uniform"
):
    """
    在采样过程中收集每一层的激活值

    Args:
        model: 全精度模型
        betas: beta schedule张量
        num_images: 生成的图像数量（默认100）
        num_timesteps: 采样步数（默认100）
        total_timesteps: 总时间步数（默认1000）
        device: 设备
        batch_size: 批大小
        skip_type: 采样类型（"uniform" 或 "quad"）

    Returns:
        layer_activations: 字典 {layer_name: {group_id: [激活值列表]}}
        layer_max_activations: 字典 {layer_name: {group_id: [每个时间步的最大值列表]}}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # 按层收集激活值
    # layer_activations[layer_name][group_id] = [激活值列表]
    layer_activations = defaultdict(lambda: defaultdict(list))

    # 按层收集每个时间步的最大激活值
    # layer_max_activations[layer_name][group_id] = [每个时间步的最大值列表]
    layer_max_activations = defaultdict(lambda: defaultdict(list))

    # 当前组ID（全局变量，用于hook）
    current_group_id = -1
    current_timestep = -1  # 当前时间步，用于标记最大值属于哪个时间步

    # 注册hook收集每一层的激活值
    hooks = []
    layer_names = []

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                # 只收集属于目标组的激活值
                if current_group_id < 0:
                    return

                # 收集激活值的绝对值
                act_abs = output.abs()

                # 1. 收集激活值（用于计算μ和σ）
                # 限制收集数量以避免内存溢出
                # 策略：小张量全部收集，大张量采样固定数量
                max_samples_per_timestep = 500  # 每个时间步每层最多收集500个值

                if act_abs.numel() <= max_samples_per_timestep:
                    # 小张量：全部收集
                    act_values = act_abs.flatten().cpu().numpy().tolist()
                else:
                    # 大张量：均匀采样
                    act_flat = act_abs.flatten()
                    step = act_flat.numel() // max_samples_per_timestep
                    indices = torch.arange(0, act_flat.numel(), step, device=act_flat.device)[:max_samples_per_timestep]
                    act_values = act_flat[indices].cpu().numpy().tolist()

                # 检查并限制每个层每组收集的总数量，防止内存溢出
                max_total_samples_per_layer_group = 10000  # 每个层每组最多10000个值
                current_count = len(layer_activations[layer_name][current_group_id])

                if current_count < max_total_samples_per_layer_group:
                    # 如果还没达到上限，继续收集
                    remaining = max_total_samples_per_layer_group - current_count
                    if remaining > 0:
                        layer_activations[layer_name][current_group_id].extend(act_values[:remaining])

                # 2. 收集当前时间步的最大激活值
                max_value = act_abs.max().item()
                layer_max_activations[layer_name][current_group_id].append(max_value)
        return hook_fn

    # 注册所有层的hook
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            layer_names.append(name)
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

    logger.info(f"注册了 {len(layer_names)} 层的hook")

    # 定义修改后的采样函数，在采样过程中收集激活值
    def generalized_steps_with_hooks(x, seq, model, b, **kwargs):
        """修改后的采样函数，收集激活值"""
        nonlocal current_group_id

        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]

            # seq是倒序的，例如[999, 989, 979, ..., 9, 0] (uniform)
            # 或 [800, 783, 752, ..., 1, 0] (quad)
            # reversed(seq)是正序的，例如[0, 9, 19, ..., 989, 999] (uniform)
            # 或 [0, 1, ..., 752, 783, 800] (quad)
            # step_idx从0开始
            # 注意：reversed(seq)的前50个元素对应前50步（组0），后50个元素对应后50步（组1）
            reversed_seq = list(reversed(seq))
            for step_idx, (i, j) in enumerate(zip(reversed_seq, reversed(seq_next))):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to(device)

                timestep_val = int(i)

                # 根据采样序列的位置判断组
                # 前50步（组0）：step_idx 0-49，对应序列的前半部分
                # 后50步（组1）：step_idx 50-99，对应序列的后半部分
                total_steps = len(seq)
                half_steps = total_steps // 2

                if step_idx < half_steps:  # 前50步（组0）
                    current_group_id = 0
                    current_timestep = timestep_val
                elif step_idx < total_steps:  # 后50步（组1）
                    current_group_id = 1
                    current_timestep = timestep_val
                else:
                    # 不应该到达这里
                    current_group_id = -1
                    current_timestep = -1
                    et = model(xt, t)
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    x0_preds.append(x0_t.to('cpu'))
                    c1 = (
                        kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    )
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                    xs.append(xt_next.to('cpu'))
                    continue

                # 前向传播（会触发hooks，自动收集到对应组）
                et = model(xt, t)

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t.to('cpu'))
                c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))

            return xs, x0_preds

    # 生成图像并收集激活值
    logger.info(f"开始生成 {num_images} 张图像并收集激活值")
    logger.info(f"采样类型: {skip_type}")

    # 计算采样序列
    if skip_type == "uniform":
        skip = total_timesteps // num_timesteps
        seq = list(range(0, total_timesteps, skip))
    elif skip_type == "quad":
        # Quadratic schedule: 生成非线性序列，早期步数密集，后期稀疏
        seq = (
            np.linspace(
                0, np.sqrt(total_timesteps * 0.8), num_timesteps
            )
            ** 2
        )
        seq = [int(s) for s in list(seq)]
        seq = sorted(list(set(seq)))  # 去重并排序
    else:
        raise ValueError(f"不支持的skip_type: {skip_type}，支持 'uniform' 或 'quad'")

    logger.info(f"采样序列长度: {len(seq)}, 序列范围: [{min(seq)}, {max(seq)}]")

    # 计算分组信息
    reversed_seq = list(reversed(seq))
    half_steps = len(seq) // 2
    group0_timesteps = reversed_seq[:half_steps]
    group1_timesteps = reversed_seq[half_steps:]

    logger.info(f"前{half_steps}步（组0）: timestep范围 [{min(group0_timesteps)}, {max(group0_timesteps)}]")
    logger.info(f"后{len(seq)-half_steps}步（组1）: timestep范围 [{min(group1_timesteps)}, {max(group1_timesteps)}]")

    for img_idx in tqdm(range(num_images), desc="生成图像"):
        # 初始化噪声
        shape = (batch_size, 3, 32, 32)  # CIFAR10
        x = torch.randn(shape, device=device)

        # 进行采样（使用修改后的函数）
        xs, _ = generalized_steps_with_hooks(
            x, seq, model, betas, eta=0.0
        )

        # 每处理10张图像后清理GPU缓存和Python缓存，防止内存溢出
        if (img_idx + 1) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        if (img_idx + 1) % 10 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 移除hooks
    for hook in hooks:
        hook.remove()

    logger.info(f"收集完成")

    # 打印统计信息
    for layer_name in layer_names[:5]:  # 只打印前5层
        for group_id in [0, 1]:
            count = len(layer_activations[layer_name][group_id])
            max_count = len(layer_max_activations[layer_name][group_id])
            logger.info(f"层 {layer_name}, 组{group_id}: 收集了 {count} 个激活值, {max_count} 个时间步最大值")

    return layer_activations, layer_max_activations


def compute_layerwise_ey_statistics(layer_activations, layer_max_activations, n: int = 5000):
    """
    对每一层计算前50步和后50步的E(Y)统计信息，以及最大值平均值

    Args:
        layer_activations: 字典 {layer_name: {group_id: [激活值列表]}}
        layer_max_activations: 字典 {layer_name: {group_id: [每个时间步的最大值列表]}}
        n: 样本数量（默认5000）

    Returns:
        statistics: 字典 {layer_name: {group_id: {
            'mu': float, 'sigma': float, 'E_Y': float,
            'max_avg': float, 'max_samples': int,
            'diff': float,
            'samples': int
        }}}
    """
    statistics = {}

    for layer_name, groups in layer_activations.items():
        statistics[layer_name] = {}

        for group_id in [0, 1]:  # 前50步(0)和后50步(1)
            activations = groups.get(group_id, [])
            max_activations = layer_max_activations.get(layer_name, {}).get(group_id, [])

            stat_entry = {}

            # 计算μ, σ和E(Y)
            if len(activations) >= 10:
                mu = np.mean(activations)
                sigma = np.std(activations, ddof=1)

                # 计算E(Y)
                E_Y = compute_activation_maximum_expectation(mu, sigma, n=min(len(activations), n))

                stat_entry['mu'] = float(mu)
                stat_entry['sigma'] = float(sigma)
                stat_entry['E_Y'] = float(E_Y)
                stat_entry['samples'] = len(activations)
            else:
                logger.warning(f"层 {layer_name}, 组{group_id}: 激活值样本数不足（{len(activations)} < 10）")
                stat_entry['mu'] = 0.0
                stat_entry['sigma'] = 0.0
                stat_entry['E_Y'] = 0.0
                stat_entry['samples'] = len(activations)

            # 计算最大值平均值
            if len(max_activations) > 0:
                max_avg = np.mean(max_activations)
                stat_entry['max_avg'] = float(max_avg)
                stat_entry['max_samples'] = len(max_activations)

                # 计算差异
                diff = max_avg - stat_entry['E_Y']
                stat_entry['diff'] = float(diff)

                logger.info(f"层 {layer_name}, 组{group_id}: "
                          f"μ={stat_entry['mu']:.4f}, σ={stat_entry['sigma']:.4f}, "
                          f"E(Y)={stat_entry['E_Y']:.4f}, "
                          f"最大值平均值={max_avg:.4f}, "
                          f"差异={stat_entry['diff']:.4f}, "
                          f"样本数={stat_entry['samples']}, 最大样本数={stat_entry['max_samples']}")
            else:
                logger.warning(f"层 {layer_name}, 组{group_id}: 最大值样本数不足（{len(max_activations)} = 0）")
                stat_entry['max_avg'] = 0.0
                stat_entry['max_samples'] = 0
                stat_entry['diff'] = 0.0

            statistics[layer_name][group_id] = stat_entry

    return statistics


def save_statistics_to_json(statistics, output_path: str):
    """
    将统计信息保存为JSON文件
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)

    logger.info(f"统计信息已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--num_images", type=int, default=100, help="生成的图像数量")
    parser.add_argument("--num_timesteps", type=int, default=100, help="采样步数")
    parser.add_argument("--total_timesteps", type=int, default=1000, help="总时间步数")
    parser.add_argument("--batch_size", type=int, default=1, help="批大小")
    parser.add_argument("--output_json", type=str, default="layerwise_activation_statistics.json", help="输出JSON文件路径")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--split", action="store_true", help="使用split策略进行跳层连接量化")
    parser.add_argument("--skip_type", type=str, default="uniform", choices=["uniform", "quad"],
                       help="采样类型：uniform（均匀）或quad（二次）")

    args = parser.parse_args()

    # 设置随机种子
    seed_everything(args.seed)

    # 加载配置
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    # 设置split_shortcut属性（严格参照sample_diffusion_ddim.py）
    config.split_shortcut = args.split

    # 创建模型
    model = Model(config)

    # 加载预训练权重
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    else:
        raise ValueError(f"不支持的数据集: {config.data.dataset}")

    ckpt = get_ckpt_path(f"ema_{name}")
    logger.info(f"加载检查点: {ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    # 计算betas
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)

    # 收集激活值
    logger.info("=" * 75)
    logger.info("开始收集每一层的激活统计信息")
    logger.info(f"生成 {args.num_images} 张图像")
    logger.info(f"采样步数: {args.num_timesteps}")
    logger.info(f"采样类型: {args.skip_type}")
    logger.info(f"每层每组收集约 {args.num_timesteps // 2}*{args.num_images} 个激活值")
    logger.info("=" * 75)

    layer_activations, layer_max_activations = collect_layerwise_activations_during_sampling(
        model=model,
        betas=betas,
        num_images=args.num_images,
        num_timesteps=args.num_timesteps,
        total_timesteps=args.total_timesteps,
        device=device,
        batch_size=args.batch_size,
        skip_type=args.skip_type
    )

    # 计算统计信息
    logger.info("\n" + "=" * 75)
    logger.info("计算每一层的E(Y)统计信息和最大值平均值 (n=5000)")
    logger.info("=" * 75)

    statistics = compute_layerwise_ey_statistics(layer_activations, layer_max_activations, n=5000)

    # 保存为JSON
    save_statistics_to_json(statistics, args.output_json)

    logger.info("\n" + "=" * 75)
    logger.info("完成！")
    logger.info(f"JSON文件已保存: {args.output_json}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()

