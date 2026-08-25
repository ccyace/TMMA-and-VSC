import argparse, os, gc, glob, datetime, yaml
import json
import logging
import math
import platform
import sys
sys.path.append(r'C:\Users\wsh\Desktop\q-diffusion-master')
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
from pytorch_lightning import seed_everything

from ddim.models.diffusion import Model
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import generalized_steps

import torchvision.utils as tvu

from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock,
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples

logger = logging.getLogger(__name__)


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
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
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def load_max_avg_statistics(json_path: str) -> dict:
    """
    从JSON文件加载max_avg统计信息（用于区分组0和组1，但不用于截断）

    Args:
        json_path: JSON文件路径

    Returns:
        max_avg_stats: 字典 {layer_name: {0: max_avg_group0, 1: max_avg_group1}}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        statistics = json.load(f)

    # 从layers字段读取
    layers = statistics.get("layers", {})
    if not layers:
        # 如果没有layers字段，可能直接就是layers字典
        layers = statistics

    max_avg_stats = {}
    for layer_name, groups in layers.items():
        if isinstance(groups, dict):
            max_avg_stats[layer_name] = {
                0: groups.get('0', {}).get('max_avg', None),
                1: groups.get('1', {}).get('max_avg', None)
            }

    logger.info(f"从 {json_path} 加载了 {len(max_avg_stats)} 层的max_avg统计信息（仅用于分组，不进行截断）")
    return max_avg_stats


def normalize_cali_sample_data(sample_data, args):
    """
    兼容不同校准集存储格式，避免将 5D 张量直接送入 Conv2d。

    支持：
    - 4D Tensor: [N, C, H, W]（原样返回）
    - 5D Tensor: [T, N, C, H, W]（按 cali_t_index 选一个时间步，返回 [N, C, H, W]）
    - 其他类型（如 dict）：交给 get_train_samples 处理
    """
    if not torch.is_tensor(sample_data):
        return sample_data

    if sample_data.dim() == 4:
        return sample_data

    if sample_data.dim() == 5:
        t_total = sample_data.shape[0]
        t_idx = max(0, min(int(args.cali_t_index), t_total - 1))
        if t_idx != int(args.cali_t_index):
            logger.warning(
                f"cali_t_index={args.cali_t_index} 越界，已自动裁剪为 {t_idx}（总步数={t_total}）"
            )
        logger.info(
            f"检测到5D校准集 [T,N,C,H,W]={tuple(sample_data.shape)}，"
            f"使用第 {t_idx} 个时间步作为校准数据"
        )
        return sample_data[t_idx]

    raise ValueError(
        f"不支持的校准集Tensor维度: dim={sample_data.dim()}, shape={tuple(sample_data.shape)}"
    )


def brecq_quantization(model: nn.Module, qnn: nn.Module, cali_data: tuple,
                       max_avg_stats: dict, group_id: int, args):
    """
    执行BRECQ量化重建（权重+激活），不进行激活截断

    Args:
        model: 原始全精度模型
        qnn: 量化模型
        cali_data: 校准数据 (xs, ts)
        max_avg_stats: max_avg统计信息（仅用于标识组，不用于截断）
        group_id: 组ID (0 或 1)
        args: 命令行参数
    """
    cali_xs, cali_ts = cali_data

    # 1. 权重量化重建
    if not args.skip_weight_recon and not args.resume_w:
        logger.info("=" * 75)
        logger.info(f"阶段{group_id + 1}: 开始权重量化重建")
        logger.info("=" * 75)

        logger.info("初始化权重量化参数")
        qnn.set_quant_state(True, False)  # 启用权重量化，禁用激活量化
        device = next(qnn.parameters()).device
        _ = qnn(cali_xs[:8].to(device), cali_ts[:8].to(device))
        logger.info("初始化完成!")

        # 权重重建参数
        kwargs = dict(
            cali_data=cali_data,
            batch_size=args.cali_batch_size,
            iters=args.cali_iters,
            weight=0.01,
            asym=True,
            b_range=(20, 2),
            warmup=0.2,
            act_quant=False,
            opt_mode='mse'
        )

        def recon_model_weight(model):
            for name, module in model.named_children():
                if isinstance(module, QuantModule):
                    if module.ignore_reconstruction:
                        continue
                    layer_reconstruction(qnn, module, **kwargs)
                elif isinstance(module, BaseQuantBlock):
                    if module.ignore_reconstruction:
                        continue
                    block_reconstruction(qnn, module, **kwargs)
                else:
                    recon_model_weight(module)

        logger.info("执行权重量化重建（BRECQ）")
        recon_model_weight(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=False)
        logger.info("权重量化重建完成")
    elif args.skip_weight_recon:
        logger.info("跳过权重重建")
        qnn.set_quant_state(True, False)
        device = next(qnn.parameters()).device
        _ = qnn(cali_xs[:8].to(device), cali_ts[:8].to(device))
        qnn.set_quant_state(weight_quant=True, act_quant=False)

    # 2. 激活量化重建（不进行截断）
    if args.quant_act:
        logger.info("=" * 75)
        logger.info(f"阶段{group_id + 1}: 开始激活量化重建（不进行激活截断）")
        logger.info("=" * 75)

        logger.info("初始化激活量化参数")
        qnn.set_quant_state(True, True)

        # 初始化激活量化参数
        device = next(qnn.parameters()).device
        with torch.no_grad():
            num_init_batches = min(4, int(np.ceil(cali_xs.shape[0] / 64)))
            for i in range(num_init_batches):
                start_idx = i * 64
                end_idx = min((i + 1) * 64, cali_xs.shape[0])
                batch_xs = cali_xs[start_idx:end_idx].to(device)
                batch_ts = cali_ts[start_idx:end_idx].to(device)
                _ = qnn(batch_xs, batch_ts)

        if args.running_stat:
            logger.info('使用运行统计更新激活量化参数')
            qnn.set_running_stat(True)
            num_batches = int(np.ceil(cali_xs.size(0) / 64))
            for i in tqdm(range(num_batches), desc="更新运行统计"):
                start_idx = i * 64
                end_idx = min((i + 1) * 64, cali_xs.size(0))
                batch_xs = cali_xs[start_idx:end_idx].to(device)
                batch_ts = cali_ts[start_idx:end_idx].to(device)
                _ = qnn(batch_xs, batch_ts)
            qnn.set_running_stat(False)
            logger.info('运行统计更新完成')

        # 激活重建参数
        kwargs = dict(
            cali_data=cali_data,
            iters=args.cali_iters_a,
            act_quant=True,
            opt_mode='mse',
            lr=args.cali_lr,
            p=args.cali_p
        )

        def recon_model_act(model):
            for name, module in model.named_children():
                if isinstance(module, QuantModule):
                    if module.ignore_reconstruction:
                        continue
                    layer_reconstruction(qnn, module, **kwargs)
                elif isinstance(module, BaseQuantBlock):
                    if module.ignore_reconstruction:
                        continue
                    block_reconstruction(qnn, module, **kwargs)
                else:
                    recon_model_act(module)

        logger.info("执行激活量化重建（不进行激活截断）")
        recon_model_act(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)

    logger.info(f"阶段{group_id + 1}: BRECQ量化重建完成（未进行激活截断）")


def first_half_sampling(qnn: nn.Module, num_samples: int, betas: torch.Tensor,
                        num_timesteps: int, total_timesteps: int, device: torch.device,
                        config, batch_size: int = 50, save_dir: str = None,
                        skip_type: str = "uniform", eta: float = 0.0):
    """
    前半段采样：从噪声生成中间噪声图像

    Args:
        qnn: 量化模型
        num_samples: 生成样本数
        betas: beta schedule
        num_timesteps: 采样步数
        total_timesteps: 总时间步数
        device: 设备
        config: 配置
        batch_size: 批大小
        save_dir: 保存目录
        skip_type: 采样类型 (uniform/quad)
        eta: eta参数

    Returns:
        intermediate_images: 中间噪声图像
    """
    logger.info("=" * 75)
    logger.info(f"前半段采样：生成 {num_samples} 张中间噪声图像")
    logger.info("=" * 75)

    qnn.eval()

    # 计算采样序列
    if skip_type == "uniform":
        skip = total_timesteps // num_timesteps
        seq = list(range(0, total_timesteps, skip))
    elif skip_type == "quad":
        seq = (
                np.linspace(0, np.sqrt(total_timesteps * 0.8), num_timesteps) ** 2
        )
        seq = [int(s) for s in list(seq)]
        seq = sorted(list(set(seq)))
    else:
        raise ValueError(f"不支持的skip_type: {skip_type}")

    # 只使用前半段序列（从高噪声到低噪声，反向序列的前一半）
    reversed_seq = list(reversed(seq))
    half_steps = len(reversed_seq) // 2
    half_seq = reversed_seq[:half_steps]  # 前一半时间步
    logger.info(f"前半段采样序列: {half_seq[0]} -> {half_seq[-1]} (共{len(half_seq)}步, skip_type={skip_type})")

    intermediate_images = []

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="前半段采样"):
            current_batch_size = min(batch_size, num_samples - i)
            shape = (current_batch_size, config.data.channels,
                     config.data.image_size, config.data.image_size)
            x = torch.randn(shape, device=device)

            # 执行前半段采样
            xs, _ = generalized_steps(x, half_seq, qnn, betas, eta=eta)
            intermediate_images.append(xs[-1].cpu())

    intermediate_images = torch.cat(intermediate_images, dim=0)

    # 保存中间噪声图像
    if save_dir:
        save_path = os.path.join(save_dir, 'intermediate_noise_images.pt')
        torch.save(intermediate_images, save_path)
        logger.info(f"中间噪声图像已保存到: {save_path}")

    return intermediate_images


def second_half_sampling(qnn: nn.Module, intermediate_images: torch.Tensor,
                         betas: torch.Tensor, num_timesteps: int, total_timesteps: int,
                         device: torch.device, config, batch_size: int = 50,
                         skip_type: str = "uniform", eta: float = 0.0):
    """
    后半段采样：从中间噪声图像生成最终图像

    Args:
        qnn: 量化模型
        intermediate_images: 中间噪声图像
        betas: beta schedule
        num_timesteps: 采样步数
        total_timesteps: 总时间步数
        device: 设备
        config: 配置
        batch_size: 批大小
        skip_type: 采样类型 (uniform/quad)
        eta: eta参数

    Returns:
        final_images: 最终图像
    """
    logger.info("=" * 75)
    logger.info(f"后半段采样：从中间噪声图像生成最终图像")
    logger.info("=" * 75)

    qnn.eval()

    # 计算采样序列
    if skip_type == "uniform":
        skip = total_timesteps // num_timesteps
        seq = list(range(0, total_timesteps, skip))
    elif skip_type == "quad":
        seq = (
                np.linspace(0, np.sqrt(total_timesteps * 0.8), num_timesteps) ** 2
        )
        seq = [int(s) for s in list(seq)]
        seq = sorted(list(set(seq)))
    else:
        raise ValueError(f"不支持的skip_type: {skip_type}")

    # 只使用后半段序列
    reversed_seq = list(reversed(seq))
    half_steps = len(reversed_seq) // 2
    half_seq = reversed_seq[half_steps:]  # 后一半时间步
    logger.info(f"后半段采样序列: {half_seq[0]} -> {half_seq[-1]} (共{len(half_seq)}步, skip_type={skip_type})")

    final_images = []

    with torch.no_grad():
        for i in tqdm(range(0, intermediate_images.shape[0], batch_size), desc="后半段采样"):
            end_idx = min(i + batch_size, intermediate_images.shape[0])
            x = intermediate_images[i:end_idx].to(device)

            # 执行后半段采样
            xs, _ = generalized_steps(x, half_seq, qnn, betas, eta=eta)
            final_images.append(xs[-1].cpu())

    final_images = torch.cat(final_images, dim=0)
    logger.info(f"生成 {final_images.shape[0]} 张最终图像")

    return final_images


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        config.split_shortcut = self.args.split
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.betas = self.betas.to(self.device)
        betas = self.betas
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        # 检查是否启用两阶段量化模式
        use_two_stage = hasattr(self.args, 'max_avg_json') and self.args.max_avg_json

        if use_two_stage:
            logger.info("=" * 75)
            logger.info("两阶段量化采样模式（不进行激活截断）")
            logger.info("=" * 75)

            # 加载全精度模型
            model = Model(self.config)
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            logger.info("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            assert (self.args.cond == False)

            # 加载max_avg统计信息（仅用于分组，不用于截断）
            max_avg_stats = load_max_avg_statistics(self.args.max_avg_json)

            # 加载校准数据
            logger.info(f"加载校准数据: {self.args.cali_data_path}")
            sample_data = torch.load(self.args.cali_data_path, map_location=self.device)
            cali_data = get_train_samples(self.args, sample_data, custom_steps=0)
            del sample_data
            gc.collect()

            # 处理校准数据格式：如果是5D tensor，需要reshape为4D
            cali_xs, cali_ts = cali_data
            logger.info(f"原始校准数据形状: xs={cali_xs.shape}, ts={cali_ts.shape}")

            if len(cali_xs.shape) == 5:
                # 5D格式: [num_timesteps, batch_per_timestep, channels, height, width]
                # 需要reshape为4D: [num_timesteps * batch_per_timestep, channels, height, width]
                num_timesteps, batch_per_step = cali_xs.shape[0], cali_xs.shape[1]
                expected_channels = self.config.data.channels
                expected_size = self.config.data.image_size

                logger.info(f"检测到5D格式，reshape为4D")
                logger.info(f"  原始形状: {cali_xs.shape}")

                # 验证后三个维度是否正确
                if cali_xs.shape[2] != expected_channels or cali_xs.shape[3] != expected_size or cali_xs.shape[
                    4] != expected_size:
                    raise ValueError(f"校准数据维度不匹配: 期望 (N, {expected_channels}, {expected_size}, {expected_size}), "
                                     f"得到 {cali_xs.shape}")

                # Reshape为4D
                cali_xs = cali_xs.reshape(-1, expected_channels, expected_size, expected_size)
                total_samples = cali_xs.shape[0]
                logger.info(f"  Reshape后: {cali_xs.shape} (共 {total_samples} 个样本)")

                # 处理cali_ts以匹配新的batch size
                if len(cali_ts.shape) == 1:
                    if cali_ts.shape[0] == num_timesteps:
                        # 为每个时间步的每个样本复制timestep值
                        cali_ts = cali_ts.repeat_interleave(batch_per_step)
                        logger.info(f"  ts扩展: {num_timesteps} -> {cali_ts.shape[0]} (每个timestep复制 {batch_per_step} 次)")
                    elif cali_ts.shape[0] == total_samples:
                        logger.info(f"  ts长度已匹配: {cali_ts.shape[0]}")
                    else:
                        logger.warning(f"  ts长度不匹配: {cali_ts.shape[0]} != {total_samples}，尝试修复...")
                        if cali_ts.shape[0] < total_samples:
                            default_ts = 800
                            last_ts = cali_ts[-1].item() if cali_ts.shape[0] > 0 else default_ts
                            padding_size = total_samples - cali_ts.shape[0]
                            padding = torch.full((padding_size,), last_ts, dtype=cali_ts.dtype)
                            cali_ts = torch.cat([cali_ts, padding])
                            logger.info(f"  已扩展ts: {cali_ts.shape[0] - padding_size} -> {cali_ts.shape[0]}")
                        else:
                            cali_ts = cali_ts[:total_samples]
                            logger.info(
                                f"  已截断ts: {total_samples + (cali_ts.shape[0] - total_samples)} -> {cali_ts.shape[0]}")
                elif len(cali_ts.shape) == 2:
                    if cali_ts.shape[0] == num_timesteps and cali_ts.shape[1] == batch_per_step:
                        cali_ts = cali_ts.flatten()
                        logger.info(f"  ts从2D flatten为1D: {cali_ts.shape}")
                    else:
                        logger.warning(f"  ts是2D但形状不匹配: {cali_ts.shape}，尝试flatten...")
                        cali_ts = cali_ts.flatten()
                        if cali_ts.shape[0] != total_samples:
                            if cali_ts.shape[0] < total_samples:
                                default_ts = 800
                                padding_size = total_samples - cali_ts.shape[0]
                                padding = torch.full((padding_size,), default_ts, dtype=cali_ts.dtype)
                                cali_ts = torch.cat([cali_ts, padding])
                            else:
                                cali_ts = cali_ts[:total_samples]

            # 确保数据在CPU上（稍后移动到GPU）
            if cali_xs.device.type != 'cpu':
                cali_xs = cali_xs.cpu()
            if cali_ts.device.type != 'cpu':
                cali_ts = cali_ts.cpu()

            # 最终验证
            if len(cali_xs.shape) != 4:
                raise ValueError(f"cali_xs必须是4D tensor，得到: {cali_xs.shape}")
            if cali_xs.shape[0] != cali_ts.shape[0]:
                raise ValueError(f"cali_xs和cali_ts的batch size不匹配: {cali_xs.shape[0]} != {cali_ts.shape[0]}")

            logger.info(f"处理后的校准数据形状: xs={cali_xs.shape}, ts={cali_ts.shape}")
            cali_data = (cali_xs, cali_ts)

            if self.args.ptq:
                # ========== 第一阶段：使用组0进行BRECQ量化（不进行激活截断）==========
                logger.info("=" * 75)
                logger.info("第一阶段：使用组0进行BRECQ量化（不进行激活截断）")
                logger.info("=" * 75)

                # 创建第一阶段的量化模型
                fp_model_stage1 = Model(self.config)
                fp_model_stage1.load_state_dict(model.state_dict())
                fp_model_stage1.to(self.device)
                fp_model_stage1.eval()

                wq_params = {'n_bits': self.args.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
                aq_params = {'n_bits': self.args.act_bit, 'symmetric': self.args.a_sym, 'channel_wise': False,
                             'scale_method': 'max', 'leaf_param': self.args.quant_act}

                qnn_stage1 = QuantModel(
                    model=fp_model_stage1, weight_quant_params=wq_params, act_quant_params=aq_params,
                    sm_abit=self.args.sm_abit)
                qnn_stage1.to(self.device)
                qnn_stage1.eval()

                # 第一阶段BRECQ量化（不进行激活截断）
                brecq_quantization(fp_model_stage1, qnn_stage1, cali_data, max_avg_stats, group_id=0, args=self.args)

                # 第一阶段采样：生成中间噪声图像
                logger.info("第一阶段：开始采样生成中间噪声图像")
                intermediate_images = first_half_sampling(
                    qnn_stage1, self.args.max_images, self.betas,
                    self.args.timesteps, self.num_timesteps, self.device,
                    self.config, batch_size=50, save_dir=self.args.logdir,
                    skip_type=self.args.skip_type, eta=self.args.eta
                )

                # 清理第一阶段模型
                del qnn_stage1, fp_model_stage1
                torch.cuda.empty_cache()
                gc.collect()

                # ========== 第二阶段：使用组1进行BRECQ量化（不进行激活截断）==========
                logger.info("=" * 75)
                logger.info("第二阶段：使用组1进行BRECQ量化（不进行激活截断）")
                logger.info("=" * 75)

                # 创建第二阶段的量化模型
                fp_model_stage2 = Model(self.config)
                fp_model_stage2.load_state_dict(model.state_dict())
                fp_model_stage2.to(self.device)
                fp_model_stage2.eval()

                qnn_stage2 = QuantModel(
                    model=fp_model_stage2, weight_quant_params=wq_params, act_quant_params=aq_params,
                    sm_abit=self.args.sm_abit)
                qnn_stage2.to(self.device)
                qnn_stage2.eval()

                # 第二阶段BRECQ量化（不进行激活截断）
                brecq_quantization(fp_model_stage2, qnn_stage2, cali_data, max_avg_stats, group_id=1, args=self.args)

                # 第二阶段采样：从中间噪声图像生成最终图像
                logger.info("第二阶段：从中间噪声图像生成最终图像")
                final_images = second_half_sampling(
                    qnn_stage2, intermediate_images, self.betas,
                    self.args.timesteps, self.num_timesteps, self.device,
                    self.config, batch_size=50,
                    skip_type=self.args.skip_type, eta=self.args.eta
                )

                # 保存最终图像
                logger.info("保存最终图像")
                # 保存为.pt文件
                pt_path = os.path.join(self.args.logdir, 'final_samples.pt')
                torch.save(final_images, pt_path)
                logger.info(f"最终图像已保存到: {pt_path}")

                # 转换为PNG并保存
                final_images = inverse_data_transform(self.config, final_images)
                img_dir = os.path.join(self.args.logdir, 'images')
                os.makedirs(img_dir, exist_ok=True)

                for i in range(final_images.shape[0]):
                    tvu.save_image(final_images[i], os.path.join(img_dir, f"{i}.png"))

                logger.info(f"最终图像已保存为PNG到: {img_dir}")
                logger.info("两阶段量化采样完成（未进行激活截断）！")
        else:
            # 原有的单阶段量化流程
            model = Model(self.config)

            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            logger.info("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))

            model.to(self.device)
            model.eval()
            assert (self.args.cond == False)
            if self.args.ptq:
                if self.args.quant_mode == 'qdiff':
                    wq_params = {'n_bits': self.args.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
                    aq_params = {'n_bits': self.args.act_bit, 'symmetric': self.args.a_sym, 'channel_wise': False,
                                 'scale_method': 'max', 'leaf_param': self.args.quant_act}
                    if self.args.resume:
                        logger.info('Load with min-max quick initialization')
                        wq_params['scale_method'] = 'max'
                        aq_params['scale_method'] = 'max'
                    if self.args.resume_w:
                        wq_params['scale_method'] = 'max'
                    qnn = QuantModel(
                        model=model, weight_quant_params=wq_params, act_quant_params=aq_params,
                        sm_abit=self.args.sm_abit)
                    qnn.to(self.device)
                    qnn.eval()

                    if self.args.resume:
                        image_size = self.config.data.image_size
                        channels = self.config.data.channels
                        cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)))
                        resume_cali_model(qnn, self.args.cali_ckpt, cali_data, self.args.quant_act, "qdiff", cond=False)
                    else:
                        logger.info(f"Sampling data from {self.args.cali_st} timesteps for calibration")
                        sample_data = torch.load(self.args.cali_data_path)
                        sample_data = normalize_cali_sample_data(sample_data, self.args)
                        cali_data = get_train_samples(self.args, sample_data, custom_steps=0)
                        del (sample_data)
                        gc.collect()
                        logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape}")

                        cali_xs, cali_ts = cali_data
                        if self.args.resume_w:
                            resume_cali_model(qnn, self.args.cali_ckpt, cali_data, False, cond=False)
                        else:
                            logger.info("Initializing weight quantization parameters")
                            qnn.set_quant_state(True, False)  # enable weight quantization, disable act quantization
                            _ = qnn(cali_xs[:8].cuda(), cali_ts[:8].cuda())
                            logger.info("Initializing has done!")

                        # Kwargs for weight rounding calibration
                        kwargs = dict(cali_data=cali_data, batch_size=self.args.cali_batch_size,
                                      iters=self.args.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                                      warmup=0.2, act_quant=False, opt_mode='mse')

                        def recon_model(model):
                            """
                            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                            """
                            for name, module in model.named_children():
                                logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
                                if isinstance(module, QuantModule):
                                    if module.ignore_reconstruction is True:
                                        logger.info('Ignore reconstruction of layer {}'.format(name))
                                        continue
                                    else:
                                        logger.info('Reconstruction for layer {}'.format(name))
                                        layer_reconstruction(qnn, module, **kwargs)
                                elif isinstance(module, BaseQuantBlock):
                                    if module.ignore_reconstruction is True:
                                        logger.info('Ignore reconstruction of block {}'.format(name))
                                        continue
                                    else:
                                        logger.info('Reconstruction for block {}'.format(name))
                                        block_reconstruction(qnn, module, **kwargs)
                                else:
                                    recon_model(module)

                        if not self.args.resume_w:
                            logger.info("Doing weight calibration")
                            recon_model(qnn)
                            qnn.set_quant_state(weight_quant=True, act_quant=False)
                        if self.args.quant_act:
                            logger.info("UNet model")
                            logger.info(model)
                            logger.info("Doing activation calibration")
                            # Initialize activation quantization parameters
                            qnn.set_quant_state(True, True)
                            with torch.no_grad():
                                inds = np.random.choice(cali_xs.shape[0], 64, replace=False)
                                # _ = qnn(cali_xs[:64].cuda(), cali_ts[:64].cuda())
                                _ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda())

                                if self.args.running_stat:
                                    logger.info('Running stat for activation quantization')
                                    qnn.set_running_stat(True)
                                    for i in range(int(cali_xs.size(0) / 64)):
                                        _ = qnn(
                                            (cali_xs[i * 64:(i + 1) * 64].to(self.device),
                                             cali_ts[i * 64:(i + 1) * 64].to(self.device)))
                                    qnn.set_running_stat(False)

                            kwargs = dict(
                                cali_data=cali_data, iters=self.args.cali_iters_a, act_quant=True,
                                opt_mode='mse', lr=self.args.cali_lr, p=self.args.cali_p)
                            recon_model(qnn)
                            qnn.set_quant_state(weight_quant=True, act_quant=True)

                        logger.info("Saving calibrated quantized UNet model")
                        for m in qnn.model.modules():
                            if isinstance(m, AdaRoundQuantizer):
                                m.zero_point = nn.Parameter(m.zero_point)
                                m.delta = nn.Parameter(m.delta)
                            elif isinstance(m, UniformAffineQuantizer) and self.args.quant_act:
                                if m.zero_point is not None:
                                    if not torch.is_tensor(m.zero_point):
                                        m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                                    else:
                                        m.zero_point = nn.Parameter(m.zero_point)
                        torch.save(qnn.state_dict(), os.path.join(self.args.logdir, "ckpt.pth"))

                    model = qnn

            model.to(self.device)
            if self.args.verbose:
                logger.info("quantized model")
                logger.info(model)

            model.eval()

            self.sample_fid(model)

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        logger.info(f"starting from image {img_id}")
        total_n_samples = self.args.max_images
        n_rounds = math.ceil((total_n_samples - img_id) / config.sampling.batch_size)

        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.no_grad():
            for i in tqdm(
                    range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                with amp.autocast(enabled=False):
                    x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                if img_id + x.shape[0] > self.args.max_images:
                    assert (i == n_rounds - 1)
                    n = self.args.max_images - img_id
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from ddim.functions.denoising import generalized_steps

            betas = self.betas
            xs = generalized_steps(
                x, seq, model, betas, eta=self.args.eta, args=self.args)
            x = xs
        elif self.args.sample_type == "dpm_solver":
            logger.info(f"use dpm-solver with {self.args.timesteps} steps")
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type="noise"
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
            return dpm_solver.sample(
                x,
                steps=self.args.timesteps,
                order=3,
                skip_type="time_uniform",
                method="singlestep",
            )
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true",
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="qdiff",
        choices=["qdiff"],
        help="quantization mode to use"
    )
    parser.add_argument(
        "--max_images", type=int, default=10000, help="number of images to sample"
    )

    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1,
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32,
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024,
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000,
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int,
                        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float,
                        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float,
                        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    parser.add_argument(
        "--cali_t_index", type=int, default=0,
        help="当校准集为 [T,N,C,H,W] 时，选择第几个时间步作为 [N,C,H,W] 校准输入"
    )
    parser.add_argument(
        "--max_avg_json", type=str, default=r"C:\Users\wsh\Desktop\q-diffusion-master\layerwise_activation_statistics.json",
        help="Path to JSON file containing max_avg statistics for two-stage quantization (optional, enables two-stage mode)"
    )
    parser.add_argument(
        "--skip_weight_recon", action="store_true",
        help="Skip weight reconstruction, only perform activation reconstruction (for quick validation)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--a_sym", action="store_true",
        help="act quantizers use symmetric quantization"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--sm_abit", type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument("--split", action="store_true",
                        help="split shortcut connection into two parts"
                        )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    return parser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)

    # setup logger
    logdir = os.path.join(args.logdir, "samples", now)
    os.makedirs(logdir)
    args.logdir = logdir
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(75 * "=")
    # 使用跨平台方法获取主机名（Windows和Linux都支持）
    hostname = platform.node()
    logger.info(f"Host {hostname}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    args.image_folder = imglogdir

    os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    runner = Diffusion(args, config)
    runner.sample()