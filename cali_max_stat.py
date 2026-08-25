import argparse, os, gc, glob, datetime, yaml
import logging
import math
import sys
import json
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
from pytorch_lightning import seed_everything
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ddim.models
from ddim.models.diffusion import Model
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import generalized_steps
from qdiff.utils import get_train_samples
import torchvision.utils as tvu

logger = logging.getLogger(__name__)


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
    elif beta_schedule == "jsd":
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


def register_activation_hooks_for_sampling(model, storage_dict, num_timesteps, skip_type="uniform"):
    """
    注册钩子收集模型在采样过程中的激活统计信息
    区分前一半和后一半去噪时间步

    :param model: 要监控的模型
    :param storage_dict: {layer_name: {0: {t: [max_values...]}, 1: {...}}} 按组、按全局时间步 t 收集
    :param num_timesteps: 采样时间步数
    :param skip_type: 采样序列类型 ("uniform" 或 "quad")
    """
    # 计算采样序列
    total_timesteps = 1000  # 默认总时间步数
    if skip_type == "uniform":
        skip = total_timesteps // num_timesteps
        seq = list(range(0, total_timesteps, skip))
    elif skip_type == "quad":
        seq = (
                np.linspace(
                    0, np.sqrt(total_timesteps * 0.8), num_timesteps
                ) ** 2
        )
        seq = [int(s) for s in list(seq)]
        seq = sorted(list(set(seq)))
    else:
        raise ValueError(f"不支持的skip_type: {skip_type}")

    # 反转序列（从高噪声到低噪声）
    reversed_seq = list(reversed(seq))
    half_steps = len(reversed_seq) // 2
    group0_timesteps = set(reversed_seq[:half_steps])  # 前一半（组0）
    group1_timesteps = set(reversed_seq[half_steps:])  # 后一半（组1）

    logger.info(f"采样序列长度: {len(seq)}")
    logger.info(f"前一半时间步（组0）: {len(group0_timesteps)} 个，范围 [{min(group0_timesteps)}, {max(group0_timesteps)}]")
    logger.info(f"后一半时间步（组1）: {len(group1_timesteps)} 个，范围 [{min(group1_timesteps)}, {max(group1_timesteps)}]")

    # 当前时间步（全局变量，用于hook判断）
    current_timestep = [-1]  # 使用列表以便在hook中修改

    def get_hook(name):
        def hook_fn(module, input, output):
            if not model.training:  # 只在评估模式收集
                # 获取当前时间步
                timestep_val = current_timestep[0]
                if timestep_val < 0:
                    return  # 未设置时间步，跳过

                # 判断属于哪个组
                if timestep_val in group0_timesteps:
                    group_id = 0
                elif timestep_val in group1_timesteps:
                    group_id = 1
                else:
                    return  # 不属于任何组，跳过

                # 获取激活值
                if isinstance(output, torch.Tensor):
                    act = output.detach().cpu().numpy()
                    batch_size = act.shape[0]
                    flattened_act = act.reshape(batch_size, -1)

                    # 计算每个样本的最大值
                    max_vals = flattened_act.max(axis=1)

                    # 按 (组, 时间步) 追加；汇总后 per_timestep_upper[t].c_upper = 该 t 在本组内的截断阈值
                    storage_dict[name][group_id][timestep_val].extend(max_vals.tolist())

        return hook_fn

    # 为所有模块注册钩子
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(get_hook(name))
        hooks.append((name, hook))

    logger.info(f"已为 {len(hooks)} 个模块注册激活钩子")

    return hooks, current_timestep


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        # 确保配置包含 split_shortcut 属性
        if not hasattr(config, 'split_shortcut'):
            config.split_shortcut = args.split

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        # {layer_name: {0: defaultdict(list) keyed by t, 1: ...}}
        self.activation_dict = defaultdict(
            lambda: {0: defaultdict(list), 1: defaultdict(list)}
        )

        # 存储hooks和当前时间步
        self.hooks = []
        self.current_timestep = None

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
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def calibrate(self, model, cali_data_path, num_samples=100, num_timesteps=100, skip_type="uniform"):
        """
        使用采样过程计算每一层的激活统计信息（前一半和后一半时间步）

        :param model: 用于校准的模型
        :param cali_data_path: 校准集路径（用于初始化，实际使用采样过程）
        :param num_samples: 生成的样本数
        :param num_timesteps: 采样时间步数
        :param skip_type: 采样序列类型 ("uniform" 或 "quad")
        """
        logger.info("=" * 75)
        logger.info("开始收集激活统计信息（通过采样过程）")
        logger.info(f"生成样本数: {num_samples}")
        logger.info(f"采样时间步数: {num_timesteps}")
        logger.info(f"采样序列类型: {skip_type}")
        logger.info("=" * 75)

        # 注册激活钩子
        hooks, current_timestep = register_activation_hooks_for_sampling(
            model, self.activation_dict, num_timesteps, skip_type=skip_type
        )
        self.hooks = hooks
        self.current_timestep = current_timestep

        # 计算采样序列
        total_timesteps = 1000
        if skip_type == "uniform":
            skip = total_timesteps // num_timesteps
            seq = list(range(0, total_timesteps, skip))
        elif skip_type == "quad":
            seq = (
                    np.linspace(
                        0, np.sqrt(total_timesteps * 0.8), num_timesteps
                    ) ** 2
            )
            seq = [int(s) for s in list(seq)]
            seq = sorted(list(set(seq)))
        else:
            raise ValueError(f"不支持的skip_type: {skip_type}")

        # 反转序列（从高噪声到低噪声）
        reversed_seq = list(reversed(seq))
        half_steps = len(reversed_seq) // 2
        group0_list = list(reversed_seq[:half_steps])

        logger.info(f"开始采样生成图像...")
        model.eval()

        batch_size = 10  # 每批生成的图像数
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, num_samples, batch_size), desc="生成图像"):
                current_batch_size = min(batch_size, num_samples - i)
                shape = (current_batch_size, 3, 32, 32)  # CIFAR10
                x = torch.randn(shape, device=self.device)

                # 执行采样，在采样过程中收集激活
                xs = self.sample_image_with_timestep_tracking(
                    x, seq, model, return_all=True
                )

                # 每处理10张图像后清理GPU缓存
                if (i // batch_size + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        logger.info("采样完成，开始计算统计信息...")

        # 移除hooks
        for name, hook in self.hooks:
            hook.remove()
        logger.info(f"已移除 {len(self.hooks)} 个钩子")

        # 计算并保存结果
        stats = self.compute_layer_stats()
        self.save_calibration_stats(stats, cali_data_path, group0_timesteps=group0_list)

        return stats

    def sample_image_with_timestep_tracking(self, x, seq, model, return_all=False):
        """
        执行采样，并在每个时间步设置current_timestep以便hook能正确识别组
        """
        from ddim.functions.denoising import compute_alpha

        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]

        # 反转序列以便从高噪声到低噪声采样
        reversed_seq = list(reversed(seq))
        reversed_seq_next = list(reversed(seq_next))

        for i, j in zip(reversed_seq, reversed_seq_next):
            # 设置当前时间步，让hook知道当前是哪个时间步
            self.current_timestep[0] = int(i)

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(self.betas, t.long())
            at_next = compute_alpha(self.betas, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                    self.args.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

        # 清除时间步标记
        self.current_timestep[0] = -1

        if return_all:
            return xs
        else:
            return xs[0][-1]

    def _flatten_group(self, group_dict):
        """group_dict: t -> list of per-sample max; 返回 flat 列表与各 t 的截断阈值条目。"""
        flat = []
        per_t_upper = {}
        for t in sorted(group_dict.keys()):
            vals = group_dict[t]
            if not vals:
                continue
            flat.extend(vals)
            arr = np.asarray(vals, dtype=np.float64)
            per_t_upper[str(int(t))] = {
                "c_upper": float(np.max(arr)),
                "n_samples": int(len(vals)),
            }
        return flat, per_t_upper

    def compute_layer_stats(self):
        """计算每层统计：保留跨步汇总，并写入 per_timestep_upper（各时间步在其所属分组内的截断阈值 c_upper）。"""
        layer_stats = {}
        save_all = getattr(self.args, "save_all_max_values", False)

        for layer_name, groups in self.activation_dict.items():
            if not groups[0] and not groups[1]:
                continue

            layer_stats[layer_name] = {}

            if groups[0]:
                g0_flat, g0_per_t = self._flatten_group(groups[0])
                group0_max_vals = np.array(g0_flat) if g0_flat else np.array([])
            else:
                group0_max_vals = np.array([])
                g0_per_t = {}

            if groups[1]:
                g1_flat, g1_per_t = self._flatten_group(groups[1])
                group1_max_vals = np.array(g1_flat) if g1_flat else np.array([])
            else:
                group1_max_vals = np.array([])
                g1_per_t = {}

            if group0_max_vals.size > 0:
                layer_stats[layer_name]["0"] = {
                    "max_avg": float(np.mean(group0_max_vals)),
                    "min_value": float(np.min(group0_max_vals)),
                    "max_value": float(np.max(group0_max_vals)),
                    "n_samples": int(group0_max_vals.size),
                    "per_timestep_upper": g0_per_t,
                }
                if save_all:
                    layer_stats[layer_name]["0"]["all_max_values"] = [
                        float(v) for v in g0_flat
                    ]
            else:
                layer_stats[layer_name]["0"] = {
                    "max_avg": 0.0,
                    "min_value": 0.0,
                    "max_value": 0.0,
                    "n_samples": 0,
                    "per_timestep_upper": {},
                }
                if save_all:
                    layer_stats[layer_name]["0"]["all_max_values"] = []

            if group1_max_vals.size > 0:
                layer_stats[layer_name]["1"] = {
                    "max_avg": float(np.mean(group1_max_vals)),
                    "min_value": float(np.min(group1_max_vals)),
                    "max_value": float(np.max(group1_max_vals)),
                    "n_samples": int(group1_max_vals.size),
                    "per_timestep_upper": g1_per_t,
                }
                if save_all:
                    layer_stats[layer_name]["1"]["all_max_values"] = [
                        float(v) for v in g1_flat
                    ]
            else:
                layer_stats[layer_name]["1"] = {
                    "max_avg": 0.0,
                    "min_value": 0.0,
                    "max_value": 0.0,
                    "n_samples": 0,
                    "per_timestep_upper": {},
                }
                if save_all:
                    layer_stats[layer_name]["1"]["all_max_values"] = []

            logger.info(
                f"层 {layer_name}: "
                f"组0 max_avg={layer_stats[layer_name]['0']['max_avg']:.4f} "
                f"(n={layer_stats[layer_name]['0']['n_samples']}, "
                f"组0步数={len(g0_per_t)}), "
                f"组1 max_avg={layer_stats[layer_name]['1']['max_avg']:.4f} "
                f"(n={layer_stats[layer_name]['1']['n_samples']})"
            )

        return layer_stats

    def save_calibration_stats(self, stats, cali_data_path, group0_timesteps=None):
        """保存校准统计到JSON文件"""
        if not stats:
            logger.warning("无统计信息可保存")
            return

        # 创建结果字典
        output_data = {
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": cali_data_path,
            "description": (
                "每层在组0、组1下分别给出标量摘要；"
                "per_timestep_upper 的键为全局时间步 t，值为该 t 所在分组内的截断阈值 "
                "（由校准集上该 (层,t) 的 per-sample max 取 max 得到，即 c_upper）。"
            ),
            "layers": stats,
            "meta": {
                "timesteps": getattr(self.args, "timesteps", None),
                "skip_type": getattr(self.args, "skip_type", None),
                "group0_timesteps": (
                    [int(x) for x in group0_timesteps] if group0_timesteps is not None else None
                ),
            },
        }

        # 统计信息摘要
        total_values = 0
        for layer_name, layer_data in stats.items():
            total_values += layer_data.get('0', {}).get('n_samples', 0)
            total_values += layer_data.get('1', {}).get('n_samples', 0)

        logger.info(f"总共收集了 {total_values} 个激活最大值")
        logger.info(
            "每层数据包含: max_avg, min_value, max_value, n_samples, per_timestep_upper（该步所属分组的截断阈值）；"
            "若加 --save_all_max_values 则另有 all_max_values"
        )

        stats_path = os.path.join(self.args.logdir, "activation_max_avg_stats.json")

        # 保存到文件
        with open(stats_path, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"激活统计已保存至 {stats_path}")
        logger.info(f"共收集 {len(stats)} 层的统计信息")

    def sample(self):
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
            for i in tqdm.tqdm(
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
        help="sampling approach (generalized or ddpm_noisy)"
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="quad",
        help="skip according to (uniform or quadratic)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma"
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
        "--max_images", type=int, default=50000, help="number of images to sample"
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
        "--cali_data_path", type=str, default="",
        help="path to calibration dataset"
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
    parser.add_argument(
        "--split", action="store_true",
        help="split shortcut connection into two parts"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="run calibration mode instead of sampling"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="number of samples to generate for calibration"
    )
    parser.add_argument(
        "--save_all_max_values",
        action="store_true",
        help="JSON 中额外写入 all_max_values（文件巨大）；默认仅 per_timestep_upper + 标量摘要",
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

    # 确保配置有 split_shortcut 属性
    if not hasattr(config, 'split_shortcut'):
        config.split_shortcut = args.split
        logger.info(f"设置 config.split_shortcut 为 {args.split}")

    # fix random seed
    seed_everything(args.seed)

    # setup logger
    logdir = os.path.join(args.logdir, "samples", now)
    os.makedirs(logdir, exist_ok=True)
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
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    args.image_folder = imglogdir
    os.makedirs(imglogdir, exist_ok=True)
    logger.info(logdir)
    logger.info(75 * "=")

    runner = Diffusion(args, config)

    # 优先处理校准模式
    if args.calibrate:
        logger.info("=== CALIBRATION MODE ===")

        # 加载模型
        model = Model(config)
        if config.data.dataset == "CIFAR10":
            model_name = "cifar10"
        elif config.data.dataset == "LSUN":
            model_name = f"lsun_{config.data.category}"
        else:
            raise ValueError

        ckpt_path = get_ckpt_path(f"ema_{model_name}")
        logger.info(f"加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=runner.device))
        model.to(runner.device)
        model.eval()

        # 执行校准（通过采样过程收集激活统计）
        cali_path = args.cali_data_path if args.cali_data_path else "cifar_sd1236_sample2048_allst.pt"
        stats = runner.calibrate(
            model,
            cali_path,
            num_samples=args.num_samples,
            num_timesteps=args.timesteps,
            skip_type=args.skip_type
        )

        if stats:
            logger.info(f"校准完成! 共收集 {len(stats)} 层的统计信息")
            logger.info("统计信息包含每层在前一半和后一半去噪时间步的激活最大值平均值")
        else:
            logger.error("校准失败")

    else:
        logger.info("=== SAMPLING MODE ===")
        runner.sample()

