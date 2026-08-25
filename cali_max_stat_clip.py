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
import torch.nn.functional as F
sys.path.append(r'C:\Users\LPY\Desktop\ccy实验\q-diffusion-master')
import ddim.models
from ddim.models.diffusion import Model
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
from qdiff.utils import get_train_samples

import torchvision.utils as tvu

logger = logging.getLogger(__name__)


# 钩子函数
def register_activation_hooks(model, storage_dict):
    """
    注册钩子收集模型的激活统计信息
    :param model: 要监控的模型
    :param storage_dict: 用于存储统计数据的字典
    """

    def get_hook(name):
        def hook_fn(module, input, output):
            if not model.training:  # 只在eval模式收集
                device = output.device

                # 打印输入和输出的shape，方便确认
                # input通常是一个元组，取第一个张量
                # input_tensor = input[0]
                # print(
                #     f"Layer: {module.__class__.__name__}, input shape: {input_tensor.shape}, output shape: {output.shape}")

                # 获取激活值并展平为 [batch_size, C*H*W]
                act_tensor = output.detach()
                batch_size = act_tensor.shape[0]
                flattened_act = act_tensor.view(batch_size, -1).cpu().numpy()

                # 计算每个样本的最大值（用于截断范围）
                max_vals = flattened_act.max(axis=1)
                avg_max_val = np.mean(max_vals)

                # 截断激活值
                truncated_act = torch.clamp(act_tensor, -avg_max_val, avg_max_val)

                # 初始化存储结构
                if name not in storage_dict:
                    storage_dict[name] = {
                        "max_values": [],
                        "x": [],
                        "x_bar": []
                    }

                # 存储最大值和激活
                storage_dict[name]["max_values"].append(max_vals)
                storage_dict[name]["x"].append(act_tensor.detach().cpu())
                storage_dict[name]["x_bar"].append(truncated_act.detach().cpu())

        return hook_fn

    # 为所有模块注册钩子
    for name, module in model.named_modules():
        module.register_forward_hook(get_hook(name))





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
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 极
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

        # 激活存储字典
        self.activation_dict = {}

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

    # def compute_delta_weights(self, model, lambda1=1e4):
    #     delta_weights = {}
    #
    #     for layer_name, layer_data in self.activation_dict.items():
    #         if "x" not in layer_data or "x_bar" not in layer_data:
    #             print(f"[{layer_name}] Missing x or x_bar, skipping.")
    #             continue
    #         if not layer_data["x"] or not layer_data["x_bar"]:
    #             print(f"[{layer_name}] Empty x or x_bar, skipping.")
    #             continue
    #
    #         # 获取当前层模块
    #         layer_module = dict(model.named_modules()).get(layer_name, None)
    #         if layer_module is None or not hasattr(layer_module, 'weight'):
    #             print(f"[{layer_name}] No weight attribute, skipping.")
    #             continue
    #
    #         # 判断层类型：卷积层（Conv2d）还是线性层（Linear）
    #         if isinstance(layer_module, torch.nn.Conv2d):
    #             print(f"Processing Conv2d layer: {layer_name}")
    #
    #             # 卷积层的权重 W，形状是 [out_channels, in_channels, kernel_height, kernel_width]
    #             W = layer_module.weight.data.clone().detach().cpu()
    #             out_channels, in_channels, kernel_height, kernel_width = W.shape
    #             W = W.view(out_channels, -1)  # 重塑为 [out_channels, in_channels * kernel_height * kernel_width]
    #
    #             # 获取卷积层的激活数据并展开为 [N, in_channels * height * width]
    #             x_all = torch.cat(layer_data["x"], dim=0)  # [N, in_channels, height, width]
    #             N, in_channels, height, width = x_all.shape
    #             x_all = x_all.view(N, in_channels * height * width)
    #
    #             x_bar_all = torch.cat(layer_data["x_bar"], dim=0).view(N, in_channels * height * width)
    #
    #             delta_x = x_all - x_bar_all  # [N, in_channels * height * width]
    #
    #             # 打印信息：查看张量的形状
    #             print(f"W shape: {W.shape}")  # [out_channels, in_channels * kernel_height * kernel_width]
    #             print(f"x_all shape: {x_all.shape}")  # [N, in_channels * height * width]
    #             print(f"x_bar_all shape: {x_bar_all.shape}")  # [N, in_channels * height * width]
    #             print(f"delta_x shape: {delta_x.shape}")  # [N, in_channels * height * width]
    #
    #             in_dim = W.shape[1]  # 层的输入维度
    #
    #             # 协方差矩阵
    #             E_xbar_xbar = (x_bar_all @ x_bar_all.T) / x_all.shape[0]  # [in_dim, in_dim]
    #             E_xbar_delta = (x_bar_all @ delta_x.T) / x_all.shape[0]  # [in_dim, in_dim]
    #
    #             print(f"E_xbar_xbar shape: {E_xbar_xbar.shape}")  # [in_dim, in_dim]
    #             print(f"E_xbar_delta shape: {E_xbar_delta.shape}")  # [in_dim, in_dim]
    #
    #             I = torch.eye(in_dim)  # 使用正确的维度的单位矩阵
    #
    #             try:
    #                 inv_matrix = torch.inverse(E_xbar_xbar + lambda1 * I)  # 求逆
    #                 print(f"Inverse matrix shape: {inv_matrix.shape}")  # [in_dim, in_dim]
    #             except RuntimeError as e:
    #                 print(f"Failed to invert matrix in layer {layer_name}: {e}")
    #                 continue
    #
    #             # 计算卷积层的 delta_w
    #             delta_w = - W @ E_xbar_delta @ inv_matrix  # 计算权重更新
    #             print(f"delta_w shape: {delta_w.shape}")
    #
    #             delta_weights[layer_name] = delta_w
    #
    #         elif isinstance(layer_module, torch.nn.Linear):
    #             print(f"Processing Linear layer: {layer_name}")
    #
    #             # 线性层的权重 W，形状是 [out_dim, in_dim]
    #             W = layer_module.weight.data.clone().detach().cpu()
    #             W = W.view(W.shape[0], -1)  # [out_dim, in_dim]
    #
    #             # 获取线性层的激活数据并展开为 [N, in_dim]
    #             x_all = torch.cat(layer_data["x"], dim=0)  # [N, in_dim]
    #             x_all = x_all.view(x_all.shape[0], -1)  # [N, in_dim]
    #             x_bar_all = torch.cat(layer_data["x_bar"], dim=0).view(x_all.shape[0], -1)
    #
    #             delta_x = x_all - x_bar_all  # [N, in_dim]
    #
    #             # 打印信息：查看张量的形状
    #             print(f"W shape: {W.shape}")  # [out_dim, in_dim]
    #             print(f"x_all shape: {x_all.shape}")  # [N, in_dim]
    #             print(f"x_bar_all shape: {x_bar_all.shape}")  # [N, in_dim]
    #             print(f"delta_x shape: {delta_x.shape}")  # [N, in_dim]
    #
    #             in_dim = W.shape[1]  # 层的输入维度
    #
    #             # 协方差矩阵
    #             E_xbar_xbar = (x_bar_all.T @ x_bar_all) / x_all.shape[0]  # [in_dim, in_dim]
    #             E_xbar_delta = (delta_x.T @ x_bar_all) / x_all.shape[0]  # [in_dim, in_dim]
    #
    #             print(f"E_xbar_xbar shape: {E_xbar_xbar.shape}")  # [in_dim, in_dim]
    #             print(f"E_xbar_delta shape: {E_xbar_delta.shape}")  # [in_dim, in_dim]
    #
    #             I = torch.eye(in_dim)  # 使用正确的维度的单位矩阵
    #
    #             try:
    #                 inv_matrix = torch.inverse(E_xbar_xbar + lambda1 * I)  # 求逆
    #                 print(f"Inverse matrix shape: {inv_matrix.shape}")  # [in_dim, in_dim]
    #             except RuntimeError as e:
    #                 print(f"Failed to invert matrix in layer {layer_name}: {e}")
    #                 continue
    #
    #             # 计算线性层的 delta_w
    #             delta_w = - W @ E_xbar_delta @ inv_matrix  # 计算权重更新
    #             print(f"delta_w shape: {delta_w.shape}")
    #
    #             delta_weights[layer_name] = delta_w
    #
    #         else:
    #             print(f"[{layer_name}] Unsupported layer type, skipping.")
    #
    #     return delta_weights
    def compute_delta_weights(self, model, lambda1=1e4):
        delta_weights = {}

        for layer_name, layer_data in self.activation_dict.items():
            if "x" not in layer_data or "x_bar" not in layer_data:
                print(f"[{layer_name}] Missing x or x_bar, skipping.")
                continue
            if not layer_data["x"] or not layer_data["x_bar"]:
                print(f"[{layer_name}] Empty x or x_bar, skipping.")
                continue

            # === 修改点：跳过 conv1 和 nin_shortcut 层 ===
            if "conv1" in layer_name or "nin_shortcut" in layer_name:
                print(f"[{layer_name}] Skipping conv1 or shortcut layer due to likely concat/residual mismatch.")
                continue

            layer_module = dict(model.named_modules()).get(layer_name, None)
            if layer_module is None or not hasattr(layer_module, 'weight'):
                print(f"[{layer_name}] No weight attribute, skipping.")
                continue

            W = layer_module.weight.data.clone().detach().cpu()
            if isinstance(layer_module, torch.nn.Conv2d):
                print(f"Processing Conv2d layer: {layer_name}")
                out_channels, in_channels, kh, kw = W.shape
                print(W.shape)
                W_flat = W.view(out_channels, -1)
                expected_in_dim = W_flat.shape[1]

                x_all = torch.cat(layer_data["x"], dim=0).cpu()
                x_bar_all = torch.cat(layer_data["x_bar"], dim=0).cpu()

                try:
                    x_all_unfold = F.unfold(x_all, kernel_size=(kh, kw))
                    x_bar_all_unfold = F.unfold(x_bar_all, kernel_size=(kh, kw))
                    N, Ck, L = x_all_unfold.shape
                    x_all_flat = x_all_unfold.permute(0, 2, 1).reshape(N * L, Ck)
                    x_bar_all_flat = x_bar_all_unfold.permute(0, 2, 1).reshape(N * L, Ck)
                    print(f"[{layer_name}] Used unfold, flat shape: {x_bar_all_flat.shape}")
                except Exception as e:
                    print(f"[{layer_name}] Unfold failed: {e}")
                    try:
                        B, C, H, W_ = x_all.shape
                        if C == expected_in_dim:
                            x_all_flat = x_all.permute(0, 2, 3, 1).reshape(-1, C)
                            x_bar_all_flat = x_bar_all.permute(0, 2, 3, 1).reshape(-1, C)
                            print(f"[{layer_name}] Used fallback reshape (N,H,W,C), flat shape: {x_bar_all_flat.shape}")
                        elif H * W_ == expected_in_dim:
                            x_all_flat = x_all.view(-1, H * W_)
                            x_bar_all_flat = x_bar_all.view(-1, H * W_)
                            print(f"[{layer_name}] Used fallback reshape (N,C,H*W), flat shape: {x_bar_all_flat.shape}")
                        elif C * H * W_ == expected_in_dim:
                            x_all_flat = x_all.view(-1, C * H * W_)
                            x_bar_all_flat = x_bar_all.view(-1, C * H * W_)
                            print(f"[{layer_name}] Used fallback reshape (N,C*H*W), flat shape: {x_bar_all_flat.shape}")
                        else:
                            print(
                                f"[{layer_name}] Cannot reshape input to match weight dim {expected_in_dim}, skipping.")
                            continue
                    except Exception as e2:
                        print(f"[{layer_name}] Fallback reshape failed: {e2}")
                        continue

                delta_x = x_all_flat - x_bar_all_flat

                E_xbar_xbar = (x_bar_all_flat.T @ x_bar_all_flat) / x_bar_all_flat.shape[0]
                E_xbar_delta = (x_bar_all_flat.T @ delta_x) / x_bar_all_flat.shape[0]

                I = torch.eye(expected_in_dim)

                try:
                    inv_matrix = torch.inverse(E_xbar_xbar + lambda1 * I)
                except Exception as e:
                    print(f"[{layer_name}] Matrix inversion failed: {e}")
                    continue

                delta_w = - W_flat @ E_xbar_delta @ inv_matrix
                delta_w = delta_w.view_as(W)
                delta_weights[layer_name] = delta_w
                print(f"[{layer_name}] delta_w shape: {delta_w.shape}")

            elif isinstance(layer_module, torch.nn.Linear):
                print(f"Processing Linear layer: {layer_name}")
                out_dim, in_dim = W.shape
                W_flat = W.view(out_dim, in_dim)

                x_all = torch.cat(layer_data["x"], dim=0).cpu()
                x_bar_all = torch.cat(layer_data["x_bar"], dim=0).cpu()

                x_all_flat = x_all.view(x_all.shape[0], -1)
                x_bar_all_flat = x_bar_all.view(x_bar_all.shape[0], -1)

                if x_all_flat.shape[1] != in_dim:
                    print(f"[{layer_name}] Shape mismatch: {x_all_flat.shape[1]} vs {in_dim}, skipping.")
                    continue

                delta_x = x_all_flat - x_bar_all_flat

                E_xbar_xbar = (x_bar_all_flat.T @ x_bar_all_flat) / x_bar_all_flat.shape[0]
                E_xbar_delta = (x_bar_all_flat.T @ delta_x) / x_bar_all_flat.shape[0]

                I = torch.eye(in_dim)

                try:
                    inv_matrix = torch.inverse(E_xbar_xbar + lambda1 * I)
                except Exception as e:
                    print(f"[{layer_name}] Matrix inversion failed: {e}")
                    continue

                delta_w = - W_flat @ E_xbar_delta @ inv_matrix
                delta_w = delta_w.view_as(W)
                delta_weights[layer_name] = delta_w
                print(f"[{layer_name}] delta_w shape: {delta_w.shape}")
            else:
                print(f"[{layer_name}] Unsupported layer type, skipping.")

        return delta_weights

    def calibrate(self, model, cali_data_path):
        """
        使用校准集计算每一层的激活统计信息
        :param model: 用于校准的模型
        :param cali_data_path: 校准集路径
        """
        logger.info(f"加载校准集: {cali_data_path}")

        try:
            # 加载原始校准数据
            # logger.info(f"从 {self.args.cali_st} 时间步采样数据进行校准")
            sample_data = torch.load(cali_data_path)

            # 使用get_train_samples函数处理数据
            cali_data = get_train_samples(self.args, sample_data, custom_steps=0)
            del sample_data
            gc.collect()

            # 检查格式并解包数据
            if not isinstance(cali_data, tuple) or len(cali_data) != 2:
                raise ValueError(f"校准数据集格式错误: 期望元组(图像, 时间步)，实际得到: {type(cali_data)}")

            # 直接从元组中解包图像和时间步
            cali_xs, cali_ts = cali_data
            logger.info(f"校准数据形状: 图像={cali_xs.shape}, 时间步={cali_ts.shape}")

            num_samples = min(100, cali_xs.shape[0])
            indices = torch.randperm(cali_xs.shape[0])[:num_samples]
            cali_xs = cali_xs[indices]
            cali_ts = cali_ts[indices]
            # 确保时间步是1维向量
            tensor_t = cali_ts.squeeze()
            tensor_x = cali_xs

        except Exception as e:
            logger.error(f"无法加载校准集: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

        # 注册激活钩子
        register_activation_hooks(model, self.activation_dict)
        logger.info(f"已为模型注册激活钩子")

        # 创建数据加载器
        dataset = TensorDataset(tensor_x, tensor_t)
        cali_loader = DataLoader(
            dataset,
            batch_size=self.args.cali_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        # 开始校准过程
        model.eval()
        logger.info(f"开始校准，批大小: {self.args.cali_batch_size}, 总样本数: {len(tensor_x)}")

        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(cali_loader), desc="校准进度")

            for batch_x, batch_t in cali_loader:
                # 移动数据到设备
                batch_x = batch_x.to(self.device)
                batch_t = batch_t.to(self.device)

                # 执行前向传播（数据将通过钩子记录）
                model(batch_x, batch_t)

                pbar.update(1)

            pbar.close()

        logger.info("校准数据收集完成")

        # 计算并保存结果
        stats = self.compute_layer_stats()
        delta_w_stats = self.compute_delta_weights(model, lambda1=1e4)

        # 合并保存到同一个 json 中
        self.save_calibration_stats(stats, cali_data_path, delta_w_stats)

        # 另外也单独保存 .pt 版（可选）
        self.save_delta_weights(delta_w_stats)

        return stats

    def save_delta_weights(self, delta_weights):
        save_path = os.path.join(self.args.logdir, "delta_weights.pt")
        torch.save(delta_weights, save_path)
        logger.info(f"δW* 保存至 {save_path}")

    def compute_layer_stats(self):
        """计算每层的统计信息"""
        layer_stats = {}

        # 遍历所有收集到的层
        for layer_name, layer_data in self.activation_dict.items():
            if "max_values" not in layer_data or not layer_data["max_values"]:  # 跳过空列表
                continue

            # 拼接所有最大值
            all_max_vals = np.concatenate(layer_data["max_values"])

            # 计算均值
            mean_max_val = np.mean(all_max_vals)

            # 收集统计信息
            layer_stats[layer_name] = {
                "mean_activation_max": float(mean_max_val),
                # "min_value": float(np.min(all_max_vals)),
                # "max_value": float(np.max(all_max_vals)),
                # "n_samples": len(all_max_vals)
            }

        return layer_stats

    def save_calibration_stats(self, stats, cali_data_path, delta_weights=None):
        """保存校准统计信息和 δW* 到 JSON 文件"""
        if not stats:
            logger.warning("无统计信息可保存")
            return

        # 构建最终输出数据结构
        output_data = {
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layers": {}
        }

        for layer_name in stats:
            output_data["layers"][layer_name] = stats[layer_name]

            # 如果 δW* 可用，则添加其 shape 和部分值
            if delta_weights and layer_name in delta_weights:
                delta = delta_weights[layer_name]
                delta_np = delta.cpu().numpy()
                output_data["layers"][layer_name]["delta_w_shape"] = list(delta_np.shape)

                # 截断以防太大（你也可以换成均值/std之类）
                preview_len = min(5, delta_np.shape[0])
                output_data["layers"][layer_name]["delta_w_preview"] = delta_np[:preview_len].tolist()

        stats_path = os.path.join(self.args.logdir, "calibration_stats.json")

        with open(stats_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"校准统计和 δW* 已保存至 {stats_path}")

    def sample(self):
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
        default="uniform",
        help="skip according to (uniform or quadratic)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
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
    # logger.info(f"Host {os.uname()[1]}")
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

        # 检查校准集路径
        cali_path = "cifar_sd1236_sample2048_allst.pt"
        if not os.path.exists(cali_path):
            logger.error(f"校准集路径不存在: {cali_path}")
            sys.exit(1)

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

        # 执行校准
        stats = runner.calibrate(model, cali_path)

        if stats:
            logger.info(f"校准完成! 共收集 {len(stats)} 层的统计信息")
        else:
            logger.error("校准失败")
    else:
        logger.info("=== SAMPLING MODE ===")
        runner.sample()
