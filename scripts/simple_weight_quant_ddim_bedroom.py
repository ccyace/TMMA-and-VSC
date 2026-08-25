"""
LSUN Bedroom 256 像素 DDIM：轻量权重量化（min-max + 校准前向，无 BRECQ）+ 可选采样。

仅量化 UNet 权重（wN a32），激活保持 FP32。比 8 条随机样本 init 更接近 Q-Diffusion，
但仍低于完整 BRECQ。

示例：
  # 路线 2：用 allst 校准集 init scale（推荐）
  python scripts/simple_weight_quant_ddim_bedroom.py \\
    --weight_bit 8 --split --quantize_only \\
    --cali_data_path lsun_bedroom_ddim_sd1236_sample256_allst.pt \\
    --cali_st 20 --cali_n 64 --cali_batch_size 4 \\
    -l bedroom_w8a32_cali_init

  # 仅 8 条随机样本（质量最差，仅快速试）
  python scripts/simple_weight_quant_ddim_bedroom.py \\
    --weight_bit 8 --seed 41 --max_images 64 -l bedroom_w8a32

  # 从已保存 ckpt 直接采样（min-max ckpt）
  python scripts/simple_weight_quant_ddim_bedroom.py \\
    --quant_ckpt bedroom_w8a32_cali_init/samples/<时间戳>/ckpt.pth --max_images 64

  # 从 bedroom_lite_brecq 导出的 BRECQ ckpt 采样（自动走 AdaRound 加载）
  python scripts/simple_weight_quant_ddim_bedroom.py \\
    --quant_ckpt bedroom_w8_lite_brecq/samples/<时间戳>/ckpt.pth --max_images 10000
"""

from __future__ import annotations

import argparse
import datetime
import gc
import logging
import math
import os
import sys

import numpy as np
import torch
import torchvision.utils as tvu
import tqdm
import yaml
from pytorch_lightning import seed_everything

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ddim.datasets import inverse_data_transform
from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import generalized_steps
from ddim.models.diffusion import Model
from qdiff import QuantModel
from qdiff.utils import get_train_samples, resume_cali_model

logger = logging.getLogger(__name__)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def load_fp32_model(config, device, ckpt_path=""):
    model = Model(config)
    if ckpt_path:
        ckpt = ckpt_path
    elif str(config.data.dataset).upper() == "LSUN":
        ckpt = get_ckpt_path(f"ema_lsun_{config.data.category}")
    elif str(config.data.dataset).upper() == "CIFAR10":
        ckpt = get_ckpt_path("ema_cifar10")
    else:
        raise ValueError(f"不支持的数据集: {config.data.dataset}")
    logger.info("Loading FP32 checkpoint: %s", ckpt)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_simple_weight_qnn(fp_model, weight_bit, split_shortcut, device):
    wq_params = {
        "n_bits": weight_bit,
        "channel_wise": True,
        "scale_method": "max",
    }
    aq_params = {
        "n_bits": 8,
        "symmetric": True,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": False,
    }
    qnn = QuantModel(
        model=fp_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        sm_abit=8,
    )
    qnn.to(device)
    qnn.eval()
    return qnn


def load_simple_quant_ckpt(qnn, ckpt_path, device):
    """加载本脚本导出的 min-max ckpt（勿用 resume_cali_model，那会按 BRECQ/AdaRound 解析）。"""
    logger.info("Loading simple quant ckpt: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    qnn.load_state_dict(state, strict=False)
    qnn.set_quant_state(True, False)
    qnn.eval()


def is_brecq_ckpt(ckpt_path):
    """BRECQ 保存的 ckpt 含 weight_quantizer.alpha（AdaRound 舍入参数）。"""
    state = torch.load(ckpt_path, map_location="cpu")
    return any(k.endswith("weight_quantizer.alpha") for k in state)


def make_minimal_cali_data(config, device):
    c = config.data.channels
    h = config.data.image_size
    xs = torch.randn(1, c, h, h, device=device)
    ts = torch.tensor([500.0], device=device)
    return xs, ts


def load_brecq_quant_ckpt(qnn, ckpt_path, cali_data, device):
    """加载 bedroom_lite_brecq / sample_diffusion_ddim 导出的 BRECQ ckpt。"""
    logger.info("Loading BRECQ ckpt (AdaRound): %s", ckpt_path)
    resume_cali_model(
        qnn, ckpt_path, cali_data,
        quant_act=False, act_quant_mode="qdiff", cond=False,
    )
    qnn.eval()


def load_quant_ckpt(qnn, ckpt_path, args, config, device):
    if is_brecq_ckpt(ckpt_path):
        if args.cali_data_path.strip():
            path = args.cali_data_path.strip()
            if not os.path.isabs(path):
                path = os.path.join(_REPO_ROOT, path)
            sample_data = torch.load(path, map_location="cpu")
            cali_xs, cali_ts = get_train_samples(args, sample_data, custom_steps=0)
            del sample_data
            gc.collect()
            cali_data = (cali_xs[:1].to(device), cali_ts[:1].to(device))
        else:
            xs, ts = make_minimal_cali_data(config, device)
            cali_data = (xs, ts)
        load_brecq_quant_ckpt(qnn, ckpt_path, cali_data, device)
        return "brecq"
    load_simple_quant_ckpt(qnn, ckpt_path, device)
    return "simple"


def init_weight_quant(qnn, config, device, n_init=8):
    c = config.data.channels
    h = config.data.image_size
    xs = torch.randn(n_init, c, h, h, device=device)
    ts = torch.randint(0, 1000, (n_init,), device=device, dtype=torch.float32)
    qnn.set_quant_state(True, False)
    with torch.no_grad():
        _ = qnn(xs, ts)
    logger.info("Weight min-max 初始化完成（%d 条随机样本）", n_init)


def init_weight_quant_from_cali(qnn, args, device):
    """
    路线 2：用真实 (x_t, t) 校准集做多 batch 前向，更新权重量化 scale（无 BRECQ）。
    allst 格式 xs[T,N,C,H,W]、ts[T,N] 须设 cali_st>1（如 20）。
    """
    path = args.cali_data_path.strip()
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"校准集不存在: {path}")

    logger.info("加载校准集: %s", path)
    sample_data = torch.load(path, map_location="cpu")
    cali_xs, cali_ts = get_train_samples(args, sample_data, custom_steps=0)
    del sample_data
    gc.collect()

    if cali_xs.dtype != torch.float32:
        cali_xs = cali_xs.float()
    if cali_ts.dtype != torch.float32:
        cali_ts = cali_ts.float()

    n = cali_xs.shape[0]
    bs = max(1, int(args.cali_batch_size))
    logger.info(
        "校准 init：共 %d 对 (x,t)，batch=%d，cali_st=%d cali_n=%d",
        n, bs, args.cali_st, args.cali_n,
    )

    qnn.set_quant_state(True, False)
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, n, bs), desc="cali weight init"):
            x = cali_xs[i : i + bs].to(device)
            t = cali_ts[i : i + bs].to(device)
            _ = qnn(x, t)
    logger.info("Weight min-max 初始化完成（校准集 %d 前向，无 BRECQ）", n)


def ddim_sample_batch(model, betas, batch_size, config, device, timesteps, skip_type, eta):
    x = torch.randn(
        batch_size,
        config.data.channels,
        config.data.image_size,
        config.data.image_size,
        device=device,
    )
    num_train = betas.shape[0]
    if skip_type == "uniform":
        skip = num_train // timesteps
        seq = list(range(0, num_train, skip))
    elif skip_type == "quad":
        seq = [
            int(s)
            for s in np.linspace(0, np.sqrt(num_train * 0.8), timesteps) ** 2
        ]
    else:
        raise ValueError(skip_type)

    betas = betas.to(device)
    with torch.no_grad():
        xs, _ = generalized_steps(x, seq, model, betas, eta=eta)
        x = xs[-1].to(device)
    return inverse_data_transform(config, x)


def run_sampling(model, betas, args, config, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    batch_size = config.sampling.batch_size
    n_rounds = math.ceil(args.max_images / batch_size)
    img_id = 0
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for _ in tqdm.tqdm(range(n_rounds), desc="DDIM sampling"):
        n = min(batch_size, args.max_images - img_id)
        if n <= 0:
            break
        x = ddim_sample_batch(
            model, betas, n, config, device,
            args.timesteps, args.skip_type, args.eta,
        )
        for i in range(min(n, x.shape[0])):
            tvu.save_image(x[i], os.path.join(out_dir, f"{img_id}.png"))
            img_id += 1
    logger.info("Saved %d images to %s", img_id, out_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="LSUN Bedroom DDIM 简单权重量化（无 BRECQ）",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/lsun_bedroom256.yml",
        help="默认 lsun_bedroom256",
    )
    p.add_argument("--seed", type=int, default=41)
    p.add_argument(
        "-l", "--logdir", type=str, default="bedroom_ddim_simple_wq",
        help="输出根目录",
    )
    p.add_argument(
        "--weight_bit", type=int, default=8, choices=[4, 8],
        help="权重量化 bit 数",
    )
    p.add_argument(
        "--split", action="store_true", default=True,
        help="split shortcut（Q-Diffusion 默认开启）",
    )
    p.add_argument(
        "--no_split", action="store_true",
        help="关闭 split shortcut",
    )
    p.add_argument(
        "--fp32_ckpt", type=str, default="",
        help="可选 FP32 ckpt；默认 get_ckpt_path(ema_lsun_bedroom)",
    )
    p.add_argument(
        "--quant_ckpt", type=str, default="",
        help="已保存的量化 ckpt；指定则跳过权重量化，直接采样",
    )
    p.add_argument(
        "--quantize_only", action="store_true",
        help="仅量化并保存 ckpt，不采样",
    )
    p.add_argument(
        "--cali_data_path", type=str, default="",
        help="allst 或扁平 {xs,ts} 校准 .pt；指定则用路线 2（真实 x_t,t 前向 init）",
    )
    p.add_argument(
        "--cali_st", type=int, default=20,
        help="从 allst 选取的时间步子步数；allst 须 >1（100 步 DDIM 常用 20）",
    )
    p.add_argument(
        "--cali_n", type=int, default=64,
        help="每个选中时间步最多取 N 张图；总前向数 ≈ cali_st × cali_n",
    )
    p.add_argument(
        "--cali_batch_size", type=int, default=4,
        help="校准前向 batch（Bedroom 256 建议 2～4，防 OOM）",
    )
    p.add_argument(
        "--n_init", type=int, default=8,
        help="未指定 cali_data_path 时，随机样本条数",
    )
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument(
        "--skip_type", type=str, default="quad", choices=["quad", "uniform"],
    )
    p.add_argument("--max_images", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，使用 CPU")

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_REPO_ROOT, cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split and not args.no_split

    betas = torch.from_numpy(
        get_beta_schedule(
            config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
    ).float().to(device)

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.logdir, "samples", now)
    os.makedirs(logdir, exist_ok=True)
    img_dir = os.path.join(logdir, "img")
    os.makedirs(img_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(logdir, "run.log")),
            logging.StreamHandler(),
        ],
    )
    logger.info("logdir: %s", logdir)

    args.cond = False
    args.custom_steps = 0

    fp_model = load_fp32_model(config, device, args.fp32_ckpt.strip())
    qnn = build_simple_weight_qnn(
        fp_model, args.weight_bit, args.split, device,
    )

    ckpt_mode = None
    if args.quant_ckpt.strip():
        ckpt_mode = load_quant_ckpt(
            qnn, args.quant_ckpt.strip(), args, config, device,
        )
    else:
        if args.cali_data_path.strip():
            init_weight_quant_from_cali(qnn, args, device)
        else:
            init_weight_quant(qnn, config, device, n_init=args.n_init)
        ckpt_path = os.path.join(logdir, "ckpt.pth")
        torch.save(qnn.state_dict(), ckpt_path)
        logger.info("Saved quantized ckpt: %s", ckpt_path)

    if ckpt_mode == "brecq":
        logger.info("mode: w%d a32 (BRECQ AdaRound ckpt)", args.weight_bit)
    elif ckpt_mode == "simple":
        logger.info("mode: w%d a32 (loaded min-max ckpt)", args.weight_bit)
    elif args.cali_data_path.strip():
        logger.info(
            "mode: w%d a32 (cali-init min-max, no BRECQ)", args.weight_bit,
        )
    else:
        logger.info(
            "mode: w%d a32 (random min-max, no BRECQ)", args.weight_bit,
        )

    if args.quantize_only:
        logger.info("quantize_only：跳过采样")
        return

    qnn.set_quant_state(True, False)
    run_sampling(qnn, betas, args, config, device, img_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
