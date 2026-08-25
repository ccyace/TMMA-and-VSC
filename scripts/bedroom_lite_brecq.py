"""
Bedroom 256 专用 ultra-lite 权重 BRECQ（防 OOM / 假死）。

为何 sample_diffusion_ddim + cali_n=64 仍会卡死？
  BRECQ 每层会把「全部校准样本」在该层的输入/输出特征 cat 起来缓存。
  Bedroom 256 早期层特征约 [N, 128, 256, 256]：
    N=1280 → 约 40GB+ RAM，Windows 开始疯狂换页 → 像死机。
  CIFAR 32×32 同逻辑只需 ~600MB，所以论文默认 cali_n=256 在 256 分辨率不适用。

本脚本默认：cali_st=4 × cali_n=4 = **仅 16 条**校准对，cali_iters=300，batch=1。
质量低于完整 BRECQ，但通常仍明显好于 simple min-max。

示例：
  python scripts/bedroom_lite_brecq.py ^
    --cali_data_path lsun_bedroom_ddim_sd1236_sample256_allst.pt ^
    --quantize_only -l bedroom_w8_lite_brecq

  # 仍 OOM 时用极简档
  python scripts/bedroom_lite_brecq.py ^
    --cali_st 2 --cali_n 2 --cali_iters 150 --max_cali_total 4 ^
    --quantize_only -l bedroom_w8_micro_brecq
"""

from __future__ import annotations

import argparse
import datetime
import gc
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import seed_everything

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ddim.functions.ckpt_util import get_ckpt_path
from ddim.models.diffusion import Model
from qdiff import (
    BaseQuantBlock,
    QuantModel,
    QuantModule,
    block_reconstruction,
    layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import get_train_samples

logger = logging.getLogger(__name__)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def parse_args():
    p = argparse.ArgumentParser(description="Bedroom 256 ultra-lite 权重 BRECQ")
    p.add_argument("--config", type=str, default="configs/lsun_bedroom256.yml")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("-l", "--logdir", type=str, default="bedroom_w8_lite_brecq")
    p.add_argument("--weight_bit", type=int, default=8, choices=[4, 8])
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--no_split", action="store_true")
    p.add_argument(
        "--cali_data_path",
        type=str,
        default="lsun_bedroom_ddim_sd1236_sample256_allst.pt",
    )
    p.add_argument(
        "--cali_st", type=int, default=4,
        help="从 allst 均匀抽几个时间步（默认 4）",
    )
    p.add_argument(
        "--cali_n", type=int, default=4,
        help="每个时间步取几张图（默认 4；总样本≈cali_st×cali_n）",
    )
    p.add_argument(
        "--max_cali_total", type=int, default=16,
        help="校准对上限；超过则随机下采样（256 分辨率建议 ≤16）",
    )
    p.add_argument("--cali_iters", type=int, default=300, help="每层/块 BRECQ 迭代")
    p.add_argument("--cali_batch_size", type=int, default=1)
    p.add_argument(
        "--quantize_only",
        action="store_true",
        help="只量化并保存 ckpt（本脚本默认行为，加此参数仅为与其它脚本命令一致）",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_fp_model(config, device):
    model = Model(config)
    ckpt = get_ckpt_path(f"ema_lsun_{config.data.category}")
    logger.info("Loading FP32: %s", ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.to(device)
    model.eval()
    return model


def build_qnn(fp_model, weight_bit, device):
    wq = {"n_bits": weight_bit, "channel_wise": True, "scale_method": "max"}
    aq = {
        "n_bits": 8, "symmetric": True, "channel_wise": False,
        "scale_method": "max", "leaf_param": False,
    }
    qnn = QuantModel(model=fp_model, weight_quant_params=wq, act_quant_params=aq, sm_abit=8)
    qnn.to(device)
    qnn.eval()
    return qnn


def load_cali(args):
    path = args.cali_data_path
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    logger.info("加载校准集: %s", path)
    sample_data = torch.load(path, map_location="cpu")
    cali_xs, cali_ts = get_train_samples(args, sample_data, custom_steps=0)
    del sample_data
    gc.collect()
    n = cali_xs.shape[0]
    cap = int(args.max_cali_total)
    if cap > 0 and n > cap:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(n, size=cap, replace=False)
        cali_xs = cali_xs[idx]
        cali_ts = cali_ts[idx]
        logger.info("校准对下采样: %d -> %d", n, cap)
        n = cap
    if cali_xs.dtype != torch.float32:
        cali_xs = cali_xs.float()
    if cali_ts.dtype != torch.float32:
        cali_ts = cali_ts.float()
    logger.info("校准形状 xs=%s ts=%s", tuple(cali_xs.shape), tuple(cali_ts.shape))
    est_mb = cali_xs.numel() * 4 / (1024 ** 2)
    logger.info("校准 xs 内存约 %.1f MB（每层缓存同量级特征时会再乘通道因子）", est_mb)
    return cali_xs, cali_ts


def run_weight_brecq(qnn, cali_data, args, device):
    cali_xs, cali_ts = cali_data
    qnn.set_quant_state(True, False)
    with torch.no_grad():
        bs = min(8, cali_xs.shape[0])
        _ = qnn(cali_xs[:bs].to(device), cali_ts[:bs].to(device))
    logger.info("权重量化 scale 初始化完成")

    kwargs = dict(
        cali_data=cali_data,
        batch_size=args.cali_batch_size,
        iters=args.cali_iters,
        weight=0.01,
        asym=True,
        b_range=(20, 2),
        warmup=0.2,
        act_quant=False,
        opt_mode="mse",
    )

    layer_count = 0

    def recon_model(root, prefix=""):
        nonlocal layer_count
        for name, module in root.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction:
                    continue
                layer_count += 1
                logger.info("[%d] layer BRECQ: %s", layer_count, full_name)
                layer_reconstruction(qnn, module, **kwargs)
                torch.cuda.empty_cache()
                gc.collect()
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction:
                    continue
                layer_count += 1
                logger.info("[%d] block BRECQ: %s", layer_count, full_name)
                block_reconstruction(qnn, module, **kwargs)
                torch.cuda.empty_cache()
                gc.collect()
            else:
                recon_model(module, full_name)

    logger.info(
        "开始 ultra-lite 权重 BRECQ（iters=%d, cali=%d, batch=%d）…",
        args.cali_iters, cali_xs.shape[0], args.cali_batch_size,
    )
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    logger.info("BRECQ 完成，共处理 %d 个 layer/block", layer_count)


def save_ckpt(qnn, path):
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)
    torch.save(qnn.state_dict(), path)
    logger.info("已保存: %s", path)


def main():
    args = parse_args()
    args.cond = False
    args.custom_steps = 0
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.cali_st * args.cali_n > args.max_cali_total:
        logger.warning(
            "cali_st×cali_n=%d > max_cali_total=%d，加载后会被下采样",
            args.cali_st * args.cali_n, args.max_cali_total,
        )

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_REPO_ROOT, cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split and not args.no_split

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.logdir, "samples", now)
    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(logdir, "run.log")),
            logging.StreamHandler(),
        ],
    )

    fp = load_fp_model(config, device)
    qnn = build_qnn(fp, args.weight_bit, device)
    cali_xs, cali_ts = load_cali(args)
    run_weight_brecq(qnn, (cali_xs, cali_ts), args, device)
    save_ckpt(qnn, os.path.join(logdir, "ckpt.pth"))

    if not args.quantize_only:
        logger.info("采样请用 simple_weight_quant 或 sample_diffusion_ddim --resume --cali_ckpt ...")


if __name__ == "__main__":
    main()
