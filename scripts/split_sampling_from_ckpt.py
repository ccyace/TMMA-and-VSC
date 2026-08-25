"""
用已有量化 ckpt.pth 做分段 DDIM 推理（同一模型，不重新 BRECQ）。

输出路径规则（默认）:
  ckpt.pth 所在目录/
    ├── ckpt.pth
    ├── intermediate_noise_images.pt   # 前半段中间态
    ├── split_sampling.log
    └── images/                        # 后半段 PNG（默认写这里）
        ├── 0.png
        └── ...

若 ckpt 同级已有 images/:
  - 默认 --resume_images : 继续往同一 images/ 写入，从当前最大编号 +1 续跑
  - --fresh_images       : 新建 images_split_{时间戳}/，避免覆盖旧图
  - --image_folder PATH  : 手动指定输出目录（优先级最高）

示例:
  python scripts/split_sampling_from_ckpt.py \\
    --config configs/cifar10.yml \\
    --cali_ckpt two_quant_output_w4a8/samples/2025-12-02-11-26-18/ckpt.pth \\
    --weight_bit 4 --act_bit 8 --quant_act --a_sym --split \\
    --max_images 50000
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
import torchvision.utils as tvu
import yaml
from pytorch_lightning import seed_everything
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddim.datasets import inverse_data_transform
from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import compute_alpha, generalized_steps
from ddim.models.diffusion import Model
from qdiff import QuantModel
from qdiff.utils import resume_cali_model

logger = logging.getLogger(__name__)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64)
            ** 2
        )
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def build_ddim_seq(num_timesteps, ddim_steps, skip_type):
    if skip_type == "uniform":
        seq = list(range(0, num_timesteps, num_timesteps // ddim_steps))
        if len(seq) > ddim_steps:
            seq = seq[:ddim_steps]
    elif skip_type == "quad":
        seq = (
            np.linspace(0, np.sqrt(num_timesteps * 0.8), ddim_steps) ** 2
        )
        seq = [int(s) for s in list(seq)]
    else:
        raise ValueError(f"不支持的 skip_type: {skip_type}")

    if len(seq) != ddim_steps:
        raise RuntimeError(f"DDIM 步数异常: {len(seq)} != {ddim_steps}")
    return seq


def generalized_steps_limited(x, seq, model, b, max_steps, **kwargs):
    """只跑完整 DDIM 序列的前 max_steps 步（从高噪声到低噪声）。"""
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            if step_idx >= max_steps:
                break
            t = (torch.ones(n) * i).to(x.device)
            if "hook_control_dict" in kwargs and kwargs["hook_control_dict"] is not None:
                kwargs["hook_control_dict"]["current_t"] = int(i)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.cpu())
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.cpu())
    return xs, x0_preds


def build_split_plan(num_timesteps, ddim_steps, skip_type):
    """
    正确切分 DDIM 100 步为前后各 50 步:
      - 前半: 完整序列的前 50 次迭代，停在 t≈204（不是 t=-1）
      - 后半: 从中间态继续，用序列前 50 个 timestep 做剩余 50 次迭代
    """
    full_seq = build_ddim_seq(num_timesteps, ddim_steps, skip_type)
    split_idx = ddim_steps // 2
    second_half_seq = full_seq[:split_idx]
    reverse_seq = list(reversed(full_seq))
    return {
        "full_seq": full_seq,
        "split_idx": split_idx,
        "second_half_seq": second_half_seq,
        "first_start_t": reverse_seq[0],
        "first_end_t": reverse_seq[split_idx],
        "second_start_t": reverse_seq[split_idx],
        "second_end_t": reverse_seq[-1],
    }


def resolve_paths(args):
    args.cali_ckpt = os.path.abspath(args.cali_ckpt)
    if not os.path.isfile(args.cali_ckpt):
        raise FileNotFoundError(
            f"找不到 ckpt 文件: {args.cali_ckpt}\n"
            "请传入真实路径，例如:\n"
            "  --cali_ckpt two_quant_output_w4a8/samples/2025-12-02-11-26-18/ckpt.pth"
        )

    ckpt_dir = os.path.dirname(args.cali_ckpt)
    args.logdir = ckpt_dir

    if args.intermediate_path is None:
        args.intermediate_path = os.path.join(ckpt_dir, "intermediate_noise_images.pt")

    default_images = os.path.join(ckpt_dir, "images")
    if args.image_folder:
        args.image_folder = os.path.abspath(args.image_folder)
    elif args.fresh_images:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.image_folder = os.path.join(ckpt_dir, f"images_split_{stamp}")
    else:
        args.image_folder = default_images

    os.makedirs(args.image_folder, exist_ok=True)
    return args


def next_image_index(image_folder):
    """返回 images/ 中已有 PNG 的最大编号 +1；无图则从 0 开始。"""
    pattern = re.compile(r"^(\d+)\.png$")
    max_id = -1
    for name in os.listdir(image_folder):
        m = pattern.match(name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def load_quantized_model(config, args, device):
    model = Model(config)
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    elif config.data.dataset == "LSUN":
        name = f"lsun_{config.data.category}"
    else:
        raise ValueError(f"不支持的数据集: {config.data.dataset}")

    fp_ckpt = get_ckpt_path(f"ema_{name}")
    logger.info("Loading FP checkpoint: %s", fp_ckpt)
    model.load_state_dict(torch.load(fp_ckpt, map_location=device))
    model.to(device)
    model.eval()

    wq_params = {"n_bits": args.weight_bit, "channel_wise": True, "scale_method": "max"}
    aq_params = {
        "n_bits": args.act_bit,
        "symmetric": args.a_sym,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": args.quant_act,
    }
    qnn = QuantModel(
        model=model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        sm_abit=args.sm_abit,
    )
    qnn.to(device)
    qnn.eval()

    cali_data = (
        torch.randn(1, config.data.channels, config.data.image_size, config.data.image_size),
        torch.randint(0, 1000, (1,)),
    )
    logger.info("Loading quantized ckpt: %s", args.cali_ckpt)
    resume_cali_model(qnn, args.cali_ckpt, cali_data, args.quant_act, "qdiff", cond=False)
    qnn.eval()
    return qnn


def run_partial_sampling(model, start_x, seq, betas, batch_size, device, eta, desc, max_steps=None):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, start_x.shape[0], batch_size), desc=desc):
            x = start_x[i:i + batch_size].to(device)
            if max_steps is None:
                xs, _ = generalized_steps(x, seq, model, betas, eta=eta)
            else:
                xs, _ = generalized_steps_limited(x, seq, model, betas, max_steps, eta=eta)
            outputs.append(xs[-1].cpu())
    return torch.cat(outputs, dim=0)


def run_first_half(qnn, args, config, betas, split_plan, device):
    if os.path.exists(args.intermediate_path) and not args.overwrite_intermediate:
        logger.info("中间态已存在，跳过前半段: %s", args.intermediate_path)
        logger.info("若上次分段采样失真，请加 --overwrite_intermediate 重新生成中间态")
        return torch.load(args.intermediate_path, map_location="cpu")

    channels = config.data.channels
    size = config.data.image_size
    all_noise = torch.randn(args.max_images, channels, size, size)
    intermediate = run_partial_sampling(
        qnn,
        all_noise,
        split_plan["full_seq"],
        betas,
        args.batch_size,
        device,
        args.eta,
        "前半段采样",
        max_steps=split_plan["split_idx"],
    )
    torch.save(intermediate, args.intermediate_path)
    logger.info("已保存中间态: %s", args.intermediate_path)
    return intermediate


def run_second_half(qnn, args, config, betas, split_plan, device, intermediate):
    start_id = 0 if args.fresh_images else next_image_index(args.image_folder)
    if start_id > 0:
        logger.info("检测到已有 PNG，从编号 %d 续写 -> %s", start_id, args.image_folder)

    total = min(intermediate.shape[0], args.max_images)
    if start_id >= total:
        logger.info("已有图像数 %d >= max_images %d，跳过后半段采样", start_id, total)
        return

    logger.info("后半段 PNG 输出目录: %s", args.image_folder)
    logger.info("本次生成编号范围: %d ~ %d", start_id, total - 1)

    with torch.no_grad():
        for i in tqdm(range(start_id, total, args.batch_size), desc="后半段采样"):
            x = intermediate[i:i + args.batch_size]
            xs, _ = generalized_steps(
                x.to(device), split_plan["second_half_seq"], qnn, betas, eta=args.eta
            )
            batch_out = inverse_data_transform(config, xs[-1].cpu())
            n = min(batch_out.shape[0], total - i)
            for j in range(n):
                tvu.save_image(
                    batch_out[j],
                    os.path.join(args.image_folder, f"{i + j}.png"),
                )

    logger.info("后半段完成，共写入 %d 张 PNG -> %s", total - start_id, args.image_folder)


def parse_args():
    parser = argparse.ArgumentParser(description="用已有 ckpt 做分段 DDIM 推理")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cali_ckpt", type=str, required=True, help="已校准量化模型 ckpt.pth")
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)
    parser.add_argument("--quant_act", action="store_true")
    parser.add_argument("--a_sym", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--sm_abit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_images", type=int, default=50000)
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--skip_type", type=str, default="quad", choices=["quad", "uniform"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--intermediate_path", type=str, default=None)
    parser.add_argument("--overwrite_intermediate", action="store_true")
    parser.add_argument("--first_half_only", action="store_true")
    parser.add_argument("--second_half_only", action="store_true")
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="手动指定 PNG 输出目录；默认 ckpt 同级 images/",
    )
    parser.add_argument(
        "--fresh_images",
        action="store_true",
        help="ckpt 下已有 images/ 时，改写到 images_split_{时间戳}/，不覆盖旧图",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args = resolve_paths(args)

    log_path = os.path.join(args.logdir, "split_sampling.log")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )

    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))

    config.split_shortcut = args.split

    if args.batch_size is None:
        args.batch_size = config.sampling.batch_size

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    betas = torch.from_numpy(
        get_beta_schedule(
            config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
    ).float().to(device)

    split_plan = build_split_plan(
        config.diffusion.num_diffusion_timesteps, args.ddim_steps, args.skip_type
    )

    logger.info("=" * 60)
    logger.info("ckpt              : %s", args.cali_ckpt)
    logger.info("intermediate      : %s", args.intermediate_path)
    logger.info("image output      : %s", args.image_folder)
    if args.fresh_images:
        logger.info("image mode        : fresh (images_split_*)")
    elif os.path.basename(args.image_folder) == "images" and os.path.dirname(args.image_folder) == os.path.dirname(os.path.abspath(args.cali_ckpt)):
        logger.info("image mode        : resume (append to ckpt/images/)")
    else:
        logger.info("image mode        : custom folder")
    logger.info(
        "DDIM %d-step(%s): 前半 %d步(%d->%d), 后半 %d步(%d->%d)",
        args.ddim_steps, args.skip_type,
        split_plan["split_idx"], split_plan["first_start_t"], split_plan["first_end_t"],
        split_plan["split_idx"], split_plan["second_start_t"], split_plan["second_end_t"],
    )
    logger.info("=" * 60)

    qnn = load_quantized_model(config, args, device)

    if args.second_half_only:
        if not os.path.exists(args.intermediate_path):
            raise FileNotFoundError(f"未找到中间态: {args.intermediate_path}")
        intermediate = torch.load(args.intermediate_path, map_location="cpu")
        run_second_half(qnn, args, config, betas, split_plan, device, intermediate)
        return

    intermediate = run_first_half(qnn, args, config, betas, split_plan, device)
    if args.first_half_only:
        logger.info("first_half_only 完成")
        return

    run_second_half(qnn, args, config, betas, split_plan, device, intermediate)


if __name__ == "__main__":
    main()
