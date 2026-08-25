import argparse
import datetime
import glob
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn
import torchvision.utils as tvu
import tqdm
import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MQBENCH_PY_ROOT = PROJECT_ROOT / "MQBench-main" / "MQBench-main"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MQBENCH_PY_ROOT))

from ddim.datasets import inverse_data_transform  # noqa: E402
from ddim.functions.ckpt_util import get_ckpt_path  # noqa: E402
from ddim.functions.denoising import generalized_steps  # noqa: E402
from ddim.models.diffusion import Model  # noqa: E402

from mqbench.prepare_by_platform import BackendType, prepare_by_platform  # noqa: E402
from mqbench.utils.state import enable_quantization  # noqa: E402


def dict2namespace(config):
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ns = dict2namespace(raw)
    if not hasattr(ns, "split_shortcut"):
        ns.split_shortcut = False
    return ns


def get_default_ema_ckpt_name(config):
    ds = str(config.data.dataset).upper()
    if ds == "CIFAR10":
        return "ema_cifar10"
    if ds == "LSUN":
        return f"ema_lsun_{config.data.category}"
    raise ValueError(f"Unsupported dataset: {config.data.dataset}")


def get_beta_schedule(beta_schedule: str, *, beta_start: float, beta_end: float, num_diffusion_timesteps: int):
    if beta_schedule == "quad":
        betas = (
            torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=torch.float64) ** 2
        )
    elif beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64)
    elif beta_schedule == "const":
        betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=torch.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / torch.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=torch.float64)
    else:
        raise NotImplementedError(beta_schedule)
    return betas.float()


def build_seq(total_timesteps: int, sample_steps: int, skip_type: str):
    if skip_type == "uniform":
        skip = max(1, total_timesteps // sample_steps)
        return [int(s) for s in range(0, total_timesteps, skip)]
    seq = torch.linspace(0, math.sqrt(total_timesteps * 0.8), sample_steps).pow(2.0)
    return [int(s.item()) for s in seq]


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images with MQBench LSQ-QAT UNet")
    parser.add_argument("--config", type=str, default="configs/cifar10.yml")
    parser.add_argument("--qat_ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)
    parser.add_argument("--a_sym", action="store_true")
    parser.add_argument("--per_channel_w", action="store_true", default=True)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--skip_type", type=str, default="quad", choices=["uniform", "quad"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_images", type=int, default=10000)
    parser.add_argument("--outdir", type=str, default="MQBench-main/lsq_qat_samples")
    return parser.parse_args()


class DiffusionUNetWrapper(nn.Module):
    """Keep the same wrapper structure as training script for key compatibility."""

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
        self.resolution = getattr(unet, "resolution", None)

    def forward(self, x, t=None, context=None):
        if isinstance(x, torch.Tensor) and self.resolution is not None and x.dim() == 4:
            assert x.shape[2] == x.shape[3]
        return self.unet(x, t, context)


def _load_mqbench_state_with_prefix_fallback(model: nn.Module, ckpt_state: dict):
    """Load state_dict robustly for possible 'unet.' prefix mismatch."""
    model_state = model.state_dict()

    def _try_load(state):
        return model.load_state_dict(state, strict=False)

    missing, unexpected = _try_load(ckpt_state)
    # If too many keys miss, try prefix conversion.
    if len(missing) > 0 and len(unexpected) > 0:
        remapped = {}
        for k, v in ckpt_state.items():
            if k.startswith("unet."):
                remapped[k[len("unet."):]] = v
            else:
                remapped[f"unet.{k}"] = v
        missing2, unexpected2 = _try_load(remapped)
        # Keep the better one (fewer unresolved keys).
        if len(missing2) + len(unexpected2) < len(missing) + len(unexpected):
            missing, unexpected = missing2, unexpected2

    print(f"[Info] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    # Fail fast if load is clearly wrong (to avoid snowflake from random quant params).
    if len(missing) > 200:
        raise RuntimeError(
            f"Too many missing keys while loading QAT checkpoint ({len(missing)}). "
            "Likely model structure/key prefix mismatch."
        )


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    config.split_shortcut = bool(args.split)

    # Build fp32 UNet and load pretrained diffusion checkpoint.
    unet = Model(config).to("cpu").eval()
    ckpt_path = get_ckpt_path(get_default_ema_ckpt_name(config))
    unet.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    wrapped = DiffusionUNetWrapper(unet).to("cpu").eval()

    qat_obj = torch.load(args.qat_ckpt, map_location="cpu")
    # Prefer checkpoint metadata when available to avoid mismatch.
    wbit = int(qat_obj.get("weight_bit", args.weight_bit))
    abit = int(qat_obj.get("act_bit", args.act_bit))
    a_sym = bool(qat_obj.get("a_sym", args.a_sym))
    split_from_ckpt = bool(qat_obj.get("split", args.split))
    config.split_shortcut = split_from_ckpt

    # Prepare MQBench model with same qconfig used during training.
    extra_qconfig_dict = {
        "w_observer": "LSQObserver",
        "a_observer": "LSQObserver",
        "w_fakequantize": "LearnableFakeQuantize",
        "a_fakequantize": "LearnableFakeQuantize",
        "w_qscheme": {
            "symmetry": True,
            "per_channel": bool(args.per_channel_w),
            "pot_scale": False,
            "bit": wbit,
        },
        "a_qscheme": {
            "symmetry": a_sym,
            "per_channel": False,
            "pot_scale": False,
            "bit": abit,
        },
    }
    prepare_custom_config_dict = {"extra_qconfig_dict": extra_qconfig_dict}
    q_model = prepare_by_platform(wrapped, BackendType.Academic, prepare_custom_config_dict, is_qat=True)
    _load_mqbench_state_with_prefix_fallback(q_model, qat_obj["state_dict"])
    q_model.to(device).eval()
    enable_quantization(q_model)

    # Build diffusion schedule.
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    ).to(device)
    seq = build_seq(total_timesteps=betas.shape[0], sample_steps=args.timesteps, skip_type=args.skip_type)

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = Path(args.outdir) / now / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_id = len(glob.glob(str(out_dir / "*.png")))
    n_rounds = math.ceil((args.max_images - img_id) / args.batch_size)

    with torch.no_grad():
        for _ in tqdm.trange(n_rounds, desc="Sampling MQBench LSQ-QAT"):
            n = args.batch_size
            x = torch.randn(
                n,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=device,
            )
            xs = generalized_steps(x, seq, q_model, betas, eta=args.eta)
            x = inverse_data_transform(config, xs[0][-1])

            if img_id + x.shape[0] > args.max_images:
                n = args.max_images - img_id
            for i in range(n):
                tvu.save_image(x[i], out_dir / f"{img_id}.png")
                img_id += 1

    print(f"[Done] saved {img_id} images -> {out_dir}")


if __name__ == "__main__":
    main(parse_args())

