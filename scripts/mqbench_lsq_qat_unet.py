import argparse
import copy
import datetime
import logging
import math
import os
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# ----------------------------
# Path setup
# ----------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MQBENCH_PY_ROOT = PROJECT_ROOT / "MQBench-main" / "MQBench-main"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MQBENCH_PY_ROOT))

from ddim.datasets import data_transform  # noqa: E402
from ddim.functions.ckpt_util import get_ckpt_path  # noqa: E402
from ddim.models.diffusion import Model  # noqa: E402

from mqbench.prepare_by_platform import BackendType, prepare_by_platform  # noqa: E402
from mqbench.utils.state import enable_calibration, enable_quantization  # noqa: E402

from ddim.functions.denoising import generalized_steps  # noqa: E402
from ddim.datasets import inverse_data_transform  # noqa: E402


logger = logging.getLogger(__name__)


def dict2namespace(config: Dict[str, Any]) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in config.items():
        if isinstance(v, dict):
            v = dict2namespace(v)
        setattr(ns, k, v)
    return ns


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ns = dict2namespace(raw)
    # used by UNet forward for shortcut splitting
    if not hasattr(ns, "split_shortcut"):
        ns.split_shortcut = False
    return ns


def get_beta_schedule(beta_schedule: str, *, beta_start: float, beta_end: float, num_diffusion_timesteps: int):
    """Same schedule definition as your existing scripts."""

    def sigmoid(x):
        return 1 / (math.exp(-x) + 1)

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
    elif beta_schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_diffusion_timesteps, dtype=torch.float64)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape[0] == num_diffusion_timesteps
    return betas.float()


@torch.no_grad()
def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    # alpha_bar: [T]
    a = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise


def get_default_ema_ckpt_name(config: argparse.Namespace) -> str:
    # Your ckpt_util has fixed names like "ema_cifar10".
    ds = str(config.data.dataset).upper()
    if ds == "CIFAR10":
        return "ema_cifar10"
    if ds == "LSUN":
        return f"ema_lsun_{config.data.category}"
    raise ValueError(f"Unsupported dataset in config: {config.data.dataset}")


def build_cifar10_loader(dataset_root: str, batch_size: int, download: bool, *, train: bool, num_workers: int):
    transform = ToTensor()
    # NOTE: Your diffusion data_transform handles [-1,1] rescaling depending on config.
    ds = CIFAR10(root=dataset_root, train=train, download=download, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return ds, loader


def freeze_non_fakequant_params(model: nn.Module):
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Enable only Learnable fake quant params
    from torch.quantization import FakeQuantizeBase

    for m in model.modules():
        if isinstance(m, FakeQuantizeBase):
            for p in m.parameters(recurse=False):
                p.requires_grad = True


def parse_args():
    parser = argparse.ArgumentParser(description="MQBench LSQ QAT for diffusion UNet")
    parser.add_argument("--config", type=str, default="configs/cifar10.yml")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dataset_root", type=str, default="./data")
    parser.add_argument("--download", action="store_true")

    # Quant settings
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)
    parser.add_argument("--a_sym", action="store_true", help="Activation symmetric quantization (align --a_sym baseline)")
    parser.add_argument("--w_sym", action="store_true", default=True, help="Weight symmetric quantization")
    parser.add_argument("--per_channel_w", action="store_true", default=True, help="Per-channel weight quantization")

    # Diffusion/QAT settings
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--cali_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)

    # UNet forward split shortcut
    parser.add_argument("--split", action="store_true")

    # Save
    parser.add_argument("--outdir", type=str, default="MQBench-main/lsq_qat_runs")
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outdir = Path(args.outdir) / f"run_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train.log"

    # Keep console logging while also persisting logs into the same run folder as checkpoints.
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    log_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(log_fmt)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info("Run directory: %s", outdir)
    logger.info("Training log file: %s", log_path)

    config = load_config(args.config)
    config.split_shortcut = bool(args.split)

    # --- Patch original Model.forward to be FX-friendly (skip assert for Proxy) ---
    import torch.fx as fx
    import ddim.models.diffusion as diffusion_mod  # type: ignore

    _orig_model_forward = diffusion_mod.Model.forward

    def _fx_safe_forward(self, x, t=None, context=None):
        # Only run shape assert for real tensors; skip for FX Proxy.
        if isinstance(x, torch.Tensor) and not isinstance(x, fx.Proxy):
            assert x.shape[2] == x.shape[3] == self.resolution
        return _orig_model_forward(self, x, t, context)

    diffusion_mod.Model.forward = _fx_safe_forward
    # ----------------------------------------------------------------------

    logger.info("Loading fp32 UNet (denoising model) ...")
    base_unet = Model(config).to("cpu").eval()
    ckpt_name = get_default_ema_ckpt_name(config)
    ckpt_path = get_ckpt_path(ckpt_name)
    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")
    base_unet.load_state_dict(state, strict=True)

    # Wrap UNet into a small module to bypass the assert during FX tracing.
    class DiffusionUNetWrapper(nn.Module):
        def __init__(self, unet: nn.Module):
            super().__init__()
            self.unet = unet
            # Expose resolution for downstream checks if needed.
            self.resolution = getattr(unet, "resolution", None)

        def forward(self, x, t=None, context=None):
            # Do not assert on Proxy tensors; FX tracing passes symbolic args here.
            if isinstance(x, torch.Tensor) and self.resolution is not None:
                # Only check basic shape sanity for real tensors.
                if x.dim() == 4:
                    assert x.shape[2] == x.shape[3]
            return self.unet(x, t, context)

    fp32_model = DiffusionUNetWrapper(base_unet).to("cpu").eval()

    # ----------------------------
    # Prepare MQBench LSQ QAT model
    # ----------------------------
    w_sym = bool(args.w_sym)
    extra_qconfig_dict = {
        "w_observer": "LSQObserver",
        "a_observer": "LSQObserver",
        "w_fakequantize": "LearnableFakeQuantize",
        "a_fakequantize": "LearnableFakeQuantize",
        # Academic backend needs explicit qscheme definitions
        "w_qscheme": {
            "symmetry": w_sym,
            "per_channel": bool(args.per_channel_w),
            "pot_scale": False,
            "bit": int(args.weight_bit),
        },
        "a_qscheme": {
            "symmetry": bool(args.a_sym),
            "per_channel": False,
            "pot_scale": False,
            "bit": int(args.act_bit),
        },
    }
    prepare_custom_config_dict = {"extra_qconfig_dict": extra_qconfig_dict}

    logger.info("Preparing MQBench model with LSQ QAT ...")
    # Important: prepare_by_platform does FX transform; keep model on CPU here.
    qat_model = prepare_by_platform(fp32_model, BackendType.Academic, prepare_custom_config_dict, is_qat=True)
    qat_model.to(device)

    # ----------------------------
    # Prepare diffusion schedule
    # ----------------------------
    logger.info("Building diffusion schedule ...")
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    ).to(device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)

    # ----------------------------
    # Data
    # ----------------------------
    logger.info("Building CIFAR10 loader ...")
    ds_train, train_loader = build_cifar10_loader(
        dataset_root=args.dataset_root, batch_size=args.batch_size, download=args.download, train=True, num_workers=args.num_workers
    )
    train_iter = cycle(train_loader)

    # Small calibration subset (to save time)
    cali_subset_n = min(len(ds_train), args.batch_size * args.cali_steps)
    cali_ds = Subset(ds_train, indices=torch.arange(cali_subset_n))
    cali_loader = DataLoader(cali_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    cali_iter = cycle(cali_loader)

    # ----------------------------
    # Calibration + enable quantization
    # ----------------------------
    qat_model.eval()
    logger.info("Running calibration (%d steps) ...", args.cali_steps)
    enable_calibration(qat_model)
    with torch.no_grad():
        for _ in range(args.cali_steps):
            x0, _ = next(cali_iter)
            x0 = data_transform(config, x0.to(device))
            t = torch.randint(0, alpha_bar.numel(), (x0.size(0),), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise, alpha_bar)
            _ = qat_model(xt, t)
    enable_quantization(qat_model)

    # Freeze non-quant params for memory efficiency
    freeze_non_fakequant_params(qat_model)
    trainable = [p for p in qat_model.parameters() if p.requires_grad]
    logger.info("Trainable params: %d", sum(p.numel() for p in trainable))

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, qat_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ----------------------------
    # Training loop (noise prediction MSE)
    # ----------------------------
    logger.info("Start LSQ QAT training ...")
    qat_model.train()
    for step in range(args.train_steps):
        x0, _ = next(train_iter)
        x0 = data_transform(config, x0.to(device))
        t = torch.randint(0, alpha_bar.numel(), (x0.size(0),), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, noise, alpha_bar)

        eps_pred = qat_model(xt, t)
        loss = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_every == 0:
            logger.info("step=%d/%d loss=%.6f", step + 1, args.train_steps, float(loss.detach().cpu()))

    # ----------------------------
    # Save checkpoint
    # ----------------------------
    ckpt_path = outdir / "lsq_mqbench_qat_unet.pth"
    torch.save(
        {
            "state_dict": qat_model.state_dict(),
            "config": args.config,
            "weight_bit": args.weight_bit,
            "act_bit": args.act_bit,
            "a_sym": bool(args.a_sym),
            "split": bool(args.split),
            "train_steps": args.train_steps,
        },
        ckpt_path,
    )
    logger.info("Saved QAT checkpoint: %s", ckpt_path)

    # ----------------------------
    # Auto sampling with trained QAT model
    # ----------------------------
    logger.info("Start sampling with trained QAT model ...")
    qat_model.eval()
    enable_quantization(qat_model)

    # Build diffusion seq (use same schedule as training; steps configurable if needed)
    total_timesteps = betas.shape[0]
    sample_steps = min(total_timesteps, 100)
    skip_type = "quad"
    if skip_type == "uniform":
        skip = max(1, total_timesteps // sample_steps)
        seq = [int(s) for s in range(0, total_timesteps, skip)]
    else:
        seq = torch.linspace(0, math.sqrt(total_timesteps * 0.8), sample_steps).pow(2.0)
        seq = [int(s.item()) for s in seq]

    samples_dir = outdir / "samples" / "img"
    samples_dir.mkdir(parents=True, exist_ok=True)
    max_images = 10000
    batch_size = min(64, max_images)
    n_rounds = math.ceil(max_images / batch_size)
    img_id = 0

    with torch.no_grad():
        for _ in range(n_rounds):
            n = min(batch_size, max_images - img_id)
            x = torch.randn(
                n,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=device,
            )
            xs = generalized_steps(x, seq, qat_model, betas, eta=0.0)
            x_out = inverse_data_transform(config, xs[0][-1])

            for i in range(n):
                tv_path = samples_dir / f"{img_id}.png"
                import torchvision.utils as tvu  # local import to avoid top-level dependency if unused

                tvu.save_image(x_out[i], tv_path)
                img_id += 1
                if img_id >= max_images:
                    break

    logger.info("Auto sampling done, saved %d images to %s", img_id, samples_dir)


if __name__ == "__main__":
    main(parse_args())

