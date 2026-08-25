"""
DDIM 采样 + 自动 FID。

示例（pth 生成到 images，FID 仍用原 img）：
  python scripts/auto_sample_ddim.py --ckpt ...\\2025-12-03-13-59-50\\ckpt.pth --max_images 10000
"""

from __future__ import annotations

import argparse
import datetime
import gc
import glob
import logging
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as tvu
import tqdm
import yaml
from pytorch_lightning import seed_everything
from torch.cuda import amp

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper
from ddim.functions.ckpt_util import get_ckpt_path
from ddim.models.diffusion import Model
from FID import calculate_fid
from qdiff import BaseQuantBlock, QuantModel, QuantModule, block_reconstruction, layer_reconstruction
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import get_train_samples, resume_cali_model

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


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def _infer_quant_from_path(ckpt: Path) -> tuple[int, bool]:
    s = str(ckpt).lower()
    if re.search(r"w4a8", s):
        return 4, True
    if re.search(r"w8a8", s):
        return 8, True
    if re.search(r"w4", s):
        return 4, False
    if re.search(r"w8", s):
        return 8, False
    return 8, True


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
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1.0 - self.betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = self.betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        """--resume + cali_ckpt：直接加载；否则走完整 PTQ 校准/重建（与 sample_diffusion_ddim 一致）。"""
        model = Model(self.config)
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError(self.config.data.dataset)

        fp_ckpt = get_ckpt_path(f"ema_{name}")
        logger.info("Loading FP checkpoint %s", fp_ckpt)
        model.load_state_dict(torch.load(fp_ckpt, map_location=self.device))
        model.to(self.device)
        model.eval()
        assert self.args.cond is False

        if self.args.ptq and self.args.quant_mode == "qdiff":
            wq_params = {
                "n_bits": self.args.weight_bit,
                "channel_wise": True,
                "scale_method": "max",
            }
            aq_params = {
                "n_bits": self.args.act_bit,
                "symmetric": self.args.a_sym,
                "channel_wise": False,
                "scale_method": "max",
                "leaf_param": self.args.quant_act,
            }
            if self.args.resume:
                logger.info("Load with min-max quick initialization (resume)")
                wq_params["scale_method"] = "max"
                aq_params["scale_method"] = "max"
            if self.args.resume_w:
                wq_params["scale_method"] = "max"

            qnn = QuantModel(
                model=model,
                weight_quant_params=wq_params,
                act_quant_params=aq_params,
                sm_abit=self.args.sm_abit,
            )
            qnn.to(self.device)
            qnn.eval()

            if self.args.resume:
                if not self.args.cali_ckpt:
                    raise ValueError("resume 模式需要 --ckpt")
                image_size = self.config.data.image_size
                channels = self.config.data.channels
                cali_data = (
                    torch.randn(1, channels, image_size, image_size),
                    torch.randint(0, 1000, (1,)),
                )
                logger.info("Loading quantized ckpt: %s", self.args.cali_ckpt)
                resume_cali_model(
                    qnn,
                    self.args.cali_ckpt,
                    cali_data,
                    self.args.quant_act,
                    "qdiff",
                    cond=False,
                )
            else:
                if not self.args.cali_data_path:
                    raise ValueError("从头校准需要 --cali_data_path")
                logger.info(
                    "Sampling data from %s timesteps for calibration",
                    self.args.cali_st,
                )
                sample_data = torch.load(self.args.cali_data_path, map_location="cpu")
                cali_data = get_train_samples(self.args, sample_data, custom_steps=0)
                del sample_data
                gc.collect()
                logger.info(
                    "Calibration data shape: %s %s",
                    cali_data[0].shape,
                    cali_data[1].shape,
                )
                cali_xs, cali_ts = cali_data

                if self.args.resume_w:
                    if not self.args.cali_ckpt:
                        raise ValueError("resume_w 需要 --cali_ckpt")
                    resume_cali_model(
                        qnn, self.args.cali_ckpt, cali_data, False, cond=False
                    )
                else:
                    logger.info("Initializing weight quantization parameters")
                    qnn.set_quant_state(True, False)
                    _ = qnn(cali_xs[:8].to(self.device), cali_ts[:8].to(self.device))
                    logger.info("Initializing has done!")

                kwargs = dict(
                    cali_data=cali_data,
                    batch_size=self.args.cali_batch_size,
                    iters=self.args.cali_iters,
                    weight=0.01,
                    asym=True,
                    b_range=(20, 2),
                    warmup=0.2,
                    act_quant=False,
                    opt_mode="mse",
                )

                def recon_model(submodel):
                    for sub_name, module in submodel.named_children():
                        logger.info("%s %s", sub_name, isinstance(module, BaseQuantBlock))
                        if isinstance(module, QuantModule):
                            if module.ignore_reconstruction:
                                logger.info("Ignore reconstruction of layer %s", sub_name)
                                continue
                            logger.info("Reconstruction for layer %s", sub_name)
                            layer_reconstruction(qnn, module, **kwargs)
                        elif isinstance(module, BaseQuantBlock):
                            if module.ignore_reconstruction:
                                logger.info("Ignore reconstruction of block %s", sub_name)
                                continue
                            logger.info("Reconstruction for block %s", sub_name)
                            block_reconstruction(qnn, module, **kwargs)
                        else:
                            recon_model(module)

                if not self.args.resume_w:
                    logger.info("Doing weight calibration")
                    recon_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=False)

                if self.args.quant_act:
                    logger.info("Doing activation calibration")
                    qnn.set_quant_state(True, True)
                    with torch.no_grad():
                        inds = np.random.choice(cali_xs.shape[0], 64, replace=False)
                        _ = qnn(cali_xs[inds].to(self.device), cali_ts[inds].to(self.device))
                        if self.args.running_stat:
                            logger.info("Running stat for activation quantization")
                            qnn.set_running_stat(True)
                            for i in range(int(cali_xs.size(0) / 64)):
                                _ = qnn(
                                    (
                                        cali_xs[i * 64 : (i + 1) * 64].to(self.device),
                                        cali_ts[i * 64 : (i + 1) * 64].to(self.device),
                                    )
                                )
                            qnn.set_running_stat(False)

                    kwargs = dict(
                        cali_data=cali_data,
                        iters=self.args.cali_iters_a,
                        act_quant=True,
                        opt_mode="mse",
                        lr=self.args.cali_lr,
                        p=self.args.cali_p,
                    )
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
                                m.zero_point = nn.Parameter(
                                    torch.tensor(float(m.zero_point))
                                )
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
        logger.info("starting from image %s", img_id)
        total_n_samples = self.args.max_images
        n_rounds = math.ceil((total_n_samples - img_id) / config.sampling.batch_size)

        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.no_grad():
            for _ in tqdm.tqdm(
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
                    n = self.args.max_images - img_id
                for j in range(n):
                    tvu.save_image(
                        x[j], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_image(self, x, model, last=True):
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

            xs = generalized_steps(
                x, seq, model, self.betas, eta=self.args.eta, args=self.args
            )
            x = xs
        elif self.args.sample_type == "dpm_solver":
            logger.info("use dpm-solver with %s steps", self.args.timesteps)
            noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
            model_fn = model_wrapper(model, noise_schedule, model_type="noise")
            dpm_solver = DPM_Solver(
                model_fn, noise_schedule, algorithm_type="dpmsolver++"
            )
            return dpm_solver.sample(
                x,
                steps=self.args.timesteps,
                order=3,
                skip_type="time_uniform",
                method="singlestep",
            )
        else:
            raise NotImplementedError(self.args.sample_type)
        if last:
            x = x[0][-1]
        return x


def _resolve_quant_flags(cli, ckpt: Path | None) -> tuple[int, bool]:
    if ckpt is not None:
        w_bit, act_default = _infer_quant_from_path(ckpt)
    else:
        w_bit, act_default = 8, True
    weight_bit = cli.weight_bit if cli.weight_bit is not None else w_bit
    if cli.no_quant_act:
        quant_act = False
    elif cli.quant_act:
        quant_act = True
    else:
        quant_act = act_default
    return weight_bit, quant_act


def _resolve_subdir(run_dir: Path, name: str) -> Path:
    p = Path(name.strip())
    if not p.is_absolute():
        p = run_dir / p
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _setup_run_dirs(cli) -> tuple[Path, Path, Path, Path, bool]:

    real_dir = Path(cli.real_images).resolve()
    use_resume = bool(cli.ckpt and cli.ckpt.strip()) and not cli.calibrate

    if use_resume:
        ckpt = Path(cli.ckpt.strip()).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"ckpt 不存在: {ckpt}")
        run_dir = ckpt.parent
        gen_name = cli.gen_dir.strip() or "images"
        fid_name = cli.fid_dir.strip() or "img"
        gen_dir = _resolve_subdir(run_dir, gen_name)
        fid_dir = _resolve_subdir(run_dir, fid_name)
        return run_dir, gen_dir, fid_dir, real_dir, True

    if not cli.cali_data_path:
        raise ValueError("未提供 --ckpt 时必须提供 --cali_data_path 以从头校准")
    logroot = Path(cli.logdir).resolve() if cli.logdir else (_REPO_ROOT / "auto_cali_output")
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = logroot / "samples" / now
    gen_dir = _resolve_subdir(run_dir, cli.gen_dir.strip() or "img")
    fid_dir = (
        _resolve_subdir(run_dir, cli.fid_dir.strip())
        if cli.fid_dir.strip()
        else gen_dir
    )
    return run_dir, gen_dir, fid_dir, real_dir, False


def _build_args(cli, run_dir: Path, gen_dir: Path, resume_mode: bool) -> argparse.Namespace:
    ckpt_path = None
    if resume_mode:
        ckpt_path = Path(cli.ckpt.strip()).resolve()
    weight_bit, quant_act = _resolve_quant_flags(cli, ckpt_path)

    return argparse.Namespace(
        config=cli.config,
        seed=cli.seed,
        use_pretrained=True,
        sample_type="generalized",
        skip_type=cli.skip_type,
        timesteps=cli.timesteps,
        eta=cli.eta,
        ptq=True,
        quant_act=quant_act,
        weight_bit=weight_bit,
        act_bit=cli.act_bit,
        quant_mode="qdiff",
        max_images=cli.max_images,
        cali_st=cli.cali_st,
        cali_batch_size=cli.cali_batch_size,
        cali_n=cli.cali_n,
        cali_iters=cli.cali_iters,
        cali_iters_a=cli.cali_iters_a,
        cali_lr=cli.cali_lr,
        cali_p=cli.cali_p,
        cali_ckpt=str(ckpt_path) if ckpt_path else None,
        cali_data_path=cli.cali_data_path,
        resume=resume_mode,
        resume_w=cli.resume_w,
        cond=False,
        a_sym=cli.a_sym or quant_act,
        running_stat=cli.running_stat,
        sm_abit=cli.sm_abit,
        split=resume_mode or cli.split,
        verbose=cli.verbose,
        image_folder=str(gen_dir),
        logdir=str(run_dir),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DDIM 采样 + FID：--ckpt 直接推理，或从头 PTQ 校准（可选）"
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="已校准 ckpt.pth；提供则直接加载推理（默认模式）",
    )
    ap.add_argument(
        "--gen_dir",
        type=str,
        default="",
    )
    ap.add_argument(
        "--fid_dir",
        type=str,
        default="",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="强制从头 PTQ 校准（忽略 --ckpt，需 --cali_data_path 与 -l）",
    )
    ap.add_argument("-l", "--logdir", type=str, default="", help="从头校准时输出根目录")
    ap.add_argument("--max_images", type=int, default=10000)
    ap.add_argument(
        "--real_images",
        type=str,
        default=str(_REPO_ROOT / "real_images"),
    )
    ap.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "cifar10.yml"),
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timesteps", type=int, default=100)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--skip_type", type=str, default="quad", choices=["quad", "uniform"])
    ap.add_argument("--weight_bit", type=int, default=None)
    ap.add_argument("--act_bit", type=int, default=8)
    ap.add_argument("--quant_act", action="store_true")
    ap.add_argument("--no_quant_act", action="store_true")
    ap.add_argument("--a_sym", action="store_true", help="激活对称量化（w8a8 等建议开启）")
    ap.add_argument("--running_stat", action="store_true")
    ap.add_argument("--split", action="store_true", help="split shortcut（CIFAR 量化常用）")
    ap.add_argument("--resume_w", action="store_true", help="仅恢复权重后做激活校准")
    ap.add_argument("--cali_data_path", type=str, default="")
    ap.add_argument("--cali_st", type=int, default=20)
    ap.add_argument("--cali_batch_size", type=int, default=32)
    ap.add_argument("--cali_n", type=int, default=256)
    ap.add_argument("--cali_iters", type=int, default=20000)
    ap.add_argument("--cali_iters_a", type=int, default=5000)
    ap.add_argument("--cali_lr", type=float, default=4e-4)
    ap.add_argument("--cali_p", type=float, default=2.4)
    ap.add_argument("--sm_abit", type=int, default=8)
    ap.add_argument("--skip_sample", action="store_true", help="仅计算 FID")
    ap.add_argument("--fid_only", action="store_true")
    ap.add_argument("--no_fid", action="store_true", help="采样后不计算 FID")
    ap.add_argument("--verbose", action="store_true")
    cli = ap.parse_args()

    run_dir, gen_dir, fid_dir, real_dir, resume_mode = _setup_run_dirs(cli)
    args = _build_args(cli, run_dir, gen_dir, resume_mode)
    skip_sample = cli.skip_sample or cli.fid_only

    log_path = run_dir / "auto_sample_run.log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    if not skip_sample:
        with open(args.config, "r", encoding="utf-8") as f:
            config = dict2namespace(yaml.safe_load(f))
        seed_everything(args.seed)
        logger.info("=" * 75)
        logger.info("mode : %s", "resume (--ckpt)" if resume_mode else "calibrate (PTQ)")
        if args.cali_ckpt:
            logger.info("ckpt : %s", args.cali_ckpt)
        logger.info("w%d quant_act=%s", args.weight_bit, args.quant_act)
        logger.info("=" * 75)
        Diffusion(args, config).sample()

    if not cli.no_fid:
        # FID.py 仅支持含 mu/sigma 的生成图 npz；真实分布已写死在 FID.REAL_NPZ
        gen_npz = Path(str(fid_dir) + "_fid_stats.npz")
        if not gen_npz.is_file():
            raise FileNotFoundError(
                f"未找到生成图统计 npz: {gen_npz}\n"
                "请先将图片目录提取为含 mu/sigma 的 npz，再计算 FID。"
            )
        fid_val = calculate_fid(str(gen_npz))
        print(fid_val)


if __name__ == "__main__":
    main()
