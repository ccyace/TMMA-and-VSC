import argparse, os, sys, gc, glob, datetime, yaml
import logging
import time
import numpy as np
from tqdm import trange, tqdm
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image
import torchvision.utils as tvu
sys.path.append(r'C:\Users\wsh\Desktop\q-diffusion-master')
import torch
import torch.nn as nn
import sys
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config
from ddim.functions.denoising import generalized_steps
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples
from qdiff.max_avg_truncation import (
    load_max_avg_statistics,
    apply_max_avg_clipping_to_fp_model,
    set_activation_truncation_from_max_avg,
    apply_max_avg_clipping_to_model,
    remove_hooks,
)

logger = logging.getLogger(__name__)

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0
                    ):
    dpm = DPMSolverSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = dpm.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, dpm=False):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    # with model.ema_scope("Plotting"):
    t0 = time.time()
    if vanilla:
        sample, progrow = convsample(model, shape,
                                        make_prog_row=True)
    elif dpm:
        logger.info(f'Using DPM sampling with {custom_steps} sampling steps and eta={eta}')
        sample, intermediates = convsample_dpm(model,  steps=custom_steps, shape=shape,
                                                eta=eta)
    else:
        sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                eta=eta)

    t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    logger.info(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
    n_samples=50000, nplog=None, dpm=False):
    if vanilla:
        logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        logger.info(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta, dpm=dpm)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                logger.info(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume_base",
        type=str,
        nargs="?",
        help="load fp32 base model from logdir or checkpoint in logdir (will deprecate after direct quantized model loading implemented)",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=10000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "--seed",
        type=int,
        # default=42,
        required=True,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fast dpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    # linear quantization configs
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
        help="act quantizers use symmetric quantization (empirically helpful in some cases)"
    )
    parser.add_argument(
        "--a_min_max", action="store_true",
        help="act quantizers initialize with min-max (empirically helpful in some cases)"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--rs_sm_only", action="store_true",
        help="use running statistics only for softmax act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--dpm", action="store_true",
        help="use dpm solver for sampling"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--init_debug_log", action="store_true",
        help="print progress logs during weight quantizer initialization"
    )
    parser.add_argument(
        "--init_log_interval", type=int, default=50,
        help="log interval (in visited QuantModule count) for init debug progress"
    )
    parser.add_argument(
        "--two_stage",
        action="store_true",
        help="两阶段 TMMA：组0 BRECQ+前半采样，组1 BRECQ+后半采样（需 --max_avg_json）",
    )
    parser.add_argument(
        "--max_avg_json",
        type=str,
        default="",
        help="分组 max_avg 统计 JSON（两阶段必填；单阶段可选一组截断）",
    )
    parser.add_argument(
        "--max_avg_group",
        type=int,
        default=0,
        choices=[0, 1],
        help="单阶段激活 BRECQ 使用的时间组（两阶段模式下忽略）",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="quad",
        choices=["quad", "uniform"],
        help="两阶段半段采样用的 DDIM 步序列类型",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


def _clone_ldm_unet(ldm_model, config):
    """从当前 LDM 拷贝一份独立 U-Net（供两阶段各自 QuantModel 使用）。"""
    unet = instantiate_from_config(config.model.params.unet_config)
    unet.load_state_dict(ldm_model.model.diffusion_model.state_dict(), strict=False)
    unet.cuda()
    unet.eval()
    return unet


def _build_ddim_seq(num_timesteps, custom_steps, skip_type):
    total_timesteps = num_timesteps
    if skip_type == "uniform":
        skip = total_timesteps // custom_steps
        seq = list(range(0, total_timesteps, skip))
    elif skip_type == "quad":
        seq = (
            np.linspace(0, np.sqrt(total_timesteps * 0.8), custom_steps) ** 2
        )
        seq = [int(s) for s in list(seq)]
        seq = sorted(list(set(seq)))
    else:
        raise ValueError(f"不支持的 skip_type: {skip_type}")
    return seq


def _half_ddim_seq(num_timesteps, custom_steps, skip_type, first_half=True):
    seq = _build_ddim_seq(num_timesteps, custom_steps, skip_type)
    reversed_seq = list(reversed(seq))
    half_steps = len(reversed_seq) // 2
    if first_half:
        return reversed_seq[:half_steps]
    return reversed_seq[half_steps:]


def _walk_brecq(qnn, root, kwargs):
    for name, module in root.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction:
                continue
            layer_reconstruction(qnn, module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction:
                continue
            block_reconstruction(qnn, module, **kwargs)
        else:
            _walk_brecq(qnn, module, kwargs)


def _finalize_qnn_params(qnn, quant_act):
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)
        elif isinstance(m, UniformAffineQuantizer) and quant_act:
            if m.zero_point is not None:
                if not torch.is_tensor(m.zero_point):
                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                else:
                    m.zero_point = nn.Parameter(m.zero_point)


def _ldm_stage_brecq(fp_unet, opt, cali_data, cali_xs, cali_ts, max_avg_stats, group_id):
    """单阶段：权重 BRECQ + 组 group_id 激活 BRECQ；返回 (qnn, fp_unet)。"""
    a_scale_method = "mse" if not opt.a_min_max else "max"
    wq_params = {"n_bits": opt.weight_bit, "channel_wise": True, "scale_method": "max"}
    aq_params = {
        "n_bits": opt.act_bit,
        "symmetric": opt.a_sym,
        "channel_wise": False,
        "scale_method": a_scale_method,
        "leaf_param": opt.quant_act,
    }
    qnn = QuantModel(
        model=fp_unet,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        sm_abit=opt.sm_abit,
    )
    qnn.cuda()
    qnn.eval()

    kwargs_w = dict(
        cali_data=cali_data,
        batch_size=opt.cali_batch_size,
        iters=opt.cali_iters,
        weight=0.01,
        asym=True,
        b_range=(20, 2),
        warmup=0.2,
        act_quant=False,
        opt_mode="mse",
    )
    logger.info("组 %d：权重量化初始化", group_id)
    qnn.set_quant_state(True, False)
    with torch.no_grad():
        _ = qnn(cali_xs[:8].cuda(), cali_ts[:8].cuda())
    logger.info("组 %d：权重 BRECQ", group_id)
    _walk_brecq(qnn, qnn, kwargs_w)
    qnn.set_quant_state(weight_quant=True, act_quant=False)

    if opt.quant_act:
        logger.info("组 %d：激活量化（max_avg 截断）", group_id)
        qnn.set_quant_state(True, True)
        hooks = apply_max_avg_clipping_to_fp_model(fp_unet, max_avg_stats, group_id)
        with torch.no_grad():
            _ = qnn(cali_xs[:64].cuda(), cali_ts[:64].cuda())
        if opt.running_stat:
            qnn.set_running_stat(True)
            for i in range(int(cali_xs.size(0) / 64)):
                _ = qnn(
                    cali_xs[i * 64 : (i + 1) * 64].cuda(),
                    cali_ts[i * 64 : (i + 1) * 64].cuda(),
                )
            qnn.set_running_stat(False)
        kwargs_a = dict(
            cali_data=cali_data,
            iters=opt.cali_iters_a,
            act_quant=True,
            opt_mode="mse",
            lr=opt.cali_lr,
            p=opt.cali_p,
        )
        logger.info("组 %d：激活 BRECQ", group_id)
        _walk_brecq(qnn, qnn, kwargs_a)
        remove_hooks(hooks)
        set_activation_truncation_from_max_avg(qnn, max_avg_stats, group_id)
        apply_max_avg_clipping_to_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)

    return qnn


@torch.no_grad()
def _first_half_sampling_ldm(
    qnn, num_samples, betas, num_timesteps, custom_steps, skip_type, eta,
    channels, image_size, device, batch_size,
):
    half_seq = _half_ddim_seq(num_timesteps, custom_steps, skip_type, first_half=True)
    logger.info("前半段 DDIM 子序列: %s -> %s（%d 步）", half_seq[0], half_seq[-1], len(half_seq))
    chunks = []
    for i in tqdm(range(0, num_samples, batch_size), desc="前半段采样"):
        bs = min(batch_size, num_samples - i)
        x = torch.randn(bs, channels, image_size, image_size, device=device)
        xs, _ = generalized_steps(x, half_seq, qnn, betas, eta=eta)
        chunks.append(xs[-1].cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _second_half_sampling_ldm(
    qnn, intermediate, betas, num_timesteps, custom_steps, skip_type, eta, device, batch_size,
):
    half_seq = _half_ddim_seq(num_timesteps, custom_steps, skip_type, first_half=False)
    logger.info("后半段 DDIM 子序列: %s -> %s（%d 步）", half_seq[0], half_seq[-1], len(half_seq))
    chunks = []
    for i in tqdm(range(0, intermediate.shape[0], batch_size), desc="后半段采样"):
        end = min(i + batch_size, intermediate.shape[0])
        x = intermediate[i:end].to(device)
        xs, _ = generalized_steps(x, half_seq, qnn, betas, eta=eta)
        chunks.append(xs[-1].cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _save_ldm_decoded_png(ldm_model, latents, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(latents.shape[0]):
        z = latents[i : i + 1].cuda()
        img = ldm_model.decode_first_stage(z)
        tvu.save_image(img, os.path.join(out_dir, f"{i}.png"))


def two_stage_ldm_quantization(ldm_model, opt, config, logdir, imglogdir):
    """
    两阶段 TMMA：组0 BRECQ + 前半采样 -> 组1 BRECQ + 后半采样 -> VAE 解码存图。
    逻辑对齐 two_stage_quantized_sampling.py。
    """
    if not (opt.max_avg_json and str(opt.max_avg_json).strip()):
        raise ValueError("两阶段模式需要 --max_avg_json")
    if opt.resume or opt.resume_w:
        raise ValueError("两阶段模式不支持 --resume / --resume_w")

    logger.info("开始 LDM 两阶段量化采样")
    max_avg_stats = load_max_avg_statistics(opt.max_avg_json.strip())
    sample_data = torch.load(opt.cali_data_path)
    cali_data = get_train_samples(opt, sample_data)
    del sample_data
    gc.collect()
    cali_xs, cali_ts = cali_data
    logger.info("校准数据: %s %s", cali_xs.shape, cali_ts.shape)

    betas = ldm_model.betas.float().cuda()
    num_timesteps = int(ldm_model.num_timesteps)
    channels = config.model.params.channels
    image_size = config.model.params.image_size
    device = torch.device("cuda")
    skip_type = getattr(opt, "skip_type", "quad")
    batch_size = opt.batch_size

    # ---------- 阶段 1：组 0 ----------
    logger.info("=" * 75)
    logger.info("第一阶段：组 0 max_avg + BRECQ + 前半采样")
    logger.info("=" * 75)
    fp_unet1 = _clone_ldm_unet(ldm_model, config)
    qnn_stage1 = _ldm_stage_brecq(fp_unet1, opt, cali_data, cali_xs, cali_ts, max_avg_stats, 0)
    intermediate = _first_half_sampling_ldm(
        qnn_stage1,
        opt.n_samples,
        betas,
        num_timesteps,
        opt.custom_steps,
        skip_type,
        opt.eta,
        channels,
        image_size,
        device,
        batch_size,
    )
    torch.save(intermediate, os.path.join(logdir, "intermediate_noise_images.pt"))
    del qnn_stage1, fp_unet1
    torch.cuda.empty_cache()
    gc.collect()

    # ---------- 阶段 2：组 1 ----------

    fp_unet2 = _clone_ldm_unet(ldm_model, config)
    qnn_stage2 = _ldm_stage_brecq(fp_unet2, opt, cali_data, cali_xs, cali_ts, max_avg_stats, 1)
    final_latents = _second_half_sampling_ldm(
        qnn_stage2,
        intermediate,
        betas,
        num_timesteps,
        opt.custom_steps,
        skip_type,
        opt.eta,
        device,
        batch_size,
    )
    torch.save(final_latents, os.path.join(logdir, "final_latents.pt"))
    _save_ldm_decoded_png(ldm_model, final_latents, imglogdir)
    _finalize_qnn_params(qnn_stage2, opt.quant_act)
    torch.save(qnn_stage2.state_dict(), os.path.join(logdir, "ckpt.pth"))

    del fp_unet2
    return qnn_stage2


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    # fix random seed
    seed_everything(opt.seed)

    if not os.path.exists(opt.resume_base):
        raise ValueError("Cannot find {}".format(opt.resume_base))
    if os.path.isfile(opt.resume_base):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume_base.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume_base.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume_base
    else:
        assert os.path.isdir(opt.resume_base), f"{opt.resume_base} is not a directory"
        logdir = opt.resume_base.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    logdir = os.path.join(logdir, "samples", now)
    os.makedirs(logdir)
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
    print(config)

    # logger.info(75 * "=")
    # logger.info(f"Host {os.uname()[1]}")
    # logger.info("logging to:")
    # imglogdir = os.path.join(logdir, "img")
    # numpylogdir = os.path.join(logdir, "numpy")
    import os
    import platform

    logger.info(75 * "=")

    # 获取主机名，不同操作系统使用不同方法
    if platform.system() == "Windows":
        host = platform.node()  # Windows 上使用 platform.node() 获取主机名
    else:
        host = os.uname()[1]  # 类 Unix 系统上使用 os.uname()

    logger.info(f"Host {host}")
    logger.info("logging to:")

    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    logger.info(f"global step: {global_step}")
    logger.info("Switched to EMA weights")
    model.model_ema.store(model.model.parameters())
    model.model_ema.copy_to(model.model)

    # print(model.model)
    assert(not opt.cond)
    two_stage_done = False
    if opt.ptq:
        if opt.quant_mode == 'qdiff' and getattr(opt, "two_stage", False):
            qnn_final = two_stage_ldm_quantization(model, opt, config, logdir, imglogdir)
            model.model.diffusion_model = qnn_final
            two_stage_done = True
        elif opt.quant_mode == 'qdiff':
            a_scale_method = 'mse' if not opt.a_min_max else 'max'
            wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
            aq_params = {
                'n_bits': opt.act_bit, 'symmetric': opt.a_sym, 'channel_wise': False, 
                'scale_method': a_scale_method, 'leaf_param': opt.quant_act
            }
            if opt.resume:
                logger.info('Load with min-max quick initialization')
                wq_params['scale_method'] = 'max'
                aq_params['scale_method'] = 'max'
            if opt.resume_w:
                wq_params['scale_method'] = 'max'
            # with model.ema_scope("Quantizing", restore=False):
            qnn = QuantModel(
                model=model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
                sm_abit=opt.sm_abit)
            qnn.cuda()
            qnn.eval()

            if opt.resume:
                image_size = config.model.params.image_size
                channels = config.model.params.channels
                cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)))
                resume_cali_model(qnn, opt.cali_ckpt, cali_data, opt.quant_act, "qdiff", cond=False)
            else:
                logger.info(f"Sampling data from {opt.cali_st} timesteps for calibration")
                sample_data = torch.load(opt.cali_data_path)
                cali_data = get_train_samples(opt, sample_data)
                del(sample_data)
                gc.collect()
                logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape}")
                
                cali_xs, cali_ts = cali_data
                if opt.resume_w:
                    resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=False)
                else:
                    logger.info("Initializing weight quantization parameters")
                    qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
                    init_start = time.time()
                    quant_modules = [m for m in qnn.modules() if isinstance(m, QuantModule)]
                    logger.info(f"QuantModule count before init: {len(quant_modules)}")

                    hooks = []
                    visited_counter = {'count': 0}

                    if opt.init_debug_log:
                        interval = max(1, opt.init_log_interval)

                        def _make_hook(module_name):
                            def _hook(_module, _inputs):
                                visited_counter['count'] += 1
                                if visited_counter['count'] % interval == 0:
                                    logger.info(
                                        f"[init debug] visited QuantModule {visited_counter['count']} / {len(quant_modules)} "
                                        f"(latest: {module_name})"
                                    )
                            return _hook

                        for name, module in qnn.named_modules():
                            if isinstance(module, QuantModule):
                                hooks.append(module.register_forward_pre_hook(_make_hook(name)))

                    with torch.no_grad():
                        _ = qnn(cali_xs[:8].cuda(), cali_ts[:8].cuda())
                    for hook in hooks:
                        hook.remove()
                    inited_weight_quantizers = sum(
                        int(getattr(m.weight_quantizer, "inited", False))
                        for m in quant_modules
                    )
                    logger.info(
                        f"Weight init completed in {time.time() - init_start:.2f}s; "
                        f"initialized quantizers: {inited_weight_quantizers}/{len(quant_modules)}"
                    )
                    logger.info("Initializing has done!")

                # Kwargs for weight rounding calibration
                kwargs = dict(cali_data=cali_data, batch_size=opt.cali_batch_size, 
                            iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
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

                if not opt.resume_w:
                    logger.info("Doing weight calibration")
                    recon_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                if opt.quant_act:
                    logger.info("UNet model")
                    logger.info(model.model)
                    logger.info("Doing activation calibration")
                    qnn.set_quant_state(True, True)
                    max_avg_hooks = []
                    max_avg_stats = None
                    if opt.max_avg_json and str(opt.max_avg_json).strip():
                        logger.info(
                            "激活 BRECQ：max_avg 截断 group=%d",
                            opt.max_avg_group,
                        )
                        max_avg_stats = load_max_avg_statistics(opt.max_avg_json.strip())
                        max_avg_hooks = apply_max_avg_clipping_to_fp_model(
                            qnn.model, max_avg_stats, opt.max_avg_group
                        )
                    with torch.no_grad():
                        _ = qnn(cali_xs[:64].cuda(), cali_ts[:64].cuda())
                        if opt.running_stat:
                            logger.info('Running stat for activation quantization')
                            qnn.set_running_stat(True)
                            for i in trange(int(cali_xs.size(0) / 64)):
                                _ = qnn(
                                    cali_xs[i * 64:(i + 1) * 64].cuda(),
                                    cali_ts[i * 64:(i + 1) * 64].cuda(),
                                )
                            qnn.set_running_stat(False)
                    kwargs = dict(
                        cali_data=cali_data, iters=opt.cali_iters_a, act_quant=True,
                        opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p)
                    recon_model(qnn)
                    if max_avg_hooks:
                        remove_hooks(max_avg_hooks)
                    if max_avg_stats is not None:
                        set_activation_truncation_from_max_avg(
                            qnn, max_avg_stats, opt.max_avg_group
                        )
                        apply_max_avg_clipping_to_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=True)   

                logger.info("Saving calibrated quantized UNet model")
                for m in qnn.model.modules():
                    if isinstance(m, AdaRoundQuantizer):
                        m.zero_point = nn.Parameter(m.zero_point)
                        m.delta = nn.Parameter(m.delta)
                    elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                        if m.zero_point is not None:
                            if not torch.is_tensor(m.zero_point):
                                m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                            else:
                                m.zero_point = nn.Parameter(m.zero_point)
                torch.save(qnn.state_dict(), os.path.join(logdir, "ckpt.pth"))         

            model.model.diffusion_model = qnn


    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        print(sampling_conf)
        logger.info("first_stage_model")
        logger.info(model.first_stage_model)
        logger.info("UNet model")
        logger.info(model.model)


    if not two_stage_done:
        run(model, imglogdir, eta=opt.eta,
            vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
            batch_size=opt.batch_size, nplog=numpylogdir, dpm=opt.dpm)
    else:
        logger.info("两阶段已生成图像至 %s，跳过常规 run() 采样", imglogdir)

    logger.info("done.")
