"""分组 max_avg 激活截断（TMMA / 两阶段量化共用）。"""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from qdiff.quant_layer import QuantModule, UniformAffineQuantizer

logger = logging.getLogger(__name__)

HookRecord = Tuple[str, object, float, str]


def load_max_avg_statistics(json_path: str) -> dict:
    """加载 JSON：{layer_name: {0: max_avg_g0, 1: max_avg_g1}}。"""
    with open(json_path, "r", encoding="utf-8") as f:
        statistics = json.load(f)
    if "layers" in statistics:
        layer_blob = statistics["layers"]
    else:
        layer_blob = {
            k: v
            for k, v in statistics.items()
            if isinstance(v, dict) and ("0" in v or "1" in v)
        }
    max_avg_stats = {}
    for layer_name, groups in layer_blob.items():
        max_avg_stats[layer_name] = {
            0: groups.get("0", {}).get("max_avg", None),
            1: groups.get("1", {}).get("max_avg", None),
        }
    logger.info("从 %s 加载了 %d 层的 max_avg 统计", json_path, len(max_avg_stats))
    return max_avg_stats


def apply_max_avg_clipping_to_fp_model(
    model: nn.Module, max_avg_stats: dict, group_id: int
) -> List[HookRecord]:
    """为全精度 U-Net 注册 forward hook，BRECQ 重建时截断 FP 参考激活。"""
    hooks: List[HookRecord] = []

    def make_clipping_hook(max_avg_val):
        def hook_fn(_module, _input, output):
            if isinstance(output, torch.Tensor):
                output_abs = output.abs()
                output_abs_clipped = torch.clamp(output_abs, 0.0, max_avg_val)
                return torch.sign(output) * output_abs_clipped
            if isinstance(output, (list, tuple)):
                return type(output)(
                    [
                        torch.sign(item) * torch.clamp(item.abs(), 0.0, max_avg_val)
                        if isinstance(item, torch.Tensor)
                        else item
                        for item in output
                    ]
                )
            return output

        return hook_fn

    for name, module in model.named_modules():
        if len(list(module.children())) != 0:
            continue
        layer_key = None
        if name in max_avg_stats:
            layer_key = name
        else:
            for layer_name in max_avg_stats.keys():
                if layer_name in name or name.endswith(layer_name.split(".")[-1]):
                    layer_key = layer_name
                    break
        if layer_key and max_avg_stats[layer_key][group_id] is not None:
            max_avg = max_avg_stats[layer_key][group_id]
            if max_avg > 0:
                hook = module.register_forward_hook(make_clipping_hook(max_avg))
                hooks.append((name, hook, max_avg, layer_key))

    if hooks:
        logger.info("为组 %d 注册了 %d 个 FP 激活截断 hook", group_id, len(hooks))
    else:
        logger.warning("未找到与 max_avg 统计匹配的层（group_id=%d）", group_id)
    return hooks


def remove_hooks(hooks: List[HookRecord]) -> None:
    for _name, hook, _max_avg, _layer_key in hooks:
        hook.remove()
    if hooks:
        logger.info("移除了 %d 个 max_avg hook", len(hooks))


def set_activation_truncation_from_max_avg(
    qnn: nn.Module, max_avg_stats: dict, group_id: int
) -> None:
    """为量化器设置 truncation_e_y（组内 max_avg 阈值）。"""
    set_count = 0
    max_avg_values = []
    for name, module in qnn.named_modules():
        if isinstance(module, UniformAffineQuantizer):
            layer_key = _match_layer_key(name, max_avg_stats)
            if layer_key and max_avg_stats[layer_key][group_id] is not None:
                max_avg = max_avg_stats[layer_key][group_id]
                if max_avg > 0:
                    module.truncation_e_y = max_avg
                    max_avg_values.append(max_avg)
                    set_count += 1
        elif isinstance(module, QuantModule):
            aq = getattr(module, "act_quantizer", None)
            if isinstance(aq, UniformAffineQuantizer):
                layer_key = _match_layer_key(name, max_avg_stats)
                if layer_key and max_avg_stats[layer_key][group_id] is not None:
                    max_avg = max_avg_stats[layer_key][group_id]
                    if max_avg > 0:
                        aq.truncation_e_y = max_avg
                        max_avg_values.append(max_avg)
                        set_count += 1
    if max_avg_values:
        arr = np.array(max_avg_values)
        logger.info(
            "组 %d max_avg 阈值: min=%.4f max=%.4f mean=%.4f",
            group_id,
            arr.min(),
            arr.max(),
            arr.mean(),
        )
    logger.info("为 %d 个激活量化器设置了组 %d 的截断阈值", set_count, group_id)


def _match_layer_key(name: str, max_avg_stats: dict) -> Optional[str]:
    for layer_name in max_avg_stats.keys():
        if layer_name in name:
            return layer_name
    return None


def wrap_activation_quantizer_forward(quantizer: UniformAffineQuantizer) -> None:
    if getattr(quantizer, "_original_forward_wrapped", False):
        return
    original_forward = quantizer.forward
    original_init = quantizer.init_quantization_scale

    def _clip_x(x: torch.Tensor) -> torch.Tensor:
        if getattr(quantizer, "truncation_e_y", None) is None:
            return x
        max_avg = quantizer.truncation_e_y
        if quantizer.sym:
            return torch.clamp(x, -max_avg, max_avg)
        return torch.clamp(x, 0.0, max_avg)

    def forward_with_max_avg_clipping(x: torch.Tensor):
        return original_forward(_clip_x(x))

    def init_quantization_scale_with_max_avg(x: torch.Tensor, channel_wise: bool = False):
        return original_init(_clip_x(x), channel_wise)

    quantizer.forward = forward_with_max_avg_clipping
    quantizer.init_quantization_scale = init_quantization_scale_with_max_avg
    quantizer._original_forward_wrapped = True


def apply_max_avg_clipping_to_model(qnn: nn.Module) -> None:
    for module in qnn.modules():
        if isinstance(module, UniformAffineQuantizer):
            wrap_activation_quantizer_forward(module)
        elif isinstance(module, QuantModule):
            aq = getattr(module, "act_quantizer", None)
            if isinstance(aq, UniformAffineQuantizer):
                wrap_activation_quantizer_forward(aq)


def setup_max_avg_act_recon(
    qnn: nn.Module,
    fp_unet: nn.Module,
    max_avg_json: str,
    group_id: int = 0,
) -> List[HookRecord]:
    """激活 BRECQ 前调用；返回 hooks，重建结束后需 remove_hooks。"""
    stats = load_max_avg_statistics(max_avg_json)
    hooks = apply_max_avg_clipping_to_fp_model(fp_unet, stats, group_id)
    set_activation_truncation_from_max_avg(qnn, stats, group_id)
    apply_max_avg_clipping_to_model(qnn)
    return hooks
