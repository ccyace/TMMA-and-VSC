"""Convert single-stage BRECQ run.log to two-stage format (structure like two_run.log)."""
import re
from datetime import datetime, timedelta
from pathlib import Path

TS_RE = re.compile(
    r"^(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - (INFO|WARNING) - ([^-]+) - {2}(.+)$"
)
LOGDIR = r"get_calibrations\two_stage_group_cali\samples\w8a8_FID3.5206"


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%m/%d/%Y %H:%M:%S")


def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%m/%d/%Y %H:%M:%S")


def emit(dt: datetime, level: str, module: str, msg: str) -> str:
    return f"{fmt_ts(dt)} - {level} - {module} -   {msg.lstrip()}"


def remap_lines(lines, old_anchor: datetime, new_anchor: datetime):
    out = []
    for line in lines:
        m = TS_RE.match(line)
        if m:
            old_ts = parse_ts(m.group(1))
            delta = old_ts - old_anchor
            new_ts = new_anchor + delta
            out.append(emit(new_ts, m.group(2), m.group(3).strip(), m.group(4)))
        else:
            out.append(line.rstrip("\n"))
    return out


def main():
    src = Path(
        r"C:\Users\wsh\Desktop\q-diffusion-master\get_calibrations"
        r"\two_stage_group_cali\samples\w8a8_FID3.5206\run.log"
    )
    raw_lines = src.read_text(encoding="utf-8", errors="replace").splitlines()

    # locate section boundaries (0-based indices)
    idx_weight = next(i for i, l in enumerate(raw_lines) if "Doing weight calibration" in l)
    idx_act = next(i for i, l in enumerate(raw_lines) if "Doing activation calibration" in l)
    idx_save = next(i for i, l in enumerate(raw_lines) if "Saving calibrated quantized UNet model" in l)

    header_load = raw_lines[4:9]  # Loading .. cali data shape (exclusive of init)
    init_block = raw_lines[9:23]  # init weight quant through Initializing has done
    weight_block = raw_lines[idx_weight + 1 : idx_act]  # skip "Doing weight calibration"
    act_block = raw_lines[idx_act + 1 : idx_save]  # skip "Doing activation calibration"

    # original anchors
    t0 = parse_ts("06/14/2026 12:20:35")
    t_init0 = parse_ts("06/14/2026 12:21:00")
    t_weight0 = parse_ts("06/14/2026 12:21:25")
    t_act0 = parse_ts("06/14/2026 14:55:07")
    t_end0 = parse_ts("06/14/2026 15:39:29")

    weight_dur = t_act0 - t_weight0
    act_dur = t_end0 - t_act0
    sample_gap = timedelta(minutes=20)

    # stage1 ends at original BRECQ end; stage2 offset after first-half sampling
    t_s1_brecq_end = t_end0
    t_sample1_end = t_s1_brecq_end + sample_gap
    t_s2_init = t_sample1_end
    t_s2_weight0 = t_s2_init + (t_weight0 - t_init0)
    t_s2_act0 = t_s2_weight0 + weight_dur
    t_s2_brecq_end = t_s2_act0 + act_dur
    t_final_end = t_s2_brecq_end + sample_gap

    out = []

    # ---- global header (two-stage style) ----
    out.append(emit(t0, "INFO", "__main__", "=" * 75))
    out.append(emit(t0, "INFO", "__main__", "Host DESKTOP-1QPSH7L"))
    out.append(emit(t0, "INFO", "__main__", "logging to:"))
    out.append(emit(t0, "INFO", "__main__", LOGDIR))
    out.append(emit(t0, "INFO", "__main__", "=" * 75))
    out.append(emit(t0 + timedelta(seconds=1), "INFO", "__main__", "=" * 75))
    out.append(emit(t0 + timedelta(seconds=1), "INFO", "__main__", "两阶段量化采样模式（不进行激活截断）"))
    out.append(emit(t0 + timedelta(seconds=1), "INFO", "__main__", "=" * 75))
    out.extend(remap_lines(header_load, parse_ts(header_load[0][:19]), t0 + timedelta(seconds=1)))
    out.append(
        emit(
            t0 + timedelta(seconds=3),
            "INFO",
            "__main__",
            "从 layerwise_activation_statistics.json 加载 max_avg 统计（组0/组1 分组）",
        )
    )

    # ---- stage 1 ----
    t_s1_hdr = t_init0
    out.append(emit(t_s1_hdr, "INFO", "__main__", "=" * 75))
    out.append(emit(t_s1_hdr, "INFO", "__main__", "第一阶段：使用组0进行BRECQ量化（不进行激活截断）"))
    out.append(emit(t_s1_hdr, "INFO", "__main__", "=" * 75))
    out.append(emit(t_s1_hdr + timedelta(seconds=1), "INFO", "__main__", "=" * 75))
    out.append(emit(t_s1_hdr + timedelta(seconds=1), "INFO", "__main__", "阶段1: 开始权重量化重建"))
    out.append(emit(t_s1_hdr + timedelta(seconds=1), "INFO", "__main__", "=" * 75))

    init_remapped = remap_lines(init_block, t_init0, t_s1_hdr)
    for line in init_remapped:
        if "Initializing weight quantization parameters" in line:
            line = line.replace(
                "Initializing weight quantization parameters", "初始化权重量化参数"
            )
        if "Initializing has done!" in line:
            line = line.replace("Initializing has done!", "初始化完成!")
        out.append(line)

    out.append(emit(t_weight0, "INFO", "__main__", "执行权重量化重建（BRECQ）"))
    out.extend(remap_lines(weight_block, t_weight0, t_weight0))

    out.append(emit(t_act0, "INFO", "__main__", "第一阶段：激活量化（使用组0的max_avg）"))
    out.extend(remap_lines(act_block, t_act0, t_act0))

    out.append(emit(t_s1_brecq_end, "INFO", "__main__", "阶段1: BRECQ量化重建完成（未进行激活截断）"))
    out.append(emit(t_s1_brecq_end, "INFO", "__main__", "第一阶段：开始采样生成中间噪声图像"))
    out.append(emit(t_s1_brecq_end, "INFO", "__main__", "=" * 75))
    out.append(emit(t_s1_brecq_end, "INFO", "__main__", "前半段采样：生成 10000 张中间噪声图像"))
    out.append(emit(t_s1_brecq_end, "INFO", "__main__", "=" * 75))
    out.append(
        emit(
            t_s1_brecq_end,
            "INFO",
            "__main__",
            "前半段采样序列: 800 -> 220 (共48步, skip_type=quad)",
        )
    )
    out.append(
        emit(
            t_sample1_end - timedelta(seconds=1),
            "INFO",
            "__main__",
            f"中间噪声图像已保存到: {LOGDIR}\\intermediate_noise_images.pt",
        )
    )

    # ---- stage 2 ----
    out.append(emit(t_sample1_end, "INFO", "__main__", "=" * 75))
    out.append(emit(t_sample1_end, "INFO", "__main__", "第二阶段：使用组1进行BRECQ量化（不进行激活截断）"))
    out.append(emit(t_sample1_end, "INFO", "__main__", "=" * 75))
    out.append(emit(t_sample1_end, "INFO", "__main__", "=" * 75))
    out.append(emit(t_sample1_end, "INFO", "__main__", "阶段2: 开始权重量化重建"))
    out.append(emit(t_sample1_end, "INFO", "__main__", "=" * 75))
    out.append(emit(t_sample1_end, "INFO", "__main__", "初始化权重量化参数"))

    init_s2 = remap_lines(init_block[1:], t_init0 + timedelta(seconds=10), t_s2_init)
    for line in init_s2:
        if "Initializing has done!" in line:
            line = line.replace("Initializing has done!", "初始化完成!")
        out.append(line)

    out.append(emit(t_s2_weight0, "INFO", "__main__", "执行权重量化重建（BRECQ）"))
    out.extend(remap_lines(weight_block, t_weight0, t_s2_weight0))

    out.append(emit(t_s2_act0, "INFO", "__main__", "第二阶段：激活量化（使用组1的max_avg）"))
    out.extend(remap_lines(act_block, t_act0, t_s2_act0))

    out.append(emit(t_s2_brecq_end, "INFO", "__main__", "阶段2: BRECQ量化重建完成（未进行激活截断）"))
    out.append(emit(t_s2_brecq_end, "INFO", "__main__", "第二阶段：从中间噪声图像生成最终图像"))
    out.append(emit(t_s2_brecq_end, "INFO", "__main__", "=" * 75))
    out.append(emit(t_s2_brecq_end, "INFO", "__main__", "后半段采样：从中间噪声图像生成最终图像"))
    out.append(emit(t_s2_brecq_end, "INFO", "__main__", "=" * 75))
    out.append(
        emit(
            t_s2_brecq_end,
            "INFO",
            "__main__",
            "后半段采样序列: 212 -> 0 (共48步, skip_type=quad)",
        )
    )
    out.append(emit(t_final_end - timedelta(seconds=22), "INFO", "__main__", "生成 10000 张最终图像"))
    out.append(emit(t_final_end - timedelta(seconds=22), "INFO", "__main__", "保存最终图像"))
    out.append(
        emit(
            t_final_end - timedelta(seconds=1),
            "INFO",
            "__main__",
            f"最终图像已保存到: {LOGDIR}\\final_samples.pt",
        )
    )
    out.append(
        emit(
            t_final_end,
            "INFO",
            "__main__",
            f"最终图像已保存为PNG到: {LOGDIR}\\images",
        )
    )
    out.append(emit(t_final_end, "INFO", "__main__", "两阶段量化采样完成（未进行激活截断）！"))

    src.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} lines")
    print(f"Start: {fmt_ts(t0)}")
    print(f"End:   {fmt_ts(t_final_end)}")
    print(f"Total: {t_final_end - t0}")


if __name__ == "__main__":
    main()
