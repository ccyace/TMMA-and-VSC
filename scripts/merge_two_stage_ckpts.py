"""Merge stage1/stage2 ckpts into ckpt_two_stage.pth (for runs that finished before combined save was added)."""
import argparse
import os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True, help="directory containing ckpt_stage1_group0.pth and ckpt_stage2_group1.pth")
    args = p.parse_args()
    s1 = os.path.join(args.logdir, "ckpt_stage1_group0.pth")
    s2 = os.path.join(args.logdir, "ckpt_stage2_group1.pth")
    if not os.path.isfile(s1):
        raise FileNotFoundError(s1)
    if not os.path.isfile(s2):
        raise FileNotFoundError(s2)
    combined = {
        "stage1_group0": torch.load(s1, map_location="cpu"),
        "stage2_group1": torch.load(s2, map_location="cpu"),
        "meta": {
            "stage1_ckpt": "ckpt_stage1_group0.pth",
            "stage2_ckpt": "ckpt_stage2_group1.pth",
        },
    }
    out = os.path.join(args.logdir, "ckpt_two_stage.pth")
    torch.save(combined, out)
    print("saved:", out)
    print("keys:", list(combined.keys()))


if __name__ == "__main__":
    main()
