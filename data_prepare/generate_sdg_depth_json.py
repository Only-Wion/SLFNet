"""Generate an SLFNet split file for one organized SDG-Depth sequence."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    sequence = args.sequence_root.resolve()
    data_root = sequence.parent
    if args.train_ratio < 0 or args.val_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio must be less than 1")

    left_dir = sequence / "images_rectified" / "left"
    right_dir = sequence / "images_rectified" / "right"
    dense_dir = sequence / "depth_gt_rectified"
    sparse_dir = sequence / "lidar_fake" / "samples"
    collections = [
        {path.stem for path in directory.glob(f"*.{suffix}")}
        for directory, suffix in (
            (left_dir, "png"),
            (right_dir, "png"),
            (dense_dir, "png"),
            (sparse_dir, "npz"),
        )
    ]
    common = sorted(set.intersection(*collections))
    if not common:
        raise RuntimeError(f"No aligned samples found under {sequence}")
    if any(frames != set(common) for frames in collections):
        raise RuntimeError("Modality frame sets do not match exactly")

    with open(sequence / "calibration" / "extrinsics.json", encoding="utf-8") as f:
        extrinsics = json.load(f)
    translation_mm = np.asarray(
        extrinsics["Cam_R_to_Cam_L"]["T"], dtype=np.float64
    ).reshape(3)
    baseline_m = float(np.linalg.norm(translation_mm) / 1000.0)

    samples = []
    for frame in common:
        relative = lambda path: path.relative_to(data_root).as_posix()
        samples.append(
            {
                "sequence": sequence.name,
                "frame": frame,
                "left": relative(left_dir / f"{frame}.png"),
                "right": relative(right_dir / f"{frame}.png"),
                "dense": relative(dense_dir / f"{frame}.png"),
                "sparse_npz": relative(sparse_dir / f"{frame}.npz"),
                "baseline_m": baseline_m,
            }
        )

    train_end = int(len(samples) * args.train_ratio)
    val_end = train_end + int(len(samples) * args.val_ratio)
    output = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
        "metadata": {
            "data_root": str(data_root),
            "sequence": sequence.name,
            "frames": len(samples),
            "baseline_m": baseline_m,
            "warning": "Single-sequence temporal split is for pipeline validation only.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(
        f"Wrote {args.output}: train={len(output['train'])}, "
        f"val={len(output['val'])}, test={len(output['test'])}, "
        f"baseline={baseline_m:.6f} m"
    )


if __name__ == "__main__":
    main()
