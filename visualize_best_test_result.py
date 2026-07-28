"""Visualize the test sample with the lowest absolute relative error."""

import argparse
import json
import os

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


cli_args = parse_args()
with open(cli_args.config, encoding="utf-8") as handle:
    args = yaml.safe_load(handle)
os.environ["CUDA_VISIBLE_DEVICES"] = args["test"]["gpus_id"]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset.sdg_depth import SDGDepth
from models import get_model


def load_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    for key in ("state_dict", "net", "model"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    state = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(state)


def denormalize_rgb(tensor):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = tensor.cpu().numpy() * std + mean
    return np.clip(np.transpose(image, (1, 2, 0)), 0, 1)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dataset = SDGDepth(args, args["test"]["split"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = get_model(args["model"]["model_name"])(args).cuda()
    load_weights(model, args["path"]["loadmodel"])
    model.eval()

    best = None
    with torch.no_grad():
        for index, batch in enumerate(loader):
            rgb_cpu = batch["rgb1"][0].clone()
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            output = model(batch)
            focal = batch["K"][:, 1, None, None, None]
            baseline = batch["K"][:, 0, None, None, None]
            prediction = torch.clamp(
                focal * baseline / output["disp"].clamp_min(1e-8),
                1,
                args["model"]["max_depth"],
            )[0, 0]
            ground_truth = batch["gt_depth"][0, 0]
            valid = (ground_truth > 1e-4) & (
                ground_truth < args["model"]["max_depth"]
            )
            difference = prediction - ground_truth
            relative_error = torch.abs(difference[valid]) / ground_truth[valid]
            rel = float(relative_error.mean().cpu())
            rmse_mm = float(torch.sqrt(torch.mean(difference[valid] ** 2)).cpu() * 1000)
            mae_mm = float(torch.mean(torch.abs(difference[valid])).cpu() * 1000)
            if best is None or rel < best["rel"]:
                error_map = torch.full_like(ground_truth, torch.nan)
                error_map[valid] = torch.abs(difference[valid]) / ground_truth[valid]
                best = {
                    "index": index,
                    "frame": dataset.sample_list[index]["frame"],
                    "rel": rel,
                    "rmse_mm": rmse_mm,
                    "mae_mm": mae_mm,
                    "rgb": denormalize_rgb(rgb_cpu),
                    "prediction": prediction.cpu().numpy(),
                    "ground_truth": ground_truth.cpu().numpy(),
                    "error": error_map.cpu().numpy(),
                }

    valid_depth = best["ground_truth"] > 0
    depth_values = best["ground_truth"][valid_depth]
    depth_min, depth_max = np.percentile(depth_values, [1, 99])
    valid_error = best["error"][np.isfinite(best["error"])]
    error_max = max(float(np.percentile(valid_error, 99)), 0.05)

    figure, axes = plt.subplots(2, 2, figsize=(16, 7.2), constrained_layout=True)
    axes[0, 0].imshow(best["rgb"])
    axes[0, 0].set_title(f"Rectified left RGB — frame {best['frame']}")

    prediction_plot = axes[0, 1].imshow(
        best["prediction"], cmap="turbo", vmin=depth_min, vmax=depth_max
    )
    axes[0, 1].set_title(f"Predicted depth (m) — RMSE {best['rmse_mm']:.1f} mm")
    figure.colorbar(prediction_plot, ax=axes[0, 1], fraction=0.025, pad=0.02)

    ground_truth_plot = axes[1, 0].imshow(
        np.ma.masked_where(~valid_depth, best["ground_truth"]),
        cmap="turbo",
        vmin=depth_min,
        vmax=depth_max,
    )
    axes[1, 0].set_title("Ground-truth depth (m)")
    figure.colorbar(ground_truth_plot, ax=axes[1, 0], fraction=0.025, pad=0.02)

    error_plot = axes[1, 1].imshow(
        np.ma.masked_invalid(best["error"]),
        cmap="magma",
        vmin=0,
        vmax=error_max,
    )
    axes[1, 1].set_title(
        f"Absolute relative error |pred−gt|/gt — REL {best['rel']:.4f}"
    )
    figure.colorbar(error_plot, ax=axes[1, 1], fraction=0.025, pad=0.02)

    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(
        "SLFNet best test result by relative error"
        f"  |  MAE {best['mae_mm']:.1f} mm",
        fontsize=15,
    )

    output = os.path.abspath(cli_args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    metadata = {key: best[key] for key in ("index", "frame", "rel", "rmse_mm", "mae_mm")}
    with open(os.path.splitext(output)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps({"output": output, **metadata}, indent=2))


if __name__ == "__main__":
    main()
