import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import kittidc as DA
from metrics import Metrics
from models import get_model


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_model(args):
    model = get_model(args["model"]["model_name"])(args)
    checkpoint = torch.load(args["path"]["loadmodel"], map_location="cpu")

    state_name = None
    for key in ["state_dict", "net", "model"]:
        if key in checkpoint:
            state_name = key
            break

    weights = checkpoint if state_name is None else checkpoint[state_name]
    state = OrderedDict()
    for key, value in weights.items():
        state[key[7:] if key.startswith("module.") else key] = value

    model.load_state_dict(state)
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


def to_cuda(batch):
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def infer_depth(model, batch, max_depth):
    with torch.no_grad():
        start = time.time()
        output = model(batch)
        if "depth" in output:
            pred = output["depth"]
        else:
            focal = batch["K"][:, 1][:, None, None, None].type_as(batch["rgb1"])
            baseline = batch["K"][:, 0][:, None, None, None].type_as(batch["rgb1"])
            pred = (focal * baseline) / (output["disp"] + 1e-8)
        pred = torch.clamp(pred, 1, max_depth)
        torch.cuda.synchronize()
        runtime = time.time() - start
    return pred, runtime


def summarize(rows, keys):
    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configure/kitti_partial_val_cfg.yaml")
    parser.add_argument("--out-dir", default="./outputs/batch_metrics")
    parser.add_argument("--max-samples", type=int, default=None)
    args_cli = parser.parse_args()

    args = yaml.safe_load(open(args_cli.config, "r", encoding="utf-8"))
    os.environ["CUDA_VISIBLE_DEVICES"] = args["test"]["gpus_id"]

    torch.manual_seed(args["environment"]["seed"])
    torch.cuda.manual_seed_all(args["environment"]["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = DA.KITTIDC(args, args["test"]["split"])
    total = len(dataset) if args_cli.max_samples is None else min(args_cli.max_samples, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(total)))
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    model = load_model(args)
    metric = Metrics(args).cuda()

    rows = []
    for idx, batch in enumerate(loader):
        batch_cuda = to_cuda(batch)
        pred, runtime = infer_depth(model, batch_cuda, args["model"]["max_depth"])
        result = metric(pred, batch_cuda["gt_depth"])[0].detach().cpu().numpy()
        valid = int((batch["gt_depth"] > 1e-4).sum().item())
        sample = dataset.sample_list[idx]
        row = {
            "index": idx,
            "left": sample["left"],
            "RMSE_mm": float(result[0] * 1000.0),
            "MAE_mm": float(result[1] * 1000.0),
            "iRMSE_1_per_km": float(result[2] * 1000.0),
            "iMAE_1_per_km": float(result[3] * 1000.0),
            "REL": float(result[4]),
            "Delta1": float(result[5]),
            "Delta2": float(result[6]),
            "Delta3": float(result[7]),
            "runtime_seconds": float(runtime),
            "valid_gt_pixels": valid,
        }
        rows.append(row)
        print(
            f"[{idx + 1:02d}/{total:02d}] RMSE={row['RMSE_mm']:.3f} mm "
            f"MAE={row['MAE_mm']:.3f} mm REL={row['REL']:.5f} "
            f"time={row['runtime_seconds']:.3f}s"
        )

    metric_keys = [
        "RMSE_mm",
        "MAE_mm",
        "iRMSE_1_per_km",
        "iMAE_1_per_km",
        "REL",
        "Delta1",
        "Delta2",
        "Delta3",
        "runtime_seconds",
        "valid_gt_pixels",
    ]
    summary = {
        "num_samples": total,
        "config": args_cli.config,
        "summary": summarize(rows, metric_keys),
        "per_sample": rows,
    }

    out_dir = ensure_dir(args_cli.out_dir)
    csv_path = os.path.join(out_dir, "per_sample_metrics.csv")
    json_path = os.path.join(out_dir, "metrics_summary.json")
    txt_path = os.path.join(out_dir, "metrics_summary.txt")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [f"Samples: {total}", ""]
    for key in metric_keys:
        stat = summary["summary"][key]
        lines.append(
            f"{key}: mean={stat['mean']:.6f}, std={stat['std']:.6f}, "
            f"median={stat['median']:.6f}, min={stat['min']:.6f}, max={stat['max']:.6f}"
        )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("")
    print("Saved metrics to:", os.path.abspath(out_dir))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
