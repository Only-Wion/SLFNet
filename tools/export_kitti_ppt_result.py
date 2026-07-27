import argparse
import json
import os
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset import kittidc as DA
from metrics import Metrics
from models import get_model


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_model(args):
    model = get_model(args["model"]["model_name"])(args)

    if args["path"]["loadmodel"] is not None:
        pretrain_dict = torch.load(args["path"]["loadmodel"], map_location="cpu")
        state_name = None
        for key in ["state_dict", "net", "model"]:
            if key in pretrain_dict:
                state_name = key
                break

        weights = pretrain_dict if state_name is None else pretrain_dict[state_name]
        new_state_dict = OrderedDict()
        for key, value in weights.items():
            name = key[7:] if key.startswith("module.") else key
            new_state_dict[name] = value

        model.load_state_dict(new_state_dict)

    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


def to_cuda(batch):
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def predict_depth(model, batch, max_depth):
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
        elapsed = time.time() - start
    return pred, elapsed


def tensor_to_rgb(tensor):
    image = tensor.detach().cpu()[0].permute(1, 2, 0).numpy()
    image = (image * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def tensor_depth(tensor):
    return tensor.detach().cpu()[0, 0].numpy().astype(np.float32)


def colorize_depth(depth, max_depth=90.0):
    valid = depth > 1e-4
    scaled = np.clip(depth, 0, max_depth) / max_depth
    gray = (255 - scaled * 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    color = cv2.applyColorMap(gray, cmap)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def colorize_error(error, valid, max_error=5.0):
    scaled = np.clip(error, 0, max_error) / max_error
    gray = (scaled * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def save_uint16_depth(path, depth):
    encoded = np.clip(depth * 256.0, 0, 65535).astype(np.uint16)
    Image.fromarray(encoded).save(path)


def make_metric_text(metrics, runtime):
    return [
        "SLFNet KITTI Depth Completion",
        "",
        f"RMSE  {metrics['RMSE_mm']:.3f} mm",
        f"MAE   {metrics['MAE_mm']:.3f} mm",
        f"iRMSE {metrics['iRMSE_1_per_km']:.3f} 1/km",
        f"iMAE  {metrics['iMAE_1_per_km']:.3f} 1/km",
        f"REL   {metrics['REL']:.5f}",
        f"Delta1 {metrics['Delta1']:.4f}",
        "",
        f"Runtime {runtime:.3f} s",
        f"Valid GT pixels {metrics['valid_gt_pixels']}",
    ]


def add_label(image, label):
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, canvas.width, 24], fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255), font=font)
    return canvas


def make_composite(paths, metrics_lines, out_path):
    panels = [
        ("Left RGB", np.array(Image.open(paths["left_rgb"]).convert("RGB"))),
        ("Sparse LiDAR", np.array(Image.open(paths["sparse_color"]).convert("RGB"))),
        ("Prediction", np.array(Image.open(paths["pred_color"]).convert("RGB"))),
        ("Ground Truth", np.array(Image.open(paths["gt_color"]).convert("RGB"))),
        ("Abs Error", np.array(Image.open(paths["error_color"]).convert("RGB"))),
    ]

    target_w = 520
    target_h = 170
    labeled = []
    for label, image in panels:
        img = Image.fromarray(image).resize((target_w, target_h), Image.BILINEAR)
        labeled.append(add_label(np.array(img), label))

    metric_panel = Image.new("RGB", (target_w, target_h), (248, 248, 248))
    draw = ImageDraw.Draw(metric_panel)
    font = ImageFont.load_default()
    y = 14
    for line in metrics_lines:
        draw.text((18, y), line, fill=(20, 20, 20), font=font)
        y += 14
    labeled.append(metric_panel)

    gap = 18
    margin = 24
    width = margin * 2 + target_w * 3 + gap * 2
    height = margin * 2 + target_h * 2 + gap
    composite = Image.new("RGB", (width, height), (255, 255, 255))
    for idx, panel in enumerate(labeled):
        row = idx // 3
        col = idx % 3
        x = margin + col * (target_w + gap)
        y = margin + row * (target_h + gap)
        composite.paste(panel, (x, y))

    composite.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configure/kitti_partial_val_cfg.yaml")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", default="./outputs/ppt_kitti_slfnet")
    parser.add_argument("--max-depth", type=float, default=90.0)
    parser.add_argument("--max-error", type=float, default=5.0)
    cli = parser.parse_args()

    args = yaml.safe_load(open(cli.config, "r", encoding="utf-8"))
    os.environ["CUDA_VISIBLE_DEVICES"] = args["test"]["gpus_id"]

    torch.manual_seed(args["environment"]["seed"])
    torch.cuda.manual_seed_all(args["environment"]["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = DA.KITTIDC(args, args["test"]["split"])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [cli.index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = load_model(args)
    metric = Metrics(args).cuda()
    batch = next(iter(loader))
    batch_cuda = to_cuda(batch)

    pred, runtime = predict_depth(model, batch_cuda, args["model"]["max_depth"])
    metric_result = metric(pred, batch_cuda["gt_depth"])[0].detach().cpu().numpy()

    pred_np = tensor_depth(pred)
    sparse_np = tensor_depth(batch["dep1"])
    gt_np = tensor_depth(batch["gt_depth"])
    left_rgb = tensor_to_rgb(batch["rgb1"])
    valid = gt_np > 1e-4
    error = np.zeros_like(pred_np, dtype=np.float32)
    error[valid] = np.abs(pred_np[valid] - gt_np[valid])

    sample = dataset.sample_list[cli.index]
    stem = os.path.splitext(os.path.basename(sample["left"]))[0]
    drive = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(sample["left"]))))
    out_dir = ensure_dir(os.path.join(cli.out_dir, f"{drive}_{stem}"))

    paths = {
        "left_rgb": os.path.join(out_dir, "01_left_rgb.png"),
        "sparse_color": os.path.join(out_dir, "02_sparse_lidar_color.png"),
        "pred_color": os.path.join(out_dir, "03_prediction_depth_color.png"),
        "gt_color": os.path.join(out_dir, "04_ground_truth_depth_color.png"),
        "error_color": os.path.join(out_dir, "05_abs_error_color.png"),
        "pred_uint16": os.path.join(out_dir, "prediction_depth_uint16_kitti.png"),
        "pred_npy": os.path.join(out_dir, "prediction_depth_meters.npy"),
        "metrics_json": os.path.join(out_dir, "metrics.json"),
        "metrics_txt": os.path.join(out_dir, "metrics.txt"),
        "composite": os.path.join(out_dir, "ppt_summary.png"),
    }

    Image.fromarray(left_rgb).save(paths["left_rgb"])
    Image.fromarray(colorize_depth(sparse_np, cli.max_depth)).save(paths["sparse_color"])
    Image.fromarray(colorize_depth(pred_np, cli.max_depth)).save(paths["pred_color"])
    Image.fromarray(colorize_depth(gt_np, cli.max_depth)).save(paths["gt_color"])
    Image.fromarray(colorize_error(error, valid, cli.max_error)).save(paths["error_color"])
    save_uint16_depth(paths["pred_uint16"], pred_np)
    np.save(paths["pred_npy"], pred_np)

    metrics = {
        "sample_index": cli.index,
        "sample_left": sample["left"],
        "RMSE_mm": float(metric_result[0] * 1000.0),
        "MAE_mm": float(metric_result[1] * 1000.0),
        "iRMSE_1_per_km": float(metric_result[2] * 1000.0),
        "iMAE_1_per_km": float(metric_result[3] * 1000.0),
        "REL": float(metric_result[4]),
        "Delta1": float(metric_result[5]),
        "Delta2": float(metric_result[6]),
        "Delta3": float(metric_result[7]),
        "runtime_seconds": float(runtime),
        "valid_gt_pixels": int(valid.sum()),
    }

    with open(paths["metrics_json"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    metric_lines = make_metric_text(metrics, runtime)
    with open(paths["metrics_txt"], "w", encoding="utf-8") as f:
        f.write("\n".join(metric_lines) + "\n")

    make_composite(paths, metric_lines, paths["composite"])

    print("Saved PPT-ready result to:", os.path.abspath(out_dir))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
