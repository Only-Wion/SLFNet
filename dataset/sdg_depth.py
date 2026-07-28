"""Dataset adapter for organized SDG-Depth stereo/LiDAR sequences."""

import json
import os

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from . import BaseDataset


def _depth_png_to_meters(path):
    depth = np.asarray(Image.open(path))
    if depth.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth PNG, got {depth.dtype}: {path}")
    return depth.astype(np.float32) / 256.0


def _z_buffer(u, v, depth, height, width):
    valid = (
        np.isfinite(depth)
        & (depth > 0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    flat = np.full(height * width, np.inf, dtype=np.float32)
    indices = v[valid].astype(np.int64) * width + u[valid].astype(np.int64)
    np.minimum.at(flat, indices, depth[valid].astype(np.float32))
    flat[~np.isfinite(flat)] = 0
    return flat.reshape(height, width)


class SDGDepth(BaseDataset):
    """Load native SDG files and rasterize sparse left/right depth on demand."""

    def __init__(self, args, mode):
        super().__init__(args, mode)
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Unsupported mode: {mode}")

        self.root = args["path"]["data_root"]
        with open(args["path"]["split_json"], encoding="utf-8") as handle:
            self.sample_list = json.load(handle)[mode]

        data_args = args.get("data", {})
        self.resize_h, self.resize_w = data_args.get("resize_size", [768, 1024])
        self.input_h, self.input_w = data_args.get("input_size", [614, 820])
        self.crop_top, self.crop_left = data_args.get("input_crop", [77, 102])
        self.output_crop = data_args.get("output_crop")

    def __len__(self):
        return len(self.sample_list)

    def _path(self, sample, key):
        return os.path.join(self.root, sample[key])

    def _resize_and_crop(self, image, resample):
        image = image.resize((self.resize_w, self.resize_h), resample=resample)
        return image.crop(
            (
                self.crop_left,
                self.crop_top,
                self.crop_left + self.input_w,
                self.crop_top + self.input_h,
            )
        )

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        left = self._resize_and_crop(
            Image.open(self._path(sample, "left")).convert("RGB"), Image.BILINEAR
        )
        right = self._resize_and_crop(
            Image.open(self._path(sample, "right")).convert("RGB"), Image.BILINEAR
        )
        dense_full = Image.fromarray(
            _depth_png_to_meters(self._path(sample, "dense")), mode="F"
        )
        dense = self._resize_and_crop(dense_full, Image.NEAREST)

        with np.load(self._path(sample, "sparse_npz")) as sparse:
            u = sparse["u"].astype(np.int64)
            v = sparse["v"].astype(np.int64)
            depth = sparse["depth_gt_m"].astype(np.float32)
            K_np = sparse["effective_rectified_K"].astype(np.float32)

        baseline = float(sample["baseline_m"])
        disparity = K_np[0, 0] * baseline / np.maximum(depth, 1e-8)
        right_u = np.rint(u - disparity).astype(np.int64)
        depth_left = _z_buffer(u, v, depth, self.input_h, self.input_w)
        depth_right = _z_buffer(right_u, v, depth, self.input_h, self.input_w)

        K1 = np.zeros((3, 4), dtype=np.float32)
        K2 = np.zeros((3, 4), dtype=np.float32)
        K1[:, :3] = K_np
        K2[:, :3] = K_np
        K2[0, 3] = -K_np[0, 0] * baseline

        if self.output_crop:
            crop_h, crop_w = self.output_crop
            if crop_h > self.input_h or crop_w > self.input_w:
                raise ValueError(f"output_crop {self.output_crop} exceeds input")
            top = self.input_h - crop_h
            left_offset = (self.input_w - crop_w) // 2
            left = TF.crop(left, top, left_offset, crop_h, crop_w)
            right = TF.crop(right, top, left_offset, crop_h, crop_w)
            dense = TF.crop(dense, top, left_offset, crop_h, crop_w)
            depth_left = depth_left[top : top + crop_h, left_offset : left_offset + crop_w]
            depth_right = depth_right[top : top + crop_h, left_offset : left_offset + crop_w]
            K1[0, 2] -= left_offset
            K1[1, 2] -= top
            K2[0, 2] -= left_offset
            K2[1, 2] -= top

        rgb1 = TF.normalize(
            TF.to_tensor(left), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        rgb2 = TF.normalize(
            TF.to_tensor(right), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        dep1 = torch.from_numpy(np.ascontiguousarray(depth_left))[None]
        dep2 = torch.from_numpy(np.ascontiguousarray(depth_right))[None]
        gt_depth = TF.to_tensor(np.asarray(dense, dtype=np.float32))
        calib = torch.tensor([baseline, K_np[0, 0]], dtype=torch.float32)
        disp1 = torch.where(dep1 > 0, calib.prod() / dep1.clamp_min(1e-8), 0)
        disp2 = torch.where(dep2 > 0, calib.prod() / dep2.clamp_min(1e-8), 0)

        return {
            "rgb1": rgb1,
            "rgb2": rgb2,
            "dep1": dep1,
            "dep2": dep2,
            "disp1": disp1,
            "disp2": disp2,
            "K1": torch.from_numpy(K1),
            "K2": torch.from_numpy(K2),
            "gt_depth": gt_depth,
            "K": calib,
        }
