"""Project FAST-LIO map-frame points into rectified stereo model inputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


T_LIDAR_FROM_IMU = np.array(
    [
        [-0.00584021, 0.99995500, -0.00748795, -0.056199899903680],
        [-0.99987300, -0.00572824, 0.01488950, 0.002601436907349],
        [0.01484590, 0.00757395, 0.99986100, 0.177498078345765],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=60)
    parser.add_argument("--val-count", type=int, default=7)
    parser.add_argument("--max-depth", type=float, default=30.0)
    return parser.parse_args()


def quaternion_xyzw_to_rotation(quaternion):
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_left_timestamps(path):
    result = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["side"] == "left":
                result.append((row["frame_index"], int(row["timestamp_ns"])))
    return result


def transform_points(points_map, position_map_imu, orientation_map_imu):
    rotation_map_imu = quaternion_xyzw_to_rotation(orientation_map_imu)
    points_imu = (points_map - position_map_imu) @ rotation_map_imu
    points_lidar = (
        points_imu @ T_LIDAR_FROM_IMU[:3, :3].T + T_LIDAR_FROM_IMU[:3, 3]
    )
    return points_lidar


def main():
    args = parse_args()
    sequence = args.sequence_root.resolve()
    output_root = args.output_root.resolve()
    sparse_root = output_root / "sparse_fastlio"
    sparse_root.mkdir(parents=True, exist_ok=True)

    with open(sequence / "calibration" / "intrinsics.json", encoding="utf-8") as f:
        intrinsics = json.load(f)
    with open(sequence / "calibration" / "extrinsics.json", encoding="utf-8") as f:
        extrinsics = json.load(f)

    K_full = np.asarray(intrinsics["Cam_Rect_L"]["K"], dtype=np.float64)
    K_effective = K_full.copy() * 0.5
    K_effective[2, 2] = 1.0
    K_effective[0, 2] -= 102.0
    K_effective[1, 2] -= 77.0
    lidar_to_cam = np.asarray(extrinsics["LiDAR_to_Cam_L"]["RT"], dtype=np.float64)
    cam_to_rect = np.asarray(
        extrinsics["Cam_L_to_Cam_L_Rect"]["R"], dtype=np.float64
    )
    stereo_translation_mm = np.asarray(
        extrinsics["Cam_R_to_Cam_L"]["T"], dtype=np.float64
    ).reshape(3)
    baseline_m = float(np.linalg.norm(stereo_translation_mm) / 1000.0)

    trajectory = np.load(sequence / "lidar_fastlio" / "trajectory.npz")
    fastlio_times = trajectory["header_time_ns"].astype(np.int64)
    positions = trajectory["position"].astype(np.float64)
    orientations = trajectory["orientation"].astype(np.float64)
    image_frames = load_left_timestamps(sequence / "images" / "timestamps.tsv")

    data_root = sequence.parent
    samples = []
    point_counts = []
    deltas_ms = []
    for frame, image_time in image_frames:
        fastlio_index = int(np.argmin(np.abs(fastlio_times - image_time)))
        delta_ms = float(abs(int(fastlio_times[fastlio_index]) - image_time) / 1e6)
        cloud = np.load(
            sequence / "lidar_fastlio" / "frames" / f"{fastlio_index:06d}.npz"
        )
        points_lidar = transform_points(
            cloud["xyz"].astype(np.float64),
            positions[fastlio_index],
            orientations[fastlio_index],
        )
        points_cam = (
            points_lidar @ lidar_to_cam[:3, :3].T + lidar_to_cam[:3, 3]
        )
        points_rect = points_cam @ cam_to_rect.T
        depth = points_rect[:, 2]
        projected = points_rect @ K_effective.T
        u = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int64)
        v = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int64)
        valid = (
            np.isfinite(depth)
            & (depth > 0)
            & (depth <= args.max_depth)
            & (u >= 0)
            & (u < 820)
            & (v >= 0)
            & (v < 614)
        )
        sparse_path = sparse_root / f"{frame}.npz"
        np.savez_compressed(
            sparse_path,
            u=u[valid].astype(np.int32),
            v=v[valid].astype(np.int32),
            depth_gt_m=depth[valid].astype(np.float32),
            effective_rectified_K=K_effective,
            source_fastlio_index=np.int32(fastlio_index),
            image_time_ns=np.int64(image_time),
            fastlio_time_ns=np.int64(fastlio_times[fastlio_index]),
            time_delta_ms=np.float32(delta_ms),
        )
        relative = lambda path: path.relative_to(data_root).as_posix()
        samples.append(
            {
                "sequence": sequence.name,
                "frame": frame,
                "left": relative(
                    sequence / "images_rectified" / "left_out" / f"{frame}.png"
                ),
                "right": relative(
                    sequence / "images_rectified" / "right_out" / f"{frame}.png"
                ),
                "dense": relative(sequence / "depth_gt_rectified" / f"{frame}.png"),
                "sparse_npz": str(sparse_path),
                "baseline_m": baseline_m,
            }
        )
        point_counts.append(int(valid.sum()))
        deltas_ms.append(delta_ms)

    train_end = args.train_count
    val_end = train_end + args.val_count
    if val_end >= len(samples):
        raise ValueError("train-count + val-count must leave at least one test sample")
    split = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
        "metadata": {
            "sequence": sequence.name,
            "rgb": ["images_rectified/left_out", "images_rectified/right_out"],
            "dense": "depth_gt_rectified",
            "sparse": "lidar_fastlio map->IMU->LiDAR->Cam_L->Cam_Rect_L",
            "baseline_m": baseline_m,
            "effective_rectified_K": K_effective.tolist(),
            "point_count_min": min(point_counts),
            "point_count_mean": float(np.mean(point_counts)),
            "point_count_max": max(point_counts),
            "time_delta_ms_max": max(deltas_ms),
            "time_delta_ms_mean": float(np.mean(deltas_ms)),
        },
    }
    args.split_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.split_json, "w", encoding="utf-8") as handle:
        json.dump(split, handle, ensure_ascii=False, indent=2)
    print(
        f"Wrote {args.split_json}: train={len(split['train'])}, "
        f"val={len(split['val'])}, test={len(split['test'])}"
    )
    print(
        f"Projected points min/mean/max: {min(point_counts)}/"
        f"{np.mean(point_counts):.1f}/{max(point_counts)}"
    )
    print(
        f"Nearest FAST-LIO delta mean/max: {np.mean(deltas_ms):.2f}/"
        f"{max(deltas_ms):.2f} ms"
    )


if __name__ == "__main__":
    main()
