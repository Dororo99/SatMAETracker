"""
Visualize satellite image alignment with GT HD map polylines.

Overlays GT map elements (ped_crossing, divider, boundary) onto
the AID4AD satellite image to verify spatial alignment.

Usage:
    python tools/visualize_sat_alignment.py \
        --ann_file ./datasets/nuscenes/nuscenes_map_infos_train_newsplit_third.pkl \
        --sat_root /workspace/sumin/2026.Online-Mapping.for-NeurIPS-2026/datasets/nuscenes/AID4AD_frames \
        --output_dir ./vis_sat_alignment \
        --num_samples 10
"""
import os
import argparse
import pickle
import numpy as np
import cv2
from PIL import Image
from pyquaternion import Quaternion

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file',
                        default='./datasets/nuscenes/nuscenes_map_infos_train_newsplit_third.pkl')
    parser.add_argument('--data_root', default='./datasets/nuscenes')
    parser.add_argument('--sat_root',
                        default='/workspace/sumin/2026.Online-Mapping.for-NeurIPS-2026/datasets/nuscenes/AID4AD_frames')
    parser.add_argument('--output_dir', default='./vis_sat_alignment')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated sample indices, e.g. "0,755,1708,3858"')
    parser.add_argument('--roi_size', nargs=2, type=float, default=[60, 30])
    parser.add_argument('--sat_img_size', nargs=2, type=int, default=[200, 100])  # W, H
    return parser.parse_args()


def build_sat_index(sat_root):
    """Build token -> file path mapping."""
    token2file = {}
    for loc in os.listdir(sat_root):
        loc_dir = os.path.join(sat_root, loc)
        if not os.path.isdir(loc_dir):
            continue
        for fname in os.listdir(loc_dir):
            if not fname.endswith('.png'):
                continue
            token = fname.split('_', 1)[1].replace('.png', '')
            token2file[token] = os.path.join(loc_dir, fname)
    print(f'Indexed {len(token2file)} satellite frames')
    return token2file


def world_to_pixel(coords, roi_size, img_size):
    """Convert ego-centric world coords to pixel coords.

    Args:
        coords: (N, 2) in ego frame, x=forward, y=left
        roi_size: (roi_x, roi_y) e.g. (60, 30)
        img_size: (W, H) e.g. (200, 100)
    Returns:
        pixel_coords: (N, 2) in (px_x, px_y)
    """
    # Ego coords: x ∈ [-30, 30], y ∈ [-15, 15]
    # Pixel coords: px_x ∈ [0, W], px_y ∈ [0, H]
    roi_x, roi_y = roi_size
    W, H = img_size

    px_x = (coords[:, 0] + roi_x / 2) / roi_x * W
    px_y = (coords[:, 1] + roi_y / 2) / roi_y * H

    # Flip y-axis (image y=0 is top, ego y=+ is left)
    px_y = H - px_y

    return np.stack([px_x, px_y], axis=1).astype(int)


def draw_polylines_on_image(img, polylines, roi_size, img_size, colors, thickness=2):
    """Draw GT polylines on satellite image."""
    vis = img.copy()
    cat2id = {'ped_crossing': 0, 'divider': 1, 'boundary': 2}

    for label_id, geom_list in polylines.items():
        color = colors.get(label_id, (255, 255, 255))
        for geom in geom_list:
            if hasattr(geom, 'coords'):
                coords = np.array(geom.coords)
            elif hasattr(geom, 'exterior'):
                coords = np.array(geom.exterior.coords)
            else:
                continue

            if len(coords) < 2:
                continue

            pixels = world_to_pixel(coords, roi_size, img_size)
            for j in range(len(pixels) - 1):
                pt1 = tuple(pixels[j])
                pt2 = tuple(pixels[j + 1])
                cv2.line(vis, pt1, pt2, color, thickness)

    return vis


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load annotations
    with open(args.ann_file, 'rb') as f:
        ann_data = pickle.load(f)
    samples = ann_data if isinstance(ann_data, list) else ann_data

    # Build satellite index
    token2file = build_sat_index(args.sat_root)

    # Map extractor
    roi_size = tuple(args.roi_size)
    map_extractor = NuscMapExtractor(args.data_root, roi_size)

    cat2id = {'ped_crossing': 0, 'divider': 1, 'boundary': 2}
    colors = {
        0: (0, 255, 0),    # ped_crossing: green
        1: (0, 0, 255),    # divider: red
        2: (255, 0, 0),    # boundary: blue
    }

    # Determine which samples to visualize
    if args.indices is not None:
        target_indices = [int(x) for x in args.indices.split(',')]
    else:
        target_indices = None

    count = 0
    for idx, sample in enumerate(samples):
        if target_indices is not None and idx not in target_indices:
            continue
        token = sample['token']
        if token not in token2file:
            continue

        location = sample['location']

        # Compute lidar2global (same as NuscDataset.get_sample)
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']

        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_translation = [float(x) for x in lidar2global_translation]
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        # Get GT map geometries
        map_geoms = map_extractor.get_map_geom(
            location, lidar2global_translation, lidar2global_rotation)

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in cat2id:
                map_label2geom[cat2id[k]] = v

        # Load satellite image
        sat_img = Image.open(token2file[token]).convert('RGB')
        sat_img = sat_img.resize(tuple(args.sat_img_size), Image.LANCZOS)
        sat_img = np.array(sat_img)

        # Draw GT polylines
        vis = draw_polylines_on_image(
            sat_img, map_label2geom, roi_size,
            tuple(args.sat_img_size), colors, thickness=2)

        # Save
        out_path = os.path.join(args.output_dir, f'{count:04d}_{token[:8]}_{location}.png')

        # Also create a side-by-side: original | overlay
        combined = np.concatenate([sat_img, vis], axis=1)
        cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f'[{count}] Saved: {out_path}')

        count += 1
        if count >= args.num_samples:
            break

    print(f'\nDone. {count} visualizations saved to {args.output_dir}/')
    print('Legend: Green=ped_crossing, Red=divider, Blue=boundary')


if __name__ == '__main__':
    main()
