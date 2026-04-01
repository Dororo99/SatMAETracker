#!/usr/bin/env python
"""
evaluate_final.py — Unified mAP + mAS evaluator for online HD mapping.

Based on:
  Paper:      "Stability Under Scrutiny" (arXiv:2510.10660, ICLR 2026)
  Code:       https://github.com/bhsh0112/MapStableTest
  OpenReview: β=15 for localization, dynamic axis resampling

Key design choices (paper > code where they differ):
  - Presence:      1.0 (consistent) / 0.5 (flickering)   [paper + metrics_fixed.py]
  - Localization:  1 - avg_L1/β with dynamic axis          [code, β=15=y_range/2]
  - Shape:         1 - |κ_cur - κ_prev| / π              [paper = code]
  - ω (loc weight): 0.7                                   [paper]
  - Detection thr:  0.3                                   [paper = code]
  - Both-miss thr:  0.01                                  [code]
  - Interval:       2                                     [code default]

Usage:
    python evaluate_final.py \\
        --submission path/to/submission_vector.json \\
        --dataroot ./datasets/nuscenes \\
        --ann-file ./datasets/nuscenes/nuscenes_map_infos_val_newsplit.pkl
"""

import argparse
import os
import numpy as np
import torch
import mmcv
import prettytable
from functools import partial
from multiprocessing import Pool
from time import time
from collections import defaultdict

from pyquaternion import Quaternion
from shapely.geometry import LineString, box, Polygon, MultiPolygon, MultiLineString, LinearRing
from shapely import ops, affinity, strtree
from scipy.optimize import linear_sum_assignment

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw

# ============================================================
# Configuration
# ============================================================
ROI_SIZE = (60, 30)            # BEV range: 60m (x) × 30m (y)
THRESHOLDS = [0.5, 1.0, 1.5]  # mAP Chamfer distance thresholds (meters)
INTERP_NUM = 200               # arc-length resampling points for mAP
MAS_RESAMPLE_NUM = 50          # arc-length resampling for mAS Hungarian matching
N_WORKERS = 16
INTERVAL = 2                  # mAS frame pair interval
LOC_WEIGHT = 0.7              # ω: localization weight in SI (paper)
BETA = 15.0                   # β: localization normalizer = y_range/2 (15m)
DET_THRESHOLD = 0.3           # presence detection threshold
BOTH_MISS_THRESHOLD = 0.01    # skip pair if both scores < this
N_STABILITY_SAMPLES = 100     # total dynamic resampling points

CAT2ID = {'ped_crossing': 0, 'divider': 1, 'boundary': 2}
ID2CAT = {v: k for k, v in CAT2ID.items()}
MAPS = ['boston-seaport', 'singapore-hollandvillage',
        'singapore-onenorth', 'singapore-queenstown']


# ============================================================
# Section 1: Distance & AP utilities
# ============================================================

def chamfer_distance_batch(pred_lines, gt_lines):
    """Batch Chamfer distance. (m, P, 2) × (n, P, 2) → (m, n)."""
    _, num_pts, coord_dims = pred_lines.shape
    if not isinstance(pred_lines, torch.Tensor):
        pred_lines = torch.tensor(pred_lines)
    if not isinstance(gt_lines, torch.Tensor):
        gt_lines = torch.tensor(gt_lines)
    dist_mat = torch.cdist(
        pred_lines.view(-1, coord_dims),
        gt_lines.view(-1, coord_dims), p=2)
    dist_mat = torch.stack(torch.split(dist_mat, num_pts))
    dist_mat = torch.stack(torch.split(dist_mat, num_pts, dim=-1))
    dist1 = dist_mat.min(-1)[0].sum(-1)
    dist2 = dist_mat.min(-2)[0].sum(-1)
    return ((dist1 + dist2).transpose(0, 1) / (2 * num_pts)).numpy()


def instance_match(pred_lines, scores, gt_lines, thresholds):
    """Greedy matching (confidence-descending) of predictions to GT per threshold."""
    num_preds = pred_lines.shape[0]
    num_gts = gt_lines.shape[0]
    tp_fp_list = []

    if num_gts == 0:
        fp = np.ones(num_preds, dtype=np.float32)
        tp = np.zeros(num_preds, dtype=np.float32)
        return [(tp.copy(), fp.copy()) for _ in thresholds]
    if num_preds == 0:
        return [(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
                for _ in thresholds]

    matrix = chamfer_distance_batch(pred_lines, gt_lines)
    matrix_min = matrix.min(axis=1)
    matrix_argmin = matrix.argmin(axis=1)
    sort_inds = np.argsort(-scores)

    for thr in thresholds:
        tp = np.zeros(num_preds, dtype=np.float32)
        fp = np.zeros(num_preds, dtype=np.float32)
        gt_covered = np.zeros(num_gts, dtype=bool)
        for i in sort_inds:
            if matrix_min[i] <= thr:
                matched_gt = matrix_argmin[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        tp_fp_list.append((tp, fp))
    return tp_fp_list


def average_precision(recalls, precisions):
    """AP via all-point interpolation of PR curve."""
    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]
    mrec = np.hstack((np.zeros((1, 1)), recalls, np.ones((1, 1))))
    mpre = np.hstack((np.zeros((1, 1)), precisions, np.zeros((1, 1))))
    for i in range(mpre.shape[1] - 1, 0, -1):
        mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
    ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]
    return np.sum((mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1])


def interp_fixed_num(vector, num_pts):
    """Arc-length uniform resampling of a polyline to fixed number of points."""
    line = LineString(vector)
    distances = np.linspace(0, line.length, num_pts)
    return np.array(
        [list(line.interpolate(d).coords) for d in distances]).squeeze()


# ============================================================
# Section 2: Dynamic Axis Resampling (OpenReview algorithm)
# ============================================================

def interpolate_along_axis(poly, samples, axis=0):
    """
    Piecewise linear interpolation of a polyline along a specified axis.

    Args:
        poly: (N, 2) array of polyline points
        samples: 1D array of primary-axis values to interpolate at
        axis: 0 = X-sampling (given x, return y), 1 = Y-sampling (given y, return x)

    Returns:
        1D array of interpolated secondary-axis values
    """
    poly = np.array(poly, dtype=np.float64)
    sec = 1 - axis
    if len(poly) == 0:
        return np.zeros_like(samples, dtype=np.float64)
    if len(poly) == 1:
        return np.full_like(samples, poly[0, sec], dtype=np.float64)
    sort_idx = np.argsort(poly[:, axis])
    poly_sorted = poly[sort_idx]
    return np.interp(samples, poly_sorted[:, axis], poly_sorted[:, sec])


def dynamic_axis_resample_deviation(cur_poly, prev_poly, N=N_STABILITY_SAMPLES):
    """
    Compute mean absolute deviation between two polylines using dynamic axis
    selection (per OpenReview response to Weakness-3 / Question-2).

    Algorithm:
    1. For each segment of the reference polyline (cur_poly), classify as
       X-sampling (|dx| >= |dy|) or Y-sampling (|dy| > |dx|).
    2. Group consecutive same-type segments into intervals.
    3. Allocate N total sampling points proportionally to interval lengths.
    4. Within each interval, sample uniformly along the primary axis and
       interpolate the secondary axis for BOTH polylines.
    5. The deviation at each point is |secondary_cur - secondary_prev|.

    Args:
        cur_poly: (M, 2) current frame prediction polyline (reference)
        prev_poly: (M', 2) previous frame prediction polyline (transformed to cur frame)
        N: total number of sampling points

    Returns:
        avg_L1: mean absolute deviation, or -1.0 if not computable
    """
    cur_poly = np.array(cur_poly, dtype=np.float64)
    prev_poly = np.array(prev_poly, dtype=np.float64)

    if len(cur_poly) < 2 or len(prev_poly) < 2:
        return -1.0

    n_segs = len(cur_poly) - 1

    # Step 1: Classify each segment
    seg_is_y = []     # True if Y-sampling (vertical-ish segment)
    seg_lengths = []  # length along primary axis
    for j in range(n_segs):
        dx = abs(cur_poly[j + 1, 0] - cur_poly[j, 0])
        dy = abs(cur_poly[j + 1, 1] - cur_poly[j, 1])
        is_y = dy > dx
        seg_is_y.append(is_y)
        seg_lengths.append(dy if is_y else dx)

    # Step 2: Group consecutive same-type segments into intervals
    intervals = []
    start_idx = 0
    for j in range(1, n_segs):
        if seg_is_y[j] != seg_is_y[start_idx]:
            intervals.append({
                'is_y': seg_is_y[start_idx],
                'start': start_idx,      # first segment index
                'end': j,                # exclusive segment index
                'length': sum(seg_lengths[start_idx:j]),
            })
            start_idx = j
    intervals.append({
        'is_y': seg_is_y[start_idx],
        'start': start_idx,
        'end': n_segs,
        'length': sum(seg_lengths[start_idx:n_segs]),
    })

    # Step 3: Allocate sampling points proportionally
    total_length = sum(iv['length'] for iv in intervals)
    if total_length < 1e-9:
        return -1.0

    K = len(intervals)
    n_points = [max(2, round(N * iv['length'] / total_length)) for iv in intervals]

    # Adjustment mechanism: ensure sum == N
    while sum(n_points) < N:
        # Add to longest-ratio intervals first
        order = sorted(range(K), key=lambda i: intervals[i]['length'], reverse=True)
        for i in order:
            if sum(n_points) >= N:
                break
            n_points[i] += 1
    while sum(n_points) > N:
        # Remove from shortest-ratio intervals first
        order = sorted(range(K), key=lambda i: intervals[i]['length'])
        for i in order:
            if sum(n_points) <= N:
                break
            if n_points[i] > 2:
                n_points[i] -= 1

    # Step 4: Generate samples and compute deviations
    all_deviations = []

    for k, iv in enumerate(intervals):
        # Reference points spanning this interval: point indices [start, end]
        pt_start = iv['start']
        pt_end = iv['end']  # segment end → point index is end (inclusive)
        interval_pts = cur_poly[pt_start:pt_end + 1]

        axis = 1 if iv['is_y'] else 0  # primary sampling axis
        primary_min = interval_pts[:, axis].min()
        primary_max = interval_pts[:, axis].max()
        if primary_max - primary_min < 1e-6:
            continue

        samples = np.linspace(primary_min, primary_max, n_points[k])

        # Interpolate secondary axis for both polylines
        vals_cur = interpolate_along_axis(cur_poly, samples, axis=axis)
        vals_prev = interpolate_along_axis(prev_poly, samples, axis=axis)

        all_deviations.extend(np.abs(vals_cur - vals_prev))

    if len(all_deviations) == 0:
        return -1.0

    return float(np.mean(all_deviations))


# ============================================================
# Section 3: NuScenes GT extraction
# ============================================================

def _split_collections(geom):
    """Split Multi-geometries into a flat list of valid simple geometries."""
    if 'Multi' in geom.geom_type:
        return [g for g in geom.geoms if g.is_valid and not g.is_empty]
    if geom.is_valid and not geom.is_empty:
        return [geom]
    return []


def _get_ped_crossing_contour(polygon, local_patch):
    """Extract ped crossing boundary as a closed polyline."""
    ext = polygon.exterior
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    lines = ext.intersection(local_patch)
    if lines.type != 'LineString':
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)
        if lines.type != 'LineString':
            ls = [np.array(l.coords) for l in lines.geoms]
            lines = LineString(np.concatenate(ls, axis=0))
    if not lines.is_empty:
        start = list(lines.coords[0])
        end = list(lines.coords[-1])
        if not np.allclose(start, end, atol=1e-3):
            lines = LineString(list(lines.coords) + [start])
        return lines
    return None


def _union_ped(ped_geoms, local_patch):
    """Merge close ped crossings (replicates nuscmap_extractor logic)."""
    ped_geoms = sorted(ped_geoms, key=lambda x: x.area, reverse=True)

    def get_rec_direction(geom):
        rect = geom.minimum_rotated_rectangle
        rect_v_p = np.array(rect.exterior.coords)[:3]
        rect_v = rect_v_p[1:] - rect_v_p[:-1]
        v_len = np.linalg.norm(rect_v, axis=-1)
        longest_v_i = v_len.argmax()
        return rect_v[longest_v_i], v_len[longest_v_i]

    tree = strtree.STRtree(ped_geoms)
    index_by_id = {id(pt): i for i, pt in enumerate(ped_geoms)}

    final_pgeom = []
    remain_idx = list(range(len(ped_geoms)))
    for i, pgeom in enumerate(ped_geoms):
        if i not in remain_idx:
            continue
        remain_idx.remove(i)
        pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
        final_pgeom.append(pgeom)

        intersect_pgeom = tree.query(pgeom)
        intersect_pgeom = sorted(intersect_pgeom, key=lambda x: x.area, reverse=True)
        for o in intersect_pgeom:
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue
            o_v, o_v_norm = get_rec_direction(o)
            cos = pgeom_v.dot(o_v) / (pgeom_v_norm * o_v_norm)
            o_pgeom_union = o.union(pgeom)
            ch_union = o_pgeom_union.convex_hull
            ch_area_ratio = o_pgeom_union.area / ch_union.area
            if 1 - np.abs(cos) < 0.01 and ch_area_ratio > 0.8:
                final_pgeom[-1] = final_pgeom[-1].union(o)
                remain_idx.remove(o_idx)

    # Second pass: handle small peds
    def get_two_rec_directions(geom):
        rect = geom.minimum_rotated_rectangle
        rect_v_p = np.array(rect.exterior.coords)[:3]
        rect_v = rect_v_p[1:] - rect_v_p[:-1]
        v_len = np.linalg.norm(rect_v, axis=-1)
        return rect_v, v_len

    tree2 = strtree.STRtree(final_pgeom)
    index_by_id2 = {id(pt): i for i, pt in enumerate(final_pgeom)}
    final2 = []
    remain_idx2 = list(range(len(final_pgeom)))
    for i, pgeom in enumerate(final_pgeom):
        if i not in remain_idx2:
            continue
        remain_idx2.remove(i)
        final2.append(pgeom)
        pgeom_v, pgeom_v_norm = get_two_rec_directions(pgeom)
        intersect_pgeom = tree2.query(pgeom)
        intersect_pgeom = sorted(intersect_pgeom, key=lambda x: x.area, reverse=True)
        for o in intersect_pgeom:
            o_idx = index_by_id2[id(o)]
            if o_idx not in remain_idx2:
                continue
            if o.area >= pgeom.area:
                continue
            o_pgeom_union = o.union(pgeom)
            o_v, o_v_norm = get_two_rec_directions(o_pgeom_union)
            ch_union = o_pgeom_union.convex_hull
            ch_area_ratio = o_pgeom_union.area / ch_union.area
            cos_00 = pgeom_v[0].dot(o_v[0]) / (pgeom_v_norm[0] * o_v_norm[0])
            cos_01 = pgeom_v[0].dot(o_v[1]) / (pgeom_v_norm[0] * o_v_norm[1])
            cos_10 = pgeom_v[1].dot(o_v[0]) / (pgeom_v_norm[1] * o_v_norm[0])
            cos_11 = pgeom_v[1].dot(o_v[1]) / (pgeom_v_norm[1] * o_v_norm[1])
            cos_checks = np.array([(1 - np.abs(c) < 0.001)
                                   for c in [cos_00, cos_01, cos_10, cos_11]])
            if cos_checks.sum() == 2 and ch_area_ratio > 0.8:
                final2[-1] = final2[-1].union(o)
                remain_idx2.remove(o_idx)

    # Filter small polygons
    updated = []
    for p_idx, p in enumerate(final2):
        area = p.area
        if area < 1:
            continue
        elif area < 20:
            covered = False
            for other_idx, p_other in enumerate(final2):
                if other_idx != p_idx and p.covered_by(p_other):
                    covered = True
                    break
            if not covered:
                updated.append(p)
        else:
            updated.append(p)

    results = []
    for p in updated:
        results.extend(_split_collections(p))
    return results


def extract_gt_for_frame(nusc_map, map_explorer, location,
                         lidar2global_translation, lidar2global_rotation, roi_size):
    """
    Extract GT polylines for one frame.

    Returns:
        gt_by_label: {label_id: [ndarray, ...]}
            mAP GT in MapTracker convention (with -90° rotation).
        gt_instances: [(instance_id, label_id, ndarray), ...]
            mAS GT in standard ego frame (NO -90° rotation).
            instance_id uses nuScenes record tokens for cross-frame stability.
    """
    map_pose = lidar2global_translation[:2]
    rotation = Quaternion(lidar2global_rotation)
    patch_box = (map_pose[0], map_pose[1], roi_size[0], roi_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180
    patch_x, patch_y = patch_box[0], patch_box[1]
    patch_geom = map_explorer.get_patch_coord(patch_box, patch_angle)

    gt_by_label = {label: [] for label in ID2CAT}
    gt_instances = []

    def to_local(geom):
        geom = affinity.rotate(geom, -patch_angle,
                               origin=(patch_x, patch_y), use_radians=False)
        return affinity.affine_transform(geom, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])

    rotated_local_patch = box(-roi_size[1] / 2, -roi_size[0] / 2,
                               roi_size[1] / 2, roi_size[0] / 2)
    standard_local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2,
                                roi_size[0] / 2, roi_size[1] / 2)

    # --- Dividers (lane_divider + road_divider) ---
    for layer_name in ['lane_divider', 'road_divider']:
        for record in getattr(nusc_map, layer_name):
            line = nusc_map.extract_line(record['line_token'])
            if line.is_empty:
                continue
            new_line = line.intersection(patch_geom)
            if new_line.is_empty:
                continue
            new_line = to_local(new_line)

            # mAP GT: with -90° rotation (MapTracker convention)
            rotated = affinity.rotate(new_line, -90, origin=(0, 0), use_radians=False)
            for sl in _split_collections(rotated):
                if sl.geom_type == 'LineString':
                    c = np.array(sl.simplify(0.2, preserve_topology=True).coords)[:, :2]
                    gt_by_label[CAT2ID['divider']].append(c)

            # mAS GT: NO rotation, record token as stable instance ID
            for sub_idx, sl in enumerate(_split_collections(new_line)):
                if sl.geom_type == 'LineString':
                    c = np.array(sl.simplify(0.2, preserve_topology=True).coords)[:, :2]
                    gt_instances.append(
                        (f"{layer_name}_{record['token']}_{sub_idx}",
                         CAT2ID['divider'], c))

    # --- Boundary ---
    # mAP GT: union approach (merged road_segment + lane exteriors)
    polygon_list = []
    for layer_name in ['road_segment', 'lane']:
        for record in getattr(nusc_map, layer_name):
            polygon = nusc_map.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch_geom)
                if not new_polygon.is_empty:
                    new_polygon = to_local(new_polygon)
                    new_polygon = affinity.rotate(new_polygon, -90,
                                                  origin=(0, 0), use_radians=False)
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

    if polygon_list:
        union_roads_lanes = ops.unary_union(polygon_list)
        hx, hy = roi_size[0] / 2, roi_size[1] / 2
        local_patch_boundary = box(-hx + 0.2, -hy + 0.2, hx - 0.2, hy - 0.2)
        if union_roads_lanes.geom_type != 'MultiPolygon':
            union_roads_lanes = MultiPolygon([union_roads_lanes])

        for poly in union_roads_lanes.geoms:
            for ring_type, ring in ([('ext', poly.exterior)] +
                                    [('int', i) for i in poly.interiors]):
                if ring_type == 'ext' and ring.is_ccw:
                    ring = LinearRing(list(ring.coords)[::-1])
                elif ring_type == 'int' and not ring.is_ccw:
                    ring = LinearRing(list(ring.coords)[::-1])
                lines = ring.intersection(local_patch_boundary)
                if isinstance(lines, MultiLineString):
                    lines = ops.linemerge(lines)
                for l in _split_collections(lines):
                    if l.geom_type == 'LineString':
                        c = np.array(l.simplify(0.2, preserve_topology=True).coords)[:, :2]
                        gt_by_label[CAT2ID['boundary']].append(c)

    # mAS GT: individual polygon boundaries, NO rotation, record token IDs
    for layer_name in ['road_segment', 'lane']:
        for record in getattr(nusc_map, layer_name):
            polygon = nusc_map.extract_polygon(record['polygon_token'])
            if not polygon.is_valid or polygon.is_empty or not polygon.intersects(patch_geom):
                continue
            rings = [polygon.exterior] + list(polygon.interiors)
            for ring_idx, ring in enumerate(rings):
                ring_line = LineString(ring.coords)
                ring_local = to_local(ring_line)
                clipped = ring_local.intersection(standard_local_patch)
                if clipped.is_empty:
                    continue
                for seg_idx, seg in enumerate(_split_collections(clipped)):
                    if seg.geom_type == 'LineString' and seg.length > 1.0:
                        c = np.array(seg.simplify(0.2, preserve_topology=True).coords)[:, :2]
                        gt_instances.append((
                            f"boundary_{layer_name}_{record['token']}_{ring_idx}_{seg_idx}",
                            CAT2ID['boundary'], c))

    # --- Ped Crossing ---
    # mAP GT: merged ped crossings
    ped_layer = map_explorer._get_layer_polygon(patch_box, patch_angle, 'ped_crossing')
    ped_crossings = []
    for p in ped_layer:
        ped_crossings.extend(_split_collections(p))
    local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2,
                       roi_size[0] / 2, roi_size[1] / 2)
    ped_crossings = _union_ped(ped_crossings, local_patch)

    for p in ped_crossings:
        p_rot = affinity.rotate(p, -90, origin=(0, 0), use_radians=False)
        line = _get_ped_crossing_contour(p_rot, rotated_local_patch)
        if line is not None:
            c = np.array(line.simplify(0.2, preserve_topology=True).coords)[:, :2]
            gt_by_label[CAT2ID['ped_crossing']].append(c)

    # mAS GT: individual ped_crossing records, NO rotation
    for record in getattr(nusc_map, 'ped_crossing'):
        polygon = nusc_map.extract_polygon(record['polygon_token'])
        if not polygon.is_valid or polygon.is_empty or not polygon.intersects(patch_geom):
            continue
        ring_line = LineString(polygon.exterior.coords)
        ring_local = to_local(ring_line)
        clipped = ring_local.intersection(standard_local_patch)
        if clipped.is_empty:
            continue
        for seg_idx, seg in enumerate(_split_collections(clipped)):
            if seg.geom_type == 'LineString' and seg.length > 1.0:
                c = np.array(seg.simplify(0.2, preserve_topology=True).coords)[:, :2]
                gt_instances.append(
                    (f"ped_{record['token']}_{seg_idx}",
                     CAT2ID['ped_crossing'], c))

    return gt_by_label, gt_instances


def load_all_gt(ann_file, dataroot, roi_size):
    """Load GT for all frames with caching."""
    cache_file = f'./tmp_gts_eval_final_{roi_size[0]}x{roi_size[1]}.pkl'
    if os.path.exists(cache_file):
        print(f'Loading cached GT from {cache_file}')
        cached = mmcv.load(cache_file)
        return cached['gts'], cached['gt_instances_all'], cached['frame_infos']

    print('Loading annotations...')
    ann = mmcv.load(ann_file)
    samples = ann['samples'] if isinstance(ann, dict) and 'samples' in ann else ann

    nusc_maps = {}
    map_explorers = {}
    for loc in MAPS:
        nusc_maps[loc] = NuScenesMap(dataroot=dataroot, map_name=loc)
        map_explorers[loc] = NuScenesMapExplorer(nusc_maps[loc])

    gts, gt_instances_all, frame_infos = {}, {}, {}

    print(f'Extracting GT for {len(samples)} frames...')
    pbar = mmcv.ProgressBar(len(samples))
    for sample in samples:
        token = sample['token']
        location = sample['location']

        # Compute lidar2global transform
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']
        lidar2global = ego2global @ lidar2ego

        l2g_trans = list(lidar2global[:3, 3])
        l2g_rot = list(Quaternion(matrix=lidar2global).q)

        gt_by_label, gt_inst = extract_gt_for_frame(
            nusc_maps[location], map_explorers[location], location,
            l2g_trans, l2g_rot, roi_size)

        gts[token] = gt_by_label
        gt_instances_all[token] = gt_inst
        frame_infos[token] = {
            'scene_name': sample['scene_name'],
            'sample_idx': sample['sample_idx'],
            'lidar2global_4x4': lidar2global.copy(),
            'token': token,
        }
        pbar.update()

    print(f'\nSaving GT cache to {cache_file}')
    mmcv.dump({'gts': gts, 'gt_instances_all': gt_instances_all,
               'frame_infos': frame_infos}, cache_file)
    return gts, gt_instances_all, frame_infos


# ============================================================
# Section 4: mAP computation
# ============================================================

def _evaluate_single(pred_vectors, scores, groundtruth, thresholds):
    """Single-frame per-class matching for mAP."""
    pred_lines = []
    for v in pred_vectors:
        v = np.array(v)
        pred_lines.append(interp_fixed_num(v, INTERP_NUM) if len(v) >= 2
                          else np.zeros((INTERP_NUM, 2)))
    pred_lines = np.stack(pred_lines) if pred_lines else np.zeros((0, INTERP_NUM, 2))

    gt_lines = []
    for v in groundtruth:
        gt_lines.append(interp_fixed_num(v, INTERP_NUM) if len(v) >= 2
                        else np.zeros((INTERP_NUM, 2)))
    gt_lines = np.stack(gt_lines) if gt_lines else np.zeros((0, INTERP_NUM, 2))

    scores = np.array(scores)
    tp_fp_list = instance_match(pred_lines, scores, gt_lines, thresholds)
    result = {}
    for thr, (tp, fp) in zip(thresholds, tp_fp_list):
        if len(scores) > 0:
            result[thr] = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
        else:
            result[thr] = np.zeros((0, 3), dtype=np.float32)
    return result


def compute_mAP(results, gts, thresholds, n_workers=N_WORKERS):
    """Compute mean Average Precision across all classes and thresholds."""
    samples_by_cls = {label: [] for label in ID2CAT}
    num_gts = {label: 0 for label in ID2CAT}
    num_preds = {label: 0 for label in ID2CAT}

    for token, gt in gts.items():
        pred = results.get(token, {'vectors': [], 'scores': [], 'labels': []})
        vecs = {l: [] for l in ID2CAT}
        scrs = {l: [] for l in ID2CAT}
        for i in range(len(pred.get('labels', []))):
            l = pred['labels'][i]
            if l in vecs:
                vecs[l].append(pred['vectors'][i])
                scrs[l].append(pred['scores'][i])
        for l in ID2CAT:
            samples_by_cls[l].append((vecs[l], scrs[l], gt[l]))
            num_gts[l] += len(gt[l])
            num_preds[l] += len(scrs[l])

    result_dict = {}
    print(f'\nComputing mAP for {len(ID2CAT)} categories...')
    start = time()

    pool = Pool(n_workers) if n_workers > 0 else None
    sum_mAP = 0
    for label in ID2CAT:
        fn = partial(_evaluate_single, thresholds=thresholds)
        tpfp_list = (pool.starmap(fn, samples_by_cls[label]) if pool
                     else [fn(*s) for s in samples_by_cls[label]])

        result_dict[ID2CAT[label]] = {
            'num_gts': num_gts[label], 'num_preds': num_preds[label],
        }
        sum_AP = 0
        for thr in thresholds:
            tp_fp_score = np.vstack([i[thr] for i in tpfp_list])
            if tp_fp_score.shape[0] == 0:
                result_dict[ID2CAT[label]][f'AP@{thr}'] = 0.0
                continue
            si = np.argsort(-tp_fp_score[:, -1])
            tp = np.cumsum(tp_fp_score[si, 0])
            fp = np.cumsum(tp_fp_score[si, 1])
            eps = np.finfo(np.float32).eps
            rec = tp / max(num_gts[label], eps)
            prec = tp / np.maximum(tp + fp, eps)
            AP = average_precision(rec, prec)
            sum_AP += AP
            result_dict[ID2CAT[label]][f'AP@{thr}'] = AP

        result_dict[ID2CAT[label]]['AP'] = sum_AP / len(thresholds)
        sum_mAP += result_dict[ID2CAT[label]]['AP']

    if pool:
        pool.close()
        pool.join()

    result_dict['mAP'] = sum_mAP / len(ID2CAT)
    print(f'mAP done in {time() - start:.2f}s')
    return result_dict


# ============================================================
# Section 5: mAS computation
# ============================================================

def transform_points_between_frames(points, src_pose_4x4, dst_pose_4x4):
    """Transform 2D points from src lidar frame to dst lidar frame via global."""
    pts = np.array(points, dtype=np.float64)
    if len(pts) == 0:
        return pts.copy()
    n = pts.shape[0]
    homo = np.ones((n, 4), dtype=np.float64)
    homo[:, :2] = pts[:, :2]
    homo[:, 2] = 0.0
    T = np.linalg.inv(dst_pose_4x4) @ src_pose_4x4
    return (T @ homo.T).T[:, :2]


def undo_maptracker_rotation(pts):
    """Undo MapTracker's -90° rotation: (x,y) → (-y, x)."""
    pts = np.array(pts, dtype=np.float64)
    return np.column_stack([-pts[:, 1], pts[:, 0]])


def filter_to_range(poly, x_range, y_range):
    """Keep only points within perception range."""
    poly = np.array(poly, dtype=np.float64)
    if len(poly) == 0:
        return poly
    mask = ((poly[:, 0] >= x_range[0]) & (poly[:, 0] <= x_range[1]) &
            (poly[:, 1] >= y_range[0]) & (poly[:, 1] <= y_range[1]))
    return poly[mask]


def compute_presence(cur_score, prev_score, threshold=DET_THRESHOLD):
    """
    Presence stability (paper definition).
    1.0 if both detect or both miss, 0.5 if flickering.
    """
    cur_det = cur_score >= threshold
    prev_det = prev_score >= threshold
    return 1.0 if cur_det == prev_det else 0.5


def compute_localization(cur_poly, prev_poly, beta=BETA):
    """
    Localization stability with dynamic axis resampling + linear normalization.
    Loc(e) = clamp(1 - avg_L1 / β, 0, 1)   where β=15.0 (y_range/2).
    """
    avg_L1 = dynamic_axis_resample_deviation(cur_poly, prev_poly, N=N_STABILITY_SAMPLES)
    if avg_L1 < 0:
        return -1.0
    return float(np.clip(1.0 - avg_L1 / beta, 0.0, 1.0))


def compute_shape(cur_poly, prev_poly):
    """
    Shape stability via curvature difference.
    Shape(e) = 1 - |κ_cur - κ_prev| / π, clamped to [0, 1].
    """
    def curvature(poly):
        if len(poly) < 3:
            return 0.0
        angles = []
        for i in range(1, len(poly) - 1):
            v1 = poly[i] - poly[i - 1]
            v2 = poly[i + 1] - poly[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angles.append(np.arccos(cos_a))
        return float(np.mean(angles)) if angles else 0.0

    cur_poly = np.array(cur_poly, dtype=np.float64)
    prev_poly = np.array(prev_poly, dtype=np.float64)
    if len(cur_poly) < 2 or len(prev_poly) < 2:
        return -1.0
    if len(cur_poly) < 3 or len(prev_poly) < 3:
        return 1.0
    return max(0.0, 1.0 - abs(curvature(cur_poly) - curvature(prev_poly)) / np.pi)


def hungarian_match_preds_to_gt(pred_vectors, pred_scores, gt_polylines,
                                gt_instance_ids, num_sample=MAS_RESAMPLE_NUM):
    """
    Per-class Hungarian matching of predictions to GT for mAS.
    Tries both forward/reverse polyline directions for minimum Chamfer cost.

    Returns:
        dict: {instance_id: (matched_pred_polyline, matched_score)}
              Unmatched GT → (empty_array, 0.0)
    """
    n_gt = len(gt_polylines)
    n_pred = len(pred_vectors)

    if n_gt == 0:
        return {}
    if n_pred == 0:
        return {iid: (np.zeros((0, 2)), 0.0) for iid in gt_instance_ids}

    # Resample to fixed points for distance computation
    def safe_resample(v):
        v = np.array(v)
        if len(v) >= 2:
            return interp_fixed_num(v, num_sample)
        return np.zeros((num_sample, 2))

    pred_interp = np.stack([safe_resample(v) for v in pred_vectors])
    gt_interp = np.stack([safe_resample(v) for v in gt_polylines])

    # Chamfer distance: try both forward and reverse, take minimum
    cost_fwd = chamfer_distance_batch(pred_interp, gt_interp)
    cost_rev = chamfer_distance_batch(pred_interp[:, ::-1].copy(), gt_interp)
    cost = np.minimum(cost_fwd, cost_rev)

    # Optimal 1-to-1 assignment
    pred_idx, gt_idx = linear_sum_assignment(cost)

    result = {}
    matched_gts = set()
    for pi, gi in zip(pred_idx, gt_idx):
        iid = gt_instance_ids[gi]
        result[iid] = (np.array(pred_vectors[pi]), float(pred_scores[pi]))
        matched_gts.add(gi)

    # Unmatched GT instances → score=0 (presence will mark as miss)
    for gi, iid in enumerate(gt_instance_ids):
        if gi not in matched_gts and iid not in result:
            result[iid] = (np.zeros((0, 2)), 0.0)

    return result


def build_frame_pairs(frame_infos, interval=INTERVAL):
    """Build (cur_token, prev_token) pairs within each scene."""
    scenes = defaultdict(list)
    for token, info in frame_infos.items():
        scenes[info['scene_name']].append(info)

    pairs = []
    for frames in scenes.values():
        frames = sorted(frames, key=lambda x: x['sample_idx'])
        for i in range(interval, len(frames)):
            pairs.append((frames[i]['token'], frames[i - interval]['token']))
    return pairs


def _process_single_pair(args):
    """
    Process a single frame pair for mAS.

    Pipeline (paper Section 3):
    1. Find common GT instances across frames (by instance_id)
    2. Hungarian match predictions to GT independently per frame (per class)
    3. For each common GT: transform prev pred to cur frame, compute stability
    """
    (cur_gt_inst, prev_gt_inst, cur_pred, prev_pred,
     cur_pose, prev_pose, roi_size) = args

    x_range = (-roi_size[0] / 2, roi_size[0] / 2)
    y_range = (-roi_size[1] / 2, roi_size[1] / 2)

    if not cur_gt_inst or not prev_gt_inst:
        return []

    # Undo MapTracker's -90° rotation on predictions → standard ego frame
    cur_vecs, cur_scores_list, cur_labels = [], [], []
    for i in range(len(cur_pred.get('labels', []))):
        cur_vecs.append(undo_maptracker_rotation(cur_pred['vectors'][i]))
        cur_scores_list.append(cur_pred['scores'][i])
        cur_labels.append(cur_pred['labels'][i])

    prev_vecs, prev_scores_list, prev_labels = [], [], []
    for i in range(len(prev_pred.get('labels', []))):
        prev_vecs.append(undo_maptracker_rotation(prev_pred['vectors'][i]))
        prev_scores_list.append(prev_pred['scores'][i])
        prev_labels.append(prev_pred['labels'][i])

    # Build GT lookups
    cur_inst_dict = {iid: (lab, coords) for iid, lab, coords in cur_gt_inst}
    prev_inst_dict = {iid: (lab, coords) for iid, lab, coords in prev_gt_inst}
    common_ids = set(cur_inst_dict.keys()) & set(prev_inst_dict.keys())
    if not common_ids:
        return []

    pair_results = []

    for label_id in ID2CAT:
        # Filter GT and predictions by class
        label_common = [iid for iid in common_ids if cur_inst_dict[iid][0] == label_id]
        if not label_common:
            continue

        # Current frame: all GT of this class (not just common — for Hungarian matching)
        cur_all_gt_ids = [iid for iid, lab, _ in cur_gt_inst if lab == label_id]
        cur_all_gt_polys = [coords for _, lab, coords in cur_gt_inst if lab == label_id]
        prev_all_gt_ids = [iid for iid, lab, _ in prev_gt_inst if lab == label_id]
        prev_all_gt_polys = [coords for _, lab, coords in prev_gt_inst if lab == label_id]

        cur_cls_vecs = [v for v, l in zip(cur_vecs, cur_labels) if l == label_id]
        cur_cls_scrs = [s for s, l in zip(cur_scores_list, cur_labels) if l == label_id]
        prev_cls_vecs = [v for v, l in zip(prev_vecs, prev_labels) if l == label_id]
        prev_cls_scrs = [s for s, l in zip(prev_scores_list, prev_labels) if l == label_id]

        # Hungarian match pred→GT per frame (using ALL GT, not just common)
        cur_match = hungarian_match_preds_to_gt(
            cur_cls_vecs, cur_cls_scrs, cur_all_gt_polys, cur_all_gt_ids)
        prev_match = hungarian_match_preds_to_gt(
            prev_cls_vecs, prev_cls_scrs, prev_all_gt_polys, prev_all_gt_ids)

        # Evaluate stability for common GT instances
        cat_name = ID2CAT[label_id]
        for inst_id in label_common:
            if inst_id not in cur_match or inst_id not in prev_match:
                continue

            cur_poly, cur_score = cur_match[inst_id]
            prev_poly_raw, prev_score = prev_match[inst_id]

            # Both-miss filter: skip if both frames score < 0.01
            if cur_score < BOTH_MISS_THRESHOLD and prev_score < BOTH_MISS_THRESHOLD:
                continue

            # Presence
            pres = compute_presence(cur_score, prev_score)

            # If one side has no polyline → geometric metrics undefined → SI=0
            if len(cur_poly) < 2 or len(prev_poly_raw) < 2:
                pair_results.append((cat_name, pres, 0.0, 0.0, 0.0))
                continue

            # Transform prev pred to cur ego frame
            prev_poly_aligned = transform_points_between_frames(
                prev_poly_raw, prev_pose, cur_pose)

            # Perception range filtering
            cur_poly_f = filter_to_range(cur_poly, x_range, y_range)
            prev_poly_f = filter_to_range(prev_poly_aligned, x_range, y_range)

            if len(cur_poly_f) < 2 or len(prev_poly_f) < 2:
                pair_results.append((cat_name, pres, 0.0, 0.0, 0.0))
                continue

            # Localization (dynamic axis resampling + exponential)
            loc = compute_localization(cur_poly_f, prev_poly_f)
            if loc < 0:
                loc = 0.0

            # Shape (curvature difference)
            shp = compute_shape(cur_poly_f, prev_poly_f)
            if shp < 0:
                shp = 0.0

            # Stability Index
            base = np.clip(LOC_WEIGHT * loc + (1 - LOC_WEIGHT) * shp, 0.0, 1.0)
            si = pres * base
            pair_results.append((cat_name, pres, loc, shp, si))

    return pair_results


def compute_mAS(results, gt_instances_all, frame_infos, roi_size=ROI_SIZE,
                interval=INTERVAL, n_workers=N_WORKERS):
    """
    Compute mean Average Stability (mAS).

    For each frame pair within a scene:
    1. Find common GT instances (by nuScenes record token)
    2. Hungarian match preds to GT per frame (Chamfer cost, fwd+rev)
    3. Transform prev pred → cur frame coordinate
    4. Dynamic axis resampling for localization
    5. Combine presence × (ω·loc + (1-ω)·shape)
    """
    print(f'\nComputing mAS (interval={interval}, ω={LOC_WEIGHT}, β={BETA})...')
    start = time()

    pairs = build_frame_pairs(frame_infos, interval)
    print(f'  {len(pairs)} frame pairs')

    worker_args = []
    for cur_token, prev_token in pairs:
        cur_gt_inst = gt_instances_all.get(cur_token, [])
        prev_gt_inst = gt_instances_all.get(prev_token, [])
        if not cur_gt_inst or not prev_gt_inst:
            continue
        cur_pred = results.get(cur_token, {'vectors': [], 'scores': [], 'labels': []})
        prev_pred = results.get(prev_token, {'vectors': [], 'scores': [], 'labels': []})
        cur_pose = frame_infos[cur_token]['lidar2global_4x4']
        prev_pose = frame_infos[prev_token]['lidar2global_4x4']
        worker_args.append((
            cur_gt_inst, prev_gt_inst, cur_pred, prev_pred,
            cur_pose, prev_pose, roi_size,
        ))

    print(f'  {len(worker_args)} valid pairs to process')

    if n_workers > 0 and len(worker_args) > 0:
        pool = Pool(n_workers)
        all_pair_results = pool.map(_process_single_pair, worker_args)
        pool.close()
        pool.join()
    else:
        all_pair_results = [_process_single_pair(a) for a in worker_args]

    # Aggregate per-class
    class_metrics = defaultdict(
        lambda: {'presence': [], 'localization': [], 'shape': [], 'si': []})
    for pair_results in all_pair_results:
        for cat_name, pc, ls, ss, si in pair_results:
            class_metrics[cat_name]['presence'].append(pc)
            class_metrics[cat_name]['localization'].append(ls)
            class_metrics[cat_name]['shape'].append(ss)
            class_metrics[cat_name]['si'].append(si)

    mas_result = {}
    for cat_name in ['divider', 'ped_crossing', 'boundary']:
        m = class_metrics[cat_name]
        if m['si']:
            mas_result[cat_name] = {
                'PC': float(np.mean(m['presence'])),
                'LS': float(np.mean(m['localization'])),
                'SS': float(np.mean(m['shape'])),
                'AS': float(np.mean(m['si'])),
                'count': len(m['si']),
            }
        else:
            mas_result[cat_name] = {
                'PC': 0.0, 'LS': 0.0, 'SS': 0.0, 'AS': 0.0, 'count': 0,
            }

    # Mean across classes
    for k in ['PC', 'LS', 'SS', 'AS']:
        vals = [mas_result[c][k] for c in ['divider', 'ped_crossing', 'boundary']]
        mas_result[f'm{k}'] = float(np.mean(vals))

    print(f'mAS done in {time() - start:.2f}s')
    return mas_result


# ============================================================
# Section 6: Output
# ============================================================

def print_results(map_result, mas_result, thresholds):
    """Print mAP and mAS results in formatted tables."""
    # mAP table
    print('\n' + '=' * 70)
    print('  mAP Results')
    print('=' * 70)
    table = prettytable.PrettyTable(
        ['category', 'num_preds', 'num_gts'] +
        [f'AP@{t}' for t in thresholds] + ['AP'])
    for label_id in sorted(ID2CAT.keys()):
        cat = ID2CAT[label_id]
        r = map_result.get(cat, {})
        table.add_row([
            cat,
            r.get('num_preds', 0),
            r.get('num_gts', 0),
            *[round(r.get(f'AP@{t}', 0), 4) for t in thresholds],
            round(r.get('AP', 0), 4),
        ])
    print(table)
    print(f"mAP = {map_result.get('mAP', 0):.4f}")

    # mAS table
    print('\n' + '=' * 70)
    print(f'  mAS Results  (ω={LOC_WEIGHT}, β={BETA}, interval={INTERVAL})')
    print('=' * 70)
    table2 = prettytable.PrettyTable(
        ['category', 'count', 'Presence', 'Localization', 'Shape', 'SI'])
    for cat in ['divider', 'ped_crossing', 'boundary']:
        r = mas_result.get(cat, {})
        table2.add_row([
            cat,
            r.get('count', 0),
            round(r.get('PC', 0), 4),
            round(r.get('LS', 0), 4),
            round(r.get('SS', 0), 4),
            round(r.get('AS', 0), 4),
        ])
    print(table2)
    print(f"mAS  = {mas_result.get('mAS', 0):.4f}  |  "
          f"mPC = {mas_result.get('mPC', 0):.4f}  |  "
          f"mLS = {mas_result.get('mLS', 0):.4f}  |  "
          f"mSS = {mas_result.get('mSS', 0):.4f}")
    print('=' * 70)


# ============================================================
# Section 7: Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Unified mAP + mAS evaluator')
    parser.add_argument('--submission', required=True,
                        help='Path to submission_vector.json')
    parser.add_argument('--dataroot', required=True,
                        help='NuScenes dataset root')
    parser.add_argument('--ann-file', required=True,
                        help='Annotation pkl file')
    parser.add_argument('--roi-size', nargs=2, type=int, default=list(ROI_SIZE),
                        help='ROI size (default: 60 30)')
    parser.add_argument('--interval', type=int, default=INTERVAL,
                        help='mAS frame interval (default: 2)')
    parser.add_argument('--n-workers', type=int, default=N_WORKERS,
                        help='Parallel workers (default: 16)')
    args = parser.parse_args()

    roi_size = tuple(args.roi_size)
    thresholds = [1.0, 1.5, 2.0] if roi_size == (100, 50) else THRESHOLDS

    # Load submission
    print(f'Loading submission from {args.submission}')
    submission = mmcv.load(args.submission)
    results = submission['results']

    # Load GT
    gts, gt_instances_all, frame_infos = load_all_gt(
        args.ann_file, args.dataroot, roi_size)

    # Compute mAP
    map_result = compute_mAP(results, gts, thresholds, n_workers=args.n_workers)

    # Compute mAS
    mas_result = compute_mAS(results, gt_instances_all, frame_infos,
                             roi_size=roi_size, interval=args.interval,
                             n_workers=args.n_workers)

    # Print results
    print_results(map_result, mas_result, thresholds)

    return map_result, mas_result


if __name__ == '__main__':
    main()
