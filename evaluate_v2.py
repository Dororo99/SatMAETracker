#!/usr/bin/env python
"""
evaluate_v2.py — Unified mAP + mAS evaluator (MapStableTest-aligned methodology).

GT extraction and cross-frame transform use ego poses (matching MapStableTest).
Pred→GT matching uses MapTR assigner (imported from MapStableTest).

Parameters (paper-based, user-chosen):
  - Presence:      1.0 / 0.5                    [paper + metrics_fixed.py]
  - Localization:  1 - avg_L1/β, dynamic axis   [paper + OpenReview]
  - Shape:         1 - |κ_cur - κ_prev| / π     [paper = code]
  - ω (loc weight): 0.7                          [paper]
  - β:             15.0                           [OpenReview, = y_range/2]
  - Detection thr:  0.3                           [paper = code]

Usage:
    python evaluate_v2.py \\
        --submission path/to/submission_vector.json \\
        --dataroot ./datasets/nuscenes \\
        --ann-file ./datasets/nuscenes/nuscenes_map_infos_val_newsplit.pkl
"""

import argparse
import os
import sys
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

# Import MapStableTest modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MapStableTest', 'src'))
from maptr_stability_eval.geometry import transform_points_between_frames as mst_transform
from maptr_stability_eval.stability.alignment import align_det_and_gt_by_maptr_assigner

# ============================================================
# Configuration
# ============================================================
ROI_SIZE = (60, 30)            # BEV range: 60m (x-forward) × 30m (y-lateral)
THRESHOLDS = [0.5, 1.0, 1.5]  # mAP Chamfer distance thresholds (meters)
INTERP_NUM = 200               # arc-length resampling points for mAP
N_WORKERS = 16
INTERVAL = 2
LOC_WEIGHT = 0.7              # ω: localization weight in SI (paper)
BETA = 15.0                   # β: localization normalizer = y_range/2
DET_THRESHOLD = 0.3
BOTH_MISS_THRESHOLD = 0.01
N_STABILITY_SAMPLES = 100
# pc_range for MapTR assigner normalization (ego frame: x=fwd, y=lat)
PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

CAT2ID = {'ped_crossing': 0, 'divider': 1, 'boundary': 2}
ID2CAT = {v: k for k, v in CAT2ID.items()}
ID2NAME = {0: 'ped_crossing', 1: 'divider', 2: 'boundary'}
NAME2ID = {v: k for k, v in ID2NAME.items()}
MAPS = ['boston-seaport', 'singapore-hollandvillage',
        'singapore-onenorth', 'singapore-queenstown']


# ============================================================
# Section 1: mAP utilities (unchanged from evaluate_final.py)
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
    """Greedy matching of predictions to GT per threshold."""
    num_preds = pred_lines.shape[0]
    num_gts = gt_lines.shape[0]
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
    tp_fp_list = []
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
    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]
    mrec = np.hstack((np.zeros((1, 1)), recalls, np.ones((1, 1))))
    mpre = np.hstack((np.zeros((1, 1)), precisions, np.zeros((1, 1))))
    for i in range(mpre.shape[1] - 1, 0, -1):
        mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
    ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]
    return np.sum((mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1])


def interp_fixed_num(vector, num_pts):
    """Arc-length uniform resampling."""
    line = LineString(vector)
    distances = np.linspace(0, line.length, num_pts)
    return np.array([list(line.interpolate(d).coords) for d in distances]).squeeze()


# ============================================================
# Section 2: Dynamic Axis Resampling
# ============================================================

def interpolate_along_axis(poly, samples, axis=0):
    """Piecewise linear interpolation along specified axis."""
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
    """Dynamic axis resampling deviation (OpenReview algorithm)."""
    cur_poly = np.array(cur_poly, dtype=np.float64)
    prev_poly = np.array(prev_poly, dtype=np.float64)
    if len(cur_poly) < 2 or len(prev_poly) < 2:
        return -1.0

    n_segs = len(cur_poly) - 1
    seg_is_y = []
    seg_lengths = []
    for j in range(n_segs):
        dx = abs(cur_poly[j + 1, 0] - cur_poly[j, 0])
        dy = abs(cur_poly[j + 1, 1] - cur_poly[j, 1])
        is_y = dy > dx
        seg_is_y.append(is_y)
        seg_lengths.append(dy if is_y else dx)

    intervals = []
    start_idx = 0
    for j in range(1, n_segs):
        if seg_is_y[j] != seg_is_y[start_idx]:
            intervals.append({
                'is_y': seg_is_y[start_idx], 'start': start_idx,
                'end': j, 'length': sum(seg_lengths[start_idx:j]),
            })
            start_idx = j
    intervals.append({
        'is_y': seg_is_y[start_idx], 'start': start_idx,
        'end': n_segs, 'length': sum(seg_lengths[start_idx:n_segs]),
    })

    total_length = sum(iv['length'] for iv in intervals)
    if total_length < 1e-9:
        return -1.0

    K = len(intervals)
    n_points = [max(2, round(N * iv['length'] / total_length)) for iv in intervals]
    while sum(n_points) < N:
        order = sorted(range(K), key=lambda i: intervals[i]['length'], reverse=True)
        for i in order:
            if sum(n_points) >= N:
                break
            n_points[i] += 1
    while sum(n_points) > N:
        order = sorted(range(K), key=lambda i: intervals[i]['length'])
        for i in order:
            if sum(n_points) <= N:
                break
            if n_points[i] > 2:
                n_points[i] -= 1

    all_deviations = []
    for k, iv in enumerate(intervals):
        interval_pts = cur_poly[iv['start']:iv['end'] + 1]
        axis = 1 if iv['is_y'] else 0
        primary_min = interval_pts[:, axis].min()
        primary_max = interval_pts[:, axis].max()
        if primary_max - primary_min < 1e-6:
            continue
        samples = np.linspace(primary_min, primary_max, n_points[k])
        vals_cur = interpolate_along_axis(cur_poly, samples, axis=axis)
        vals_prev = interpolate_along_axis(prev_poly, samples, axis=axis)
        all_deviations.extend(np.abs(vals_cur - vals_prev))

    if len(all_deviations) == 0:
        return -1.0
    return float(np.mean(all_deviations))


# ============================================================
# Section 3: GT extraction (mAP: lidar-based, mAS: ego-based)
# ============================================================

def _split_collections(geom):
    if 'Multi' in geom.geom_type:
        return [g for g in geom.geoms if g.is_valid and not g.is_empty]
    if geom.is_valid and not geom.is_empty:
        return [geom]
    return []


def _get_ped_crossing_contour(polygon, local_patch):
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
            if o_idx not in remain_idx2 or o.area >= pgeom.area:
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
    updated = []
    for p_idx, p in enumerate(final2):
        area = p.area
        if area < 1:
            continue
        elif area < 20:
            covered = any(p.covered_by(q) for j, q in enumerate(final2) if j != p_idx)
            if not covered:
                updated.append(p)
        else:
            updated.append(p)
    results = []
    for p in updated:
        results.extend(_split_collections(p))
    return results


def extract_map_gt_lidar_frame(nusc_map, map_explorer, location,
                               lidar2global_translation, lidar2global_rotation, roi_size):
    """Extract mAP GT in lidar frame with -90° rotation (MapTracker convention)."""
    map_pose = lidar2global_translation[:2]
    rotation = Quaternion(lidar2global_rotation)
    patch_box = (map_pose[0], map_pose[1], roi_size[0], roi_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180
    patch_x, patch_y = patch_box[0], patch_box[1]
    patch_geom = map_explorer.get_patch_coord(patch_box, patch_angle)

    gt_by_label = {label: [] for label in ID2CAT}

    def to_local(geom):
        geom = affinity.rotate(geom, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
        return affinity.affine_transform(geom, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])

    rotated_local_patch = box(-roi_size[1]/2, -roi_size[0]/2, roi_size[1]/2, roi_size[0]/2)

    # Dividers
    for layer_name in ['lane_divider', 'road_divider']:
        for record in getattr(nusc_map, layer_name):
            line = nusc_map.extract_line(record['line_token'])
            if line.is_empty:
                continue
            new_line = line.intersection(patch_geom)
            if new_line.is_empty:
                continue
            new_line = to_local(new_line)
            rotated = affinity.rotate(new_line, -90, origin=(0, 0), use_radians=False)
            for sl in _split_collections(rotated):
                if sl.geom_type == 'LineString':
                    c = np.array(sl.simplify(0.2, preserve_topology=True).coords)[:, :2]
                    gt_by_label[CAT2ID['divider']].append(c)

    # Boundary (union approach)
    polygon_list = []
    for layer_name in ['road_segment', 'lane']:
        for record in getattr(nusc_map, layer_name):
            polygon = nusc_map.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch_geom)
                if not new_polygon.is_empty:
                    new_polygon = to_local(new_polygon)
                    new_polygon = affinity.rotate(new_polygon, -90, origin=(0, 0), use_radians=False)
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)
    if polygon_list:
        union = ops.unary_union(polygon_list)
        hx, hy = roi_size[0]/2, roi_size[1]/2
        boundary_clip = box(-hx + 0.2, -hy + 0.2, hx - 0.2, hy - 0.2)
        if union.geom_type != 'MultiPolygon':
            union = MultiPolygon([union])
        for poly in union.geoms:
            for ring_type, ring in ([('ext', poly.exterior)] +
                                    [('int', i) for i in poly.interiors]):
                if ring_type == 'ext' and ring.is_ccw:
                    ring = LinearRing(list(ring.coords)[::-1])
                elif ring_type == 'int' and not ring.is_ccw:
                    ring = LinearRing(list(ring.coords)[::-1])
                lines = ring.intersection(boundary_clip)
                if isinstance(lines, MultiLineString):
                    lines = ops.linemerge(lines)
                for l in _split_collections(lines):
                    if l.geom_type == 'LineString':
                        c = np.array(l.simplify(0.2, preserve_topology=True).coords)[:, :2]
                        gt_by_label[CAT2ID['boundary']].append(c)

    # Ped Crossing
    local_patch = box(-roi_size[0]/2, -roi_size[1]/2, roi_size[0]/2, roi_size[1]/2)
    ped_layer = map_explorer._get_layer_polygon(patch_box, patch_angle, 'ped_crossing')
    ped_crossings = []
    for p in ped_layer:
        ped_crossings.extend(_split_collections(p))
    ped_crossings = _union_ped(ped_crossings, local_patch)
    for p in ped_crossings:
        p_rot = affinity.rotate(p, -90, origin=(0, 0), use_radians=False)
        line = _get_ped_crossing_contour(p_rot, rotated_local_patch)
        if line is not None:
            c = np.array(line.simplify(0.2, preserve_topology=True).coords)[:, :2]
            gt_by_label[CAT2ID['ped_crossing']].append(c)

    return gt_by_label


def extract_mas_gt_ego_frame(nusc_map, ego_translation, ego_rotation, roi_size):
    """
    Extract mAS GT in ego frame (MapStableTest methodology).
    Uses ego_translation + ego_rotation for coordinate transform (pure 2D).
    Returns list of (instance_id, label_name, polyline_ndarray).
    """
    ego_trans = np.array(ego_translation, dtype=np.float64)
    ego_rot = ego_rotation  # Quaternion object

    def transform_to_ego(points):
        """Transform global 2D points to ego frame."""
        pts = np.array(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        pts = pts[:, :2] - ego_trans[:2]
        rot_mat = ego_rot.inverse.rotation_matrix[:2, :2]
        pts = np.dot(rot_mat, pts.T).T
        return pts

    radius = max(roi_size[0], roi_size[1]) / 2 + 10
    hx, hy = roi_size[0] / 2, roi_size[1] / 2
    local_patch = box(-hx, -hy, hx, hy)

    gt_instances = []

    # Dividers
    for layer_name in ['lane_divider', 'road_divider']:
        records = nusc_map.get_records_in_radius(
            ego_trans[0], ego_trans[1], radius, [layer_name])
        if layer_name not in records:
            continue
        for token in records[layer_name]:
            record = nusc_map.get(layer_name, token)
            line_points = []
            if 'node_tokens' in record:
                for node_token in record['node_tokens']:
                    node_record = nusc_map.get('node', node_token)
                    if 'x' in node_record and 'y' in node_record:
                        line_points.append([node_record['x'], node_record['y']])
            if len(line_points) < 2:
                continue
            line_points = np.array(line_points)
            line_points = transform_to_ego(line_points)

            line = LineString(line_points)
            clipped = line.intersection(local_patch)
            if clipped.is_empty:
                continue
            for seg in _split_collections(clipped):
                if seg.geom_type == 'LineString' and seg.length > 0.5:
                    coords = np.array(seg.coords)[:, :2]
                    gt_instances.append((f"{layer_name}_{token}", 'divider', coords))

    # Ped crossing
    records = nusc_map.get_records_in_radius(
        ego_trans[0], ego_trans[1], radius, ['ped_crossing'])
    if 'ped_crossing' in records:
        for token in records['ped_crossing']:
            record = nusc_map.get('ped_crossing', token)
            if 'polygon_token' not in record:
                continue
            polygon = nusc_map.extract_polygon(record['polygon_token'])
            if not polygon.is_valid or polygon.is_empty:
                continue
            poly_pts = np.array(polygon.exterior.coords)[:, :2]
            poly_pts = transform_to_ego(poly_pts)
            polygon_ego = Polygon(poly_pts)
            clipped = polygon_ego.intersection(local_patch)
            if clipped.is_empty:
                continue
            # Extract boundary of clipped polygon
            boundaries = []
            if clipped.geom_type == 'Polygon':
                boundaries = [clipped.exterior]
            elif clipped.geom_type == 'MultiPolygon':
                boundaries = [p.exterior for p in clipped.geoms]
            for bdry in boundaries:
                if bdry.length > 1.0:
                    coords = np.array(bdry.coords)[:, :2]
                    gt_instances.append((f"ped_{token}", 'ped_crossing', coords))

    # Boundary (road_segment + lane)
    for layer_name in ['road_segment', 'lane']:
        records = nusc_map.get_records_in_radius(
            ego_trans[0], ego_trans[1], radius, [layer_name])
        if layer_name not in records:
            continue
        for token in records[layer_name]:
            record = nusc_map.get(layer_name, token)
            if 'polygon_token' not in record:
                continue
            polygon = nusc_map.extract_polygon(record['polygon_token'])
            if not polygon.is_valid or polygon.is_empty:
                continue
            # Extract all rings (exterior + interior)
            rings = [polygon.exterior] + list(polygon.interiors)
            for ring_idx, ring in enumerate(rings):
                ring_pts = np.array(ring.coords)[:, :2]
                ring_pts = transform_to_ego(ring_pts)
                ring_line = LineString(ring_pts)
                clipped = ring_line.intersection(local_patch)
                if clipped.is_empty:
                    continue
                for seg_idx, seg in enumerate(_split_collections(clipped)):
                    if seg.geom_type == 'LineString' and seg.length > 1.0:
                        coords = np.array(seg.coords)[:, :2]
                        gt_instances.append((
                            f"boundary_{layer_name}_{token}_{ring_idx}",
                            'boundary', coords))

    return gt_instances


def load_all_gt(ann_file, dataroot, roi_size):
    """Load GT for all frames."""
    cache_file = f'./tmp_gts_eval_v2_{roi_size[0]}x{roi_size[1]}.pkl'
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

        # Compute lidar2global (for mAP GT extraction)
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']
        lidar2global = ego2global @ lidar2ego

        l2g_trans = list(lidar2global[:3, 3])
        l2g_rot = list(Quaternion(matrix=lidar2global).q)

        # Ego pose (for mAS GT extraction + cross-frame transform)
        ego_translation = np.array(sample['e2g_translation'], dtype=np.float64)
        ego_rotation = Quaternion(sample['e2g_rotation'])

        # mAP GT: lidar frame + (-90° rotation)
        gt_by_label = extract_map_gt_lidar_frame(
            nusc_maps[location], map_explorers[location], location,
            l2g_trans, l2g_rot, roi_size)

        # mAS GT: ego frame (MapStableTest methodology)
        gt_inst = extract_mas_gt_ego_frame(
            nusc_maps[location], ego_translation, ego_rotation, roi_size)

        gts[token] = gt_by_label
        gt_instances_all[token] = gt_inst
        frame_infos[token] = {
            'scene_name': sample['scene_name'],
            'sample_idx': sample['sample_idx'],
            'ego_translation': ego_translation,
            'ego_rotation': ego_rotation,
            'lidar2ego_translation': np.array(sample['lidar2ego_translation']),
            'token': token,
        }
        pbar.update()

    print(f'\nSaving GT cache to {cache_file}')
    mmcv.dump({'gts': gts, 'gt_instances_all': gt_instances_all,
               'frame_infos': frame_infos}, cache_file)
    return gts, gt_instances_all, frame_infos


# ============================================================
# Section 4: mAP computation (unchanged)
# ============================================================

def _evaluate_single(pred_vectors, scores, groundtruth, thresholds):
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
        result_dict[ID2CAT[label]] = {'num_gts': num_gts[label], 'num_preds': num_preds[label]}
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
        pool.close(); pool.join()
    result_dict['mAP'] = sum_mAP / len(ID2CAT)
    print(f'mAP done in {time() - start:.2f}s')
    return result_dict


# ============================================================
# Section 5: mAS computation (ego-based, MapStableTest-aligned)
# ============================================================

def undo_maptracker_rotation(pts):
    """Undo MapTracker's -90° rotation: (x,y) → (-y, x)."""
    pts = np.array(pts, dtype=np.float64)
    return np.column_stack([-pts[:, 1], pts[:, 0]])


def pred_lidar_to_ego(pts, lidar2ego_translation):
    """Approximate transform from lidar frame to ego frame (translation only)."""
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] += lidar2ego_translation[0]
    pts[:, 1] += lidar2ego_translation[1]
    return pts


def filter_to_range(poly, x_range, y_range):
    poly = np.array(poly, dtype=np.float64)
    if len(poly) == 0:
        return poly
    mask = ((poly[:, 0] >= x_range[0]) & (poly[:, 0] <= x_range[1]) &
            (poly[:, 1] >= y_range[0]) & (poly[:, 1] <= y_range[1]))
    return poly[mask]


def compute_presence(cur_score, prev_score, threshold=DET_THRESHOLD):
    cur_det = cur_score >= threshold
    prev_det = prev_score >= threshold
    return 1.0 if cur_det == prev_det else 0.5


def compute_localization(cur_poly, prev_poly, beta=BETA):
    avg_L1 = dynamic_axis_resample_deviation(cur_poly, prev_poly, N=N_STABILITY_SAMPLES)
    if avg_L1 < 0:
        return -1.0
    return float(np.clip(1.0 - avg_L1 / beta, 0.0, 1.0))


def compute_shape(cur_poly, prev_poly):
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


def build_frame_pairs(frame_infos, interval=INTERVAL):
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
    """Process a single frame pair for mAS using ego-based methodology."""
    (cur_gt_inst, prev_gt_inst, cur_pred, prev_pred,
     cur_ego_trans, cur_ego_rot, prev_ego_trans, prev_ego_rot,
     lidar2ego_trans, roi_size) = args

    x_range = (-roi_size[0] / 2, roi_size[0] / 2)
    y_range = (-roi_size[1] / 2, roi_size[1] / 2)
    class_names = ['divider', 'ped_crossing', 'boundary']

    if not cur_gt_inst or not prev_gt_inst:
        return []

    # Undo MapTracker's -90° rotation + lidar→ego correction on predictions
    cur_vecs, cur_scores_list, cur_type_names = [], [], []
    for i in range(len(cur_pred.get('labels', []))):
        pts = undo_maptracker_rotation(cur_pred['vectors'][i])
        pts = pred_lidar_to_ego(pts, lidar2ego_trans)
        cur_vecs.append(pts)
        cur_scores_list.append(cur_pred['scores'][i])
        cur_type_names.append(ID2NAME.get(cur_pred['labels'][i], 'unknown'))

    prev_vecs, prev_scores_list, prev_type_names = [], [], []
    for i in range(len(prev_pred.get('labels', []))):
        pts = undo_maptracker_rotation(prev_pred['vectors'][i])
        pts = pred_lidar_to_ego(pts, lidar2ego_trans)
        prev_vecs.append(pts)
        prev_scores_list.append(prev_pred['scores'][i])
        prev_type_names.append(ID2NAME.get(prev_pred['labels'][i], 'unknown'))

    # Build GT lookups
    cur_inst_dict = {iid: (lname, coords) for iid, lname, coords in cur_gt_inst}
    prev_inst_dict = {iid: (lname, coords) for iid, lname, coords in prev_gt_inst}
    common_ids = set(cur_inst_dict.keys()) & set(prev_inst_dict.keys())
    if not common_ids:
        return []

    pair_results = []

    for cat_name in class_names:
        label_common = [iid for iid in common_ids if cur_inst_dict[iid][0] == cat_name]
        if not label_common:
            continue

        # All GT of this class per frame (for matching)
        cur_all_gt = [(iid, coords) for iid, lname, coords in cur_gt_inst if lname == cat_name]
        prev_all_gt = [(iid, coords) for iid, lname, coords in prev_gt_inst if lname == cat_name]

        cur_all_gt_ids = [x[0] for x in cur_all_gt]
        cur_all_gt_polys = [x[1] for x in cur_all_gt]
        cur_all_gt_types = [cat_name] * len(cur_all_gt)

        prev_all_gt_ids = [x[0] for x in prev_all_gt]
        prev_all_gt_polys = [x[1] for x in prev_all_gt]
        prev_all_gt_types = [cat_name] * len(prev_all_gt)

        # Filter predictions by class
        cur_cls_vecs = [v for v, t in zip(cur_vecs, cur_type_names) if t == cat_name]
        cur_cls_scrs = [s for s, t in zip(cur_scores_list, cur_type_names) if t == cat_name]
        cur_cls_types = [cat_name] * len(cur_cls_vecs)

        prev_cls_vecs = [v for v, t in zip(prev_vecs, prev_type_names) if t == cat_name]
        prev_cls_scrs = [s for s, t in zip(prev_scores_list, prev_type_names) if t == cat_name]
        prev_cls_types = [cat_name] * len(prev_cls_vecs)

        # MapTR assigner matching (MapStableTest methodology)
        if len(cur_all_gt_polys) == 0:
            continue

        cur_aligned_polys, cur_aligned_scores = align_det_and_gt_by_maptr_assigner(
            cur_cls_vecs, cur_cls_types, cur_all_gt_polys, cur_all_gt_types,
            class_names, pc_range=PC_RANGE, det_scores=cur_cls_scrs)

        prev_aligned_polys, prev_aligned_scores = align_det_and_gt_by_maptr_assigner(
            prev_cls_vecs, prev_cls_types, prev_all_gt_polys, prev_all_gt_types,
            class_names, pc_range=PC_RANGE, det_scores=prev_cls_scrs)

        # Build match dicts: gt_id → (aligned_pred, score)
        cur_match = {}
        for idx, iid in enumerate(cur_all_gt_ids):
            if idx < len(cur_aligned_polys):
                cur_match[iid] = (cur_aligned_polys[idx], float(cur_aligned_scores[idx]))
        prev_match = {}
        for idx, iid in enumerate(prev_all_gt_ids):
            if idx < len(prev_aligned_polys):
                prev_match[iid] = (prev_aligned_polys[idx], float(prev_aligned_scores[idx]))

        # Evaluate stability for common GT instances
        for inst_id in label_common:
            if inst_id not in cur_match or inst_id not in prev_match:
                continue

            cur_poly, cur_score = cur_match[inst_id]
            prev_poly_raw, prev_score = prev_match[inst_id]

            if cur_score < BOTH_MISS_THRESHOLD and prev_score < BOTH_MISS_THRESHOLD:
                continue

            pres = compute_presence(cur_score, prev_score)

            if len(cur_poly) < 2 or len(prev_poly_raw) < 2:
                pair_results.append((cat_name, pres, 0.0, 0.0, 0.0))
                continue

            # Cross-frame transform using ego poses (MapStableTest methodology)
            prev_poly_aligned = mst_transform(
                prev_poly_raw,
                src_translation=prev_ego_trans,
                src_rotation=prev_ego_rot,
                dst_translation=cur_ego_trans,
                dst_rotation=cur_ego_rot)

            # Range filter (prev only, cur is already in-range)
            prev_poly_f = filter_to_range(prev_poly_aligned, x_range, y_range)

            if len(cur_poly) < 2 or len(prev_poly_f) < 2:
                pair_results.append((cat_name, pres, 0.0, 0.0, 0.0))
                continue

            loc = compute_localization(cur_poly, prev_poly_f)
            if loc < 0:
                loc = 0.0
            shp = compute_shape(cur_poly, prev_poly_f)
            if shp < 0:
                shp = 0.0

            base = np.clip(LOC_WEIGHT * loc + (1 - LOC_WEIGHT) * shp, 0.0, 1.0)
            si = pres * base
            pair_results.append((cat_name, pres, loc, shp, si))

    return pair_results


def compute_mAS(results, gt_instances_all, frame_infos, roi_size=ROI_SIZE,
                interval=INTERVAL, n_workers=N_WORKERS):
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

        cur_info = frame_infos[cur_token]
        prev_info = frame_infos[prev_token]

        worker_args.append((
            cur_gt_inst, prev_gt_inst, cur_pred, prev_pred,
            cur_info['ego_translation'], cur_info['ego_rotation'],
            prev_info['ego_translation'], prev_info['ego_rotation'],
            cur_info['lidar2ego_translation'],
            roi_size,
        ))

    print(f'  {len(worker_args)} valid pairs to process')

    # Note: MapTR assigner uses CUDA, so we run single-threaded
    all_pair_results = []
    pbar = mmcv.ProgressBar(len(worker_args))
    for args in worker_args:
        all_pair_results.append(_process_single_pair(args))
        pbar.update()

    # Aggregate
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
            mas_result[cat_name] = {'PC': 0.0, 'LS': 0.0, 'SS': 0.0, 'AS': 0.0, 'count': 0}

    for k in ['PC', 'LS', 'SS', 'AS']:
        vals = [mas_result[c][k] for c in ['divider', 'ped_crossing', 'boundary']]
        mas_result[f'm{k}'] = float(np.mean(vals))

    print(f'\nmAS done in {time() - start:.2f}s')
    return mas_result


# ============================================================
# Section 6: Output
# ============================================================

def print_results(map_result, mas_result, thresholds):
    print('\n' + '=' * 70)
    print('  mAP Results')
    print('=' * 70)
    table = prettytable.PrettyTable(
        ['category', 'num_preds', 'num_gts'] +
        [f'AP@{t}' for t in thresholds] + ['AP'])
    for label_id in sorted(ID2CAT.keys()):
        cat = ID2CAT[label_id]
        r = map_result.get(cat, {})
        table.add_row([cat, r.get('num_preds', 0), r.get('num_gts', 0),
                        *[round(r.get(f'AP@{t}', 0), 4) for t in thresholds],
                        round(r.get('AP', 0), 4)])
    print(table)
    print(f"mAP = {map_result.get('mAP', 0):.4f}")

    print('\n' + '=' * 70)
    print(f'  mAS Results  (ω={LOC_WEIGHT}, β={BETA}, interval={INTERVAL})')
    print('=' * 70)
    table2 = prettytable.PrettyTable(
        ['category', 'count', 'Presence', 'Localization', 'Shape', 'SI'])
    for cat in ['divider', 'ped_crossing', 'boundary']:
        r = mas_result.get(cat, {})
        table2.add_row([cat, r.get('count', 0),
                        round(r.get('PC', 0), 4), round(r.get('LS', 0), 4),
                        round(r.get('SS', 0), 4), round(r.get('AS', 0), 4)])
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
    parser = argparse.ArgumentParser(description='Unified mAP + mAS evaluator (v2)')
    parser.add_argument('--submission', required=True, help='Path to submission_vector.json')
    parser.add_argument('--dataroot', required=True, help='NuScenes dataset root')
    parser.add_argument('--ann-file', required=True, help='Annotation pkl file')
    parser.add_argument('--roi-size', nargs=2, type=int, default=list(ROI_SIZE),
                        help='ROI size (default: 60 30)')
    parser.add_argument('--interval', type=int, default=INTERVAL)
    parser.add_argument('--n-workers', type=int, default=N_WORKERS)
    args = parser.parse_args()

    roi_size = tuple(args.roi_size)
    thresholds = [1.0, 1.5, 2.0] if roi_size == (100, 50) else THRESHOLDS

    print(f'Loading submission from {args.submission}')
    submission = mmcv.load(args.submission)
    results = submission['results']

    gts, gt_instances_all, frame_infos = load_all_gt(
        args.ann_file, args.dataroot, roi_size)

    map_result = compute_mAP(results, gts, thresholds, n_workers=args.n_workers)
    mas_result = compute_mAS(results, gt_instances_all, frame_infos,
                             roi_size=roi_size, interval=args.interval,
                             n_workers=args.n_workers)
    print_results(map_result, mas_result, thresholds)
    return map_result, mas_result


if __name__ == '__main__':
    main()
