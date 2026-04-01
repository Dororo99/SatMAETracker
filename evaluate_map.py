#!/usr/bin/env python
"""
Standalone mAP + mAS evaluator for online HD mapping.

Usage:
    python evaluate_map.py \
        --submission path/to/submission_vector.json \
        --dataroot ./datasets/nuscenes \
        --ann-file ./datasets/nuscenes/nuscenes_map_infos_val_newsplit.pkl

Only swap --submission per model. Everything else is dataset-fixed.
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
from copy import deepcopy
from collections import defaultdict

from pyquaternion import Quaternion
from shapely.geometry import LineString, box, Polygon, MultiPolygon, MultiLineString, LinearRing
from shapely import ops, affinity, strtree
from scipy.optimize import linear_sum_assignment

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw

# ============================================================
# Config (modify here if needed)
# ============================================================
ROI_SIZE = (60, 30)           # BEV range: 60m x-axis, 30m y-axis
THRESHOLDS = [0.5, 1.0, 1.5]  # mAP chamfer distance thresholds
INTERP_NUM = 200               # interpolation points for mAP eval
N_WORKERS = 16                 # parallel workers
INTERVAL = 2                   # mAS consecutive frame interval
LOC_WEIGHT = 0.7               # SI = presence * clip(loc*0.7 + shape*0.3)  (paper default ω=0.7)
DET_THRESHOLD = 0.3            # presence detection threshold
BOTH_MISS_THRESHOLD = 0.01    # skip if both frames score < this
NUM_SAMPLES_IOU = 100          # polyline IoU sampling points
CAT2ID = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
ID2CAT = {v: k for k, v in CAT2ID.items()}
MAPS = ['boston-seaport', 'singapore-hollandvillage',
        'singapore-onenorth', 'singapore-queenstown']


# ============================================================
# Section 1: Distance & AP utilities
# ============================================================

def chamfer_distance_batch(pred_lines, gt_lines):
    """Batch chamfer distance. (m, num_pts, 2) x (n, num_pts, 2) -> (m, n)."""
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
    dist_matrix = (dist1 + dist2).transpose(0, 1) / (2 * num_pts)
    return dist_matrix.numpy()


def instance_match(pred_lines, scores, gt_lines, thresholds):
    """Match predictions to GT via chamfer distance. Returns TP/FP per threshold."""
    num_preds = pred_lines.shape[0]
    num_gts = gt_lines.shape[0]
    tp_fp_list = []
    tp = np.zeros(num_preds, dtype=np.float32)
    fp = np.zeros(num_preds, dtype=np.float32)

    if num_gts == 0:
        fp[...] = 1
        for _ in thresholds:
            tp_fp_list.append((tp.copy(), fp.copy()))
        return tp_fp_list
    if num_preds == 0:
        for _ in thresholds:
            tp_fp_list.append((tp.copy(), fp.copy()))
        return tp_fp_list

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
    """Compute AP using area under PR curve."""
    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]
    zeros = np.zeros((1, 1), dtype=recalls.dtype)
    ones = np.ones((1, 1), dtype=recalls.dtype)
    mrec = np.hstack((zeros, recalls, ones))
    mpre = np.hstack((zeros, precisions, zeros))
    for i in range(mpre.shape[1] - 1, 0, -1):
        mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
    ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]
    ap = np.sum((mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1])
    return ap


def interp_fixed_num(vector, num_pts):
    """Interpolate polyline to fixed number of points."""
    line = LineString(vector)
    distances = np.linspace(0, line.length, num_pts)
    sampled_points = np.array(
        [list(line.interpolate(d).coords) for d in distances]).squeeze()
    return sampled_points


# ============================================================
# Section 2: NuScenes GT extraction with instance tokens
# ============================================================

def _split_collections(geom):
    """Split Multi-geoms to list."""
    if 'Multi' in geom.geom_type:
        return [g for g in geom.geoms if g.is_valid and not g.is_empty]
    if geom.is_valid and not geom.is_empty:
        return [geom]
    return []


def _get_ped_crossing_contour(polygon, local_patch):
    """Extract ped crossing as a closed polyline."""
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
            new_line = list(lines.coords) + [start]
            lines = LineString(new_line)
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

    # handle small peds
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
            cos_checks = np.array([(1 - np.abs(c) < 0.001) for c in [cos_00, cos_01, cos_10, cos_11]])
            if cos_checks.sum() == 2 and ch_area_ratio > 0.8:
                final2[-1] = final2[-1].union(o)
                remain_idx2.remove(o_idx)

    # filter small
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


def extract_gt_for_frame(nusc_map, map_explorer, location, lidar2global_translation,
                         lidar2global_rotation, roi_size):
    """
    Extract GT polylines with instance tokens for one frame.

    Returns:
        gt_by_label: {label_id: [ndarray, ...]}  (for mAP, same as existing evaluator)
        gt_instances: [(instance_id, label_id, ndarray), ...]  (for mAS)
    """
    patch_size_lidar = (roi_size[0], roi_size[1])
    map_pose = lidar2global_translation[:2]
    rotation = Quaternion(lidar2global_rotation)
    patch_box = (map_pose[0], map_pose[1], patch_size_lidar[0], patch_size_lidar[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180
    local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2,
                       roi_size[0] / 2, roi_size[1] / 2)
    patch_x, patch_y = patch_box[0], patch_box[1]
    patch_geom = map_explorer.get_patch_coord(patch_box, patch_angle)

    gt_by_label = {label: [] for label in ID2CAT.keys()}
    gt_instances = []

    def to_local(geom):
        geom = affinity.rotate(geom, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
        geom = affinity.affine_transform(geom, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
        return geom

    # Rotated local patch (after -90 rotation, x and y swap)
    rotated_local_patch = box(-roi_size[1] / 2, -roi_size[0] / 2,
                               roi_size[1] / 2, roi_size[0] / 2)

    # Standard local patch (no -90 rotation, for mAS GT)
    standard_local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2,
                                roi_size[0] / 2, roi_size[1] / 2)

    # --- Dividers (lane_divider + road_divider) ---
    for layer_name in ['lane_divider', 'road_divider']:
        records = getattr(nusc_map, layer_name)
        for record in records:
            line = nusc_map.extract_line(record['line_token'])
            if line.is_empty:
                continue
            new_line = line.intersection(patch_geom)
            if new_line.is_empty:
                continue
            new_line = to_local(new_line)

            # mAP GT: with -90 rotation (MapTracker convention)
            new_line_rotated = affinity.rotate(new_line, -90, origin=(0, 0), use_radians=False)
            for single_line in _split_collections(new_line_rotated):
                if single_line.geom_type != 'LineString':
                    continue
                simplified = single_line.simplify(0.2, preserve_topology=True)
                coords_s = np.array(simplified.coords)[:, :2]
                gt_by_label[CAT2ID['divider']].append(coords_s)

            # mAS GT: NO -90 rotation (standard ego frame, MapStableTest convention)
            sub_lines = _split_collections(new_line)
            for sub_idx, single_line in enumerate(sub_lines):
                if single_line.geom_type != 'LineString':
                    continue
                simplified = single_line.simplify(0.2, preserve_topology=True)
                coords_s = np.array(simplified.coords)[:, :2]
                inst_id = f"{layer_name}_{record['token']}_{sub_idx}"
                gt_instances.append((inst_id, CAT2ID['divider'], coords_s))

    # --- Boundary (for mAP: union; for mAS: individual records) ---
    # mAP GT: union of road_segment + lane -> exterior contours
    polygon_list = []
    for layer_name in ['road_segment', 'lane']:
        records = getattr(nusc_map, layer_name)
        for record in records:
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
        union_roads_lanes = ops.unary_union(polygon_list)
        max_x = roi_size[0] / 2
        max_y = roi_size[1] / 2
        local_patch_boundary = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_roads_lanes.geom_type != 'MultiPolygon':
            union_roads_lanes = MultiPolygon([union_roads_lanes])
        for poly in union_roads_lanes.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        for ext in exteriors:
            if ext.is_ccw:
                ext = LinearRing(list(ext.coords)[::-1])
            lines = ext.intersection(local_patch_boundary)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            for l in _split_collections(lines):
                if l.geom_type == 'LineString':
                    simplified = l.simplify(0.2, preserve_topology=True)
                    coords_s = np.array(simplified.coords)[:, :2]
                    gt_by_label[CAT2ID['boundary']].append(coords_s)
        for inter in interiors:
            if not inter.is_ccw:
                inter = LinearRing(list(inter.coords)[::-1])
            lines = inter.intersection(local_patch_boundary)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            for l in _split_collections(lines):
                if l.geom_type == 'LineString':
                    simplified = l.simplify(0.2, preserve_topology=True)
                    coords_s = np.array(simplified.coords)[:, :2]
                    gt_by_label[CAT2ID['boundary']].append(coords_s)

    # mAS GT: extract boundary rings from ORIGINAL polygon, NO -90 rotation (standard ego frame)
    for layer_name in ['road_segment', 'lane']:
        records = getattr(nusc_map, layer_name)
        for record in records:
            polygon = nusc_map.extract_polygon(record['polygon_token'])
            if not polygon.is_valid or polygon.is_empty:
                continue
            if not polygon.intersects(patch_geom):
                continue
            # Extract rings from ORIGINAL polygon (before clipping!)
            rings = [polygon.exterior]
            for interior in polygon.interiors:
                rings.append(interior)
            for ring in rings:
                ring_line = LineString(ring.coords)
                # Transform to local only (NO -90 rotation)
                ring_local = to_local(ring_line)
                # Clip with standard local patch
                clipped = ring_local.intersection(standard_local_patch)
                if clipped.is_empty:
                    continue
                for seg in _split_collections(clipped):
                    if seg.geom_type == 'LineString' and seg.length > 1.0:
                        simplified = seg.simplify(0.2, preserve_topology=True)
                        coords_s = np.array(simplified.coords)[:, :2]
                        inst_id = f"boundary_{layer_name}_{record['token']}"
                        gt_instances.append((inst_id, CAT2ID['boundary'], coords_s))

    # --- Ped Crossing (for mAP: merged; for mAS: individual records) ---
    # mAP GT: merged ped crossings
    ped_layer = map_explorer._get_layer_polygon(patch_box, patch_angle, 'ped_crossing')
    ped_crossings = []
    for p in ped_layer:
        ped_crossings.extend(_split_collections(p))
    ped_crossings = _union_ped(ped_crossings, local_patch)

    for p in ped_crossings:
        p = affinity.rotate(p, -90, origin=(0, 0), use_radians=False)
        line = _get_ped_crossing_contour(p, rotated_local_patch)
        if line is not None:
            simplified = line.simplify(0.2, preserve_topology=True)
            coords_s = np.array(simplified.coords)[:, :2]
            gt_by_label[CAT2ID['ped_crossing']].append(coords_s)

    # mAS GT: extract ped_crossing rings from ORIGINAL polygon, NO -90 rotation
    ped_records = getattr(nusc_map, 'ped_crossing')
    for record in ped_records:
        polygon = nusc_map.extract_polygon(record['polygon_token'])
        if not polygon.is_valid or polygon.is_empty:
            continue
        if not polygon.intersects(patch_geom):
            continue
        # Extract exterior from ORIGINAL polygon, NO -90 rotation
        ring_line = LineString(polygon.exterior.coords)
        ring_local = to_local(ring_line)
        # Clip with standard local patch
        clipped = ring_local.intersection(standard_local_patch)
        if clipped.is_empty:
            continue
        for seg in _split_collections(clipped):
            if seg.geom_type == 'LineString' and seg.length > 1.0:
                simplified = seg.simplify(0.2, preserve_topology=True)
                coords_s = np.array(simplified.coords)[:, :2]
                inst_id = f"ped_{record['token']}"
                gt_instances.append((inst_id, CAT2ID['ped_crossing'], coords_s))

    return gt_by_label, gt_instances


def load_all_gt(ann_file, dataroot, roi_size):
    """
    Load GT for all frames.

    Returns:
        gts: {token: {label_id: [ndarray, ...]}}
        gt_instances_all: {token: [(instance_id, label_id, ndarray), ...]}
        frame_infos: {token: {scene_name, sample_idx, lidar2global_4x4}}
    """
    cache_file = f'./tmp_gts_eval_standalone_{roi_size[0]}x{roi_size[1]}.pkl'
    if os.path.exists(cache_file):
        print(f'Loading cached GT from {cache_file}')
        cached = mmcv.load(cache_file)
        return cached['gts'], cached['gt_instances_all'], cached['frame_infos']

    print('Loading annotations...')
    ann = mmcv.load(ann_file)
    if isinstance(ann, dict) and 'samples' in ann:
        samples = ann['samples']
    elif isinstance(ann, list):
        samples = ann
    else:
        samples = ann

    # Load NuScenes maps
    nusc_maps = {}
    map_explorers = {}
    for loc in MAPS:
        nusc_maps[loc] = NuScenesMap(dataroot=dataroot, map_name=loc)
        map_explorers[loc] = NuScenesMapExplorer(nusc_maps[loc])

    gts = {}
    gt_instances_all = {}
    frame_infos = {}

    print(f'Extracting GT for {len(samples)} frames...')
    pbar = mmcv.ProgressBar(len(samples))
    for sample in samples:
        token = sample['token']
        location = sample['location']

        # Compute lidar2global (same as nusc_dataset.py)
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']

        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        # Pose = full lidar2global (consistent with to_local used for GT extraction)
        pose_4x4 = lidar2global.copy()

        gt_by_label, gt_inst = extract_gt_for_frame(
            nusc_maps[location], map_explorers[location], location,
            lidar2global_translation, lidar2global_rotation, roi_size)

        gts[token] = gt_by_label
        gt_instances_all[token] = gt_inst
        frame_infos[token] = {
            'scene_name': sample['scene_name'],
            'sample_idx': sample['sample_idx'],
            'lidar2global_4x4': pose_4x4,
            'token': token,
        }
        pbar.update()

    print(f'\nSaving GT cache to {cache_file}')
    mmcv.dump({'gts': gts, 'gt_instances_all': gt_instances_all,
               'frame_infos': frame_infos}, cache_file)
    return gts, gt_instances_all, frame_infos


# ============================================================
# Section 3: mAP computation
# ============================================================

def _evaluate_single(pred_vectors, scores, groundtruth, thresholds):
    """Single-frame matching for one class."""
    pred_lines = []
    for vector in pred_vectors:
        vector = np.array(vector)
        vector_interp = interp_fixed_num(vector, INTERP_NUM)
        pred_lines.append(vector_interp)
    if pred_lines:
        pred_lines = np.stack(pred_lines)
    else:
        pred_lines = np.zeros((0, INTERP_NUM, 2))

    gt_lines = []
    for vector in groundtruth:
        vector_interp = interp_fixed_num(vector, INTERP_NUM)
        gt_lines.append(vector_interp)
    if gt_lines:
        gt_lines = np.stack(gt_lines)
    else:
        gt_lines = np.zeros((0, INTERP_NUM, 2))

    scores = np.array(scores)
    tp_fp_list = instance_match(pred_lines, scores, gt_lines, thresholds)
    tp_fp_score_by_thr = {}
    for i, thr in enumerate(thresholds):
        tp, fp = tp_fp_list[i]
        tp_fp_score = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
        tp_fp_score_by_thr[thr] = tp_fp_score
    return tp_fp_score_by_thr


def compute_mAP(results, gts, thresholds, n_workers=N_WORKERS):
    """Compute mAP (same logic as VectorEvaluate.evaluate)."""
    samples_by_cls = {label: [] for label in ID2CAT.keys()}
    num_gts = {label: 0 for label in ID2CAT.keys()}
    num_preds = {label: 0 for label in ID2CAT.keys()}

    for token, gt in gts.items():
        pred = results.get(token, {'vectors': [], 'scores': [], 'labels': []})
        vectors_by_cls = {label: [] for label in ID2CAT.keys()}
        scores_by_cls = {label: [] for label in ID2CAT.keys()}

        for i in range(len(pred.get('labels', []))):
            label = pred['labels'][i]
            vector = pred['vectors'][i]
            score = pred['scores'][i]
            if label in vectors_by_cls:
                vectors_by_cls[label].append(vector)
                scores_by_cls[label].append(score)

        for label in ID2CAT.keys():
            new_sample = (vectors_by_cls[label], scores_by_cls[label], gt[label])
            num_gts[label] += len(gt[label])
            num_preds[label] += len(scores_by_cls[label])
            samples_by_cls[label].append(new_sample)

    result_dict = {}
    print(f'\nComputing mAP for {len(ID2CAT)} categories...')
    start = time()

    if n_workers > 0:
        pool = Pool(n_workers)

    sum_mAP = 0
    for label in ID2CAT.keys():
        samples = samples_by_cls[label]
        result_dict[ID2CAT[label]] = {
            'num_gts': num_gts[label],
            'num_preds': num_preds[label],
        }
        sum_AP = 0
        fn = partial(_evaluate_single, thresholds=thresholds)
        if n_workers > 0:
            tpfp_score_list = pool.starmap(fn, samples)
        else:
            tpfp_score_list = [fn(*s) for s in samples]

        for thr in thresholds:
            tp_fp_score = [i[thr] for i in tpfp_score_list]
            tp_fp_score = np.vstack(tp_fp_score)
            sort_inds = np.argsort(-tp_fp_score[:, -1])
            tp = np.cumsum(tp_fp_score[sort_inds, 0])
            fp = np.cumsum(tp_fp_score[sort_inds, 1])
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts[label], eps)
            precisions = tp / np.maximum((tp + fp), eps)
            AP = average_precision(recalls, precisions)
            sum_AP += AP
            result_dict[ID2CAT[label]][f'AP@{thr}'] = AP

        AP = sum_AP / len(thresholds)
        sum_mAP += AP
        result_dict[ID2CAT[label]]['AP'] = AP

    if n_workers > 0:
        pool.close()
        pool.join()

    mAP = sum_mAP / len(ID2CAT)
    result_dict['mAP'] = mAP
    print(f'mAP computation done in {time() - start:.2f}s')
    return result_dict


# ============================================================
# Section 4: mAS computation
# ============================================================

def transform_points_between_frames(points, src_pose_4x4, dst_pose_4x4):
    """Transform 2D points from src frame to dst frame via global coords."""
    if len(points) == 0:
        return points.copy()
    pts = np.array(points, dtype=np.float64)
    n = pts.shape[0]
    # to homogeneous 3D (z=0)
    pts_homo = np.ones((n, 4), dtype=np.float64)
    pts_homo[:, 0] = pts[:, 0]
    pts_homo[:, 1] = pts[:, 1]
    pts_homo[:, 2] = 0.0

    # src_local -> global -> dst_local
    transform = np.linalg.inv(dst_pose_4x4) @ src_pose_4x4
    transformed = (transform @ pts_homo.T).T
    return transformed[:, :2].astype(np.float64)


def compute_presence(cur_score, prev_score, threshold=DET_THRESHOLD):
    """Presence consistency (paper definition):
    1.0 if both detect or both miss, 0.5 if flickering (one-sided)."""
    cur_det = cur_score >= threshold
    prev_det = prev_score >= threshold
    if cur_det != prev_det:
        return 0.5
    return 1.0


def interpolate_polyline_y(poly, x_samples):
    """Interpolate y values at given x samples along a polyline.
    Polyline is sorted by x-coordinate before interpolation."""
    if len(poly) == 0:
        return np.zeros_like(x_samples)
    if len(poly) == 1:
        return np.full_like(x_samples, poly[0][1])

    # Sort polyline by x-coordinate (required for segment traversal)
    poly = np.array(poly)
    sort_idx = np.argsort(poly[:, 0])
    poly = poly[sort_idx]

    y_samples = np.zeros_like(x_samples, dtype=np.float64)
    segments = list(zip(poly[:-1], poly[1:]))
    current_segment = 0

    for i, x in enumerate(x_samples):
        while (current_segment < len(segments) and
               x > segments[current_segment][1][0]):
            current_segment += 1
        if current_segment >= len(segments):
            y_samples[i] = poly[-1][1]
        elif x < segments[current_segment][0][0]:
            y_samples[i] = poly[0][1]
        else:
            p0, p1 = segments[current_segment]
            x0, y0 = p0[0], p0[1]
            x1, y1 = p1[0], p1[1]
            if x1 == x0:
                y_samples[i] = (y0 + y1) / 2.0
            else:
                t = (x - x0) / (x1 - x0)
                y_samples[i] = y0 + t * (y1 - y0)
    return y_samples


def poly_get_samples(poly, num_samples=NUM_SAMPLES_IOU):
    """Get x-direction uniform samples from a polyline (cur_pred only)."""
    if len(poly) == 0:
        return np.linspace(-30, 30, num_samples)
    x = poly[:, 0] if isinstance(poly, np.ndarray) else [p[0] for p in poly]
    min_x, max_x = float(np.min(x)), float(np.max(x))
    if min_x == max_x:
        min_x -= 0.1
        max_x += 0.1
    return np.linspace(min_x, max_x, num_samples)


def compute_localization(cur_pred, aligned_prev_pred):
    """Localization stability via polyline IoU (MapStableTest, standard ego frame).
    Samples along x (forward), measures y (lateral) deviation, normalizes by 15.0."""
    if len(cur_pred) == 0 or len(aligned_prev_pred) == 0:
        return -1.0
    x_samples = poly_get_samples(cur_pred, NUM_SAMPLES_IOU)
    y1 = interpolate_polyline_y(cur_pred, x_samples)
    y2 = interpolate_polyline_y(aligned_prev_pred, x_samples)
    total_diff = np.sum(np.abs(y1 - y2))
    iou = 1.0 - total_diff / (len(x_samples) * 15.0)
    return float(np.clip(iou, 0.0, 1.0))


def compute_shape(cur_pred, aligned_prev_pred):
    """Shape stability via curvature difference."""
    def curvature(poly):
        if len(poly) < 3:
            return 0.0
        angles = []
        for i in range(1, len(poly) - 1):
            p0, p1, p2 = poly[i - 1], poly[i], poly[i + 1]
            dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
            dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
            n1 = np.sqrt(dx1**2 + dy1**2)
            n2 = np.sqrt(dx2**2 + dy2**2)
            if n1 == 0 or n2 == 0:
                continue
            cos_a = np.clip((dx1*dx2 + dy1*dy2) / (n1 * n2), -1.0, 1.0)
            angles.append(np.arccos(cos_a))
        return np.mean(angles) if angles else 0.0

    if len(cur_pred) < 2 or len(aligned_prev_pred) < 2:
        return -1.0
    if len(cur_pred) < 3 or len(aligned_prev_pred) < 3:
        return 1.0

    cur_k = curvature(cur_pred)
    prev_k = curvature(aligned_prev_pred)
    shape_val = 1.0 - np.abs(cur_k - prev_k) / np.pi
    return max(0.0, shape_val)


def hungarian_match_preds_to_gt(pred_vectors, pred_scores, gt_polylines, gt_instance_ids):
    """
    Match predictions to ALL GT via Hungarian algorithm (paper Algorithm 3).

    Returns:
        dict: {instance_id: (matched_pred_vector, matched_score)}
              Only matched GT instances are included.
    """
    n_gt = len(gt_polylines)
    n_pred = len(pred_vectors)
    if n_gt == 0 or n_pred == 0:
        return {}

    # Interpolate for chamfer distance
    pred_interp = []
    for v in pred_vectors:
        v = np.array(v)
        if len(v) < 2:
            pred_interp.append(np.zeros((INTERP_NUM, 2)))
        else:
            pred_interp.append(interp_fixed_num(v, INTERP_NUM))
    pred_interp = np.stack(pred_interp)

    gt_interp = []
    for v in gt_polylines:
        if len(v) < 2:
            gt_interp.append(np.zeros((INTERP_NUM, 2)))
        else:
            gt_interp.append(interp_fixed_num(v, INTERP_NUM))
    gt_interp = np.stack(gt_interp)

    # Cost matrix: chamfer distance (n_pred x n_gt)
    cost_matrix = chamfer_distance_batch(pred_interp, gt_interp)

    # Hungarian algorithm (1-to-1 optimal matching)
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Build result dict: instance_id → (pred_vector, score)
    result = {}
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        inst_id = gt_instance_ids[gt_idx]
        score = float(pred_scores[pred_idx])
        # If same instance_id appears multiple times, keep the better match (lower cost)
        if inst_id not in result or cost_matrix[pred_idx, gt_idx] < result[inst_id][2]:
            result[inst_id] = (np.array(pred_vectors[pred_idx]), score, cost_matrix[pred_idx, gt_idx])

    # Remove cost from output
    return {k: (v[0], v[1]) for k, v in result.items()}


def build_frame_pairs(frame_infos, interval=INTERVAL):
    """Build consecutive frame pairs grouped by scene."""
    scenes = defaultdict(list)
    for token, info in frame_infos.items():
        scenes[info['scene_name']].append(info)

    pairs = []
    for scene_name, frames in scenes.items():
        frames = sorted(frames, key=lambda x: x['sample_idx'])
        for i in range(interval, len(frames)):
            cur_token = frames[i]['token']
            prev_token = frames[i - interval]['token']
            pairs.append((cur_token, prev_token))
    return pairs


def _undo_maptracker_rotation(pts):
    """Undo MapTracker's -90 rotation by applying +90: (x,y) → (-y, x)."""
    pts = np.array(pts, dtype=np.float64)
    return np.column_stack([-pts[:, 1], pts[:, 0]])



def _process_single_pair(args):
    """Process a single frame pair for mAS (paper Algorithm 2).
    Step 2.1: Hungarian match ALL predictions to ALL GT per frame.
    Step 2.2: Find common GT instances, compare matched predictions."""
    (cur_gt_inst, prev_gt_inst, cur_pred, prev_pred,
     cur_pose, prev_pose, roi_size, loc_weight) = args

    x_range = (-roi_size[0] / 2, roi_size[0] / 2)
    y_range = (-roi_size[1] / 2, roi_size[1] / 2)

    if not cur_gt_inst or not prev_gt_inst:
        return []

    # Rotate predictions +90 to standard ego frame (undo MapTracker's -90)
    cur_pred_vecs, cur_pred_scores, cur_pred_labels = [], [], []
    for i in range(len(cur_pred.get('labels', []))):
        cur_pred_vecs.append(_undo_maptracker_rotation(cur_pred['vectors'][i]))
        cur_pred_scores.append(cur_pred['scores'][i])
        cur_pred_labels.append(cur_pred['labels'][i])

    prev_pred_vecs, prev_pred_scores, prev_pred_labels = [], [], []
    for i in range(len(prev_pred.get('labels', []))):
        prev_pred_vecs.append(_undo_maptracker_rotation(prev_pred['vectors'][i]))
        prev_pred_scores.append(prev_pred['scores'][i])
        prev_pred_labels.append(prev_pred['labels'][i])

    # Unpack GT
    cur_gt_ids = [iid for iid, _, _ in cur_gt_inst]
    cur_gt_labels = [lab for _, lab, _ in cur_gt_inst]
    cur_gt_coords = [coords for _, _, coords in cur_gt_inst]

    prev_gt_ids = [iid for iid, _, _ in prev_gt_inst]
    prev_gt_labels = [lab for _, lab, _ in prev_gt_inst]
    prev_gt_coords = [coords for _, _, coords in prev_gt_inst]

    pair_results = []

    for label_id in ID2CAT.keys():
        # --- Step 2.1: Hungarian match ALL preds to ALL GT for this label ---
        # Current frame
        cur_label_gt_indices = [i for i, l in enumerate(cur_gt_labels) if l == label_id]
        cur_label_pred_indices = [i for i, l in enumerate(cur_pred_labels) if l == label_id]
        if not cur_label_gt_indices:
            continue

        cur_label_gt_coords = [cur_gt_coords[i] for i in cur_label_gt_indices]
        cur_label_gt_ids = [cur_gt_ids[i] for i in cur_label_gt_indices]
        cur_label_pred_v = [cur_pred_vecs[i] for i in cur_label_pred_indices]
        cur_label_pred_s = [cur_pred_scores[i] for i in cur_label_pred_indices]

        cur_matches = hungarian_match_preds_to_gt(
            cur_label_pred_v, cur_label_pred_s,
            cur_label_gt_coords, cur_label_gt_ids)

        # Previous frame
        prev_label_gt_indices = [i for i, l in enumerate(prev_gt_labels) if l == label_id]
        prev_label_pred_indices = [i for i, l in enumerate(prev_pred_labels) if l == label_id]
        if not prev_label_gt_indices:
            continue

        prev_label_gt_coords = [prev_gt_coords[i] for i in prev_label_gt_indices]
        prev_label_gt_ids = [prev_gt_ids[i] for i in prev_label_gt_indices]
        prev_label_pred_v = [prev_pred_vecs[i] for i in prev_label_pred_indices]
        prev_label_pred_s = [prev_pred_scores[i] for i in prev_label_pred_indices]

        prev_matches = hungarian_match_preds_to_gt(
            prev_label_pred_v, prev_label_pred_s,
            prev_label_gt_coords, prev_label_gt_ids)

        # --- Step 2.2: Find common GT instances ---
        common_ids = set(cur_matches.keys()) & set(prev_matches.keys())
        if not common_ids:
            continue

        cat_name = ID2CAT[label_id]
        for inst_id in common_ids:
            cur_matched_vec, cur_score = cur_matches[inst_id]
            prev_matched_vec, prev_score = prev_matches[inst_id]

            # Skip if both miss
            if cur_score < BOTH_MISS_THRESHOLD and prev_score < BOTH_MISS_THRESHOLD:
                continue

            pc = compute_presence(cur_score, prev_score)

            if cur_matched_vec is not None and prev_matched_vec is not None:
                # Transform prev prediction to cur frame
                aligned_prev = transform_points_between_frames(
                    prev_matched_vec, prev_pose, cur_pose)
                # Filter to ROI
                valid_mask = (
                    (aligned_prev[:, 0] >= x_range[0]) &
                    (aligned_prev[:, 0] <= x_range[1]) &
                    (aligned_prev[:, 1] >= y_range[0]) &
                    (aligned_prev[:, 1] <= y_range[1])
                )
                filtered_prev = aligned_prev[valid_mask]
                ls = compute_localization(np.array(cur_matched_vec), filtered_prev)
                # Shape: check FILTERED length, compute on UNFILTERED
                if len(np.array(cur_matched_vec)) < 3 or len(filtered_prev) < 3:
                    ss = 1.0
                else:
                    ss = compute_shape(np.array(cur_matched_vec), aligned_prev)
            else:
                ls = 0.0
                ss = 0.0

            if ls == -1.0:
                ls = 0.0
            if ss == -1.0:
                ss = 0.0

            si = pc * np.clip(ls * loc_weight + ss * (1 - loc_weight), 0.0, 1.0)
            pair_results.append((cat_name, pc, ls, ss, si))

    return pair_results


def compute_mAS(results, gt_instances_all, frame_infos, roi_size=ROI_SIZE,
                interval=INTERVAL, loc_weight=LOC_WEIGHT, n_workers=N_WORKERS):
    """
    Compute mAS (mean Average Stability) with multiprocessing.

    Returns dict with per-class PC, LS, SS, AS and means.
    """
    print(f'\nComputing mAS (interval={interval}, workers={n_workers})...')
    start = time()

    pairs = build_frame_pairs(frame_infos, interval)
    print(f'  {len(pairs)} frame pairs built')

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
            cur_pose, prev_pose, roi_size, loc_weight
        ))

    print(f'  {len(worker_args)} valid pairs to process')

    if n_workers > 0 and len(worker_args) > 0:
        pool = Pool(n_workers)
        all_pair_results = pool.map(_process_single_pair, worker_args)
        pool.close()
        pool.join()
    else:
        all_pair_results = [_process_single_pair(a) for a in worker_args]

    # Aggregate
    class_metrics = defaultdict(lambda: {'presence': [], 'localization': [], 'shape': [], 'si': []})
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

    mean_keys = ['PC', 'LS', 'SS', 'AS']
    for k in mean_keys:
        vals = [mas_result[c][k] for c in ['divider', 'ped_crossing', 'boundary']]
        mas_result[f'm{k}'] = float(np.mean(vals))

    print(f'mAS computation done in {time() - start:.2f}s')
    return mas_result


# ============================================================
# Section 5: Main
# ============================================================

def print_results(map_result, mas_result, thresholds):
    """Print mAP and mAS tables."""
    # mAP table
    print('\n' + '=' * 60)
    print('  mAP Results')
    print('=' * 60)
    table = prettytable.PrettyTable(
        ['category', 'num_preds', 'num_gts'] +
        [f'AP@{t}' for t in thresholds] + ['AP'])
    for label_id in sorted(ID2CAT.keys()):
        cat = ID2CAT[label_id]
        if cat not in map_result:
            continue
        r = map_result[cat]
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
    print('\n' + '=' * 60)
    print('  mAS Results')
    print('=' * 60)
    table2 = prettytable.PrettyTable(['category', 'count', 'PC', 'LS', 'SS', 'AS'])
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
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='Standalone mAP + mAS evaluator')
    parser.add_argument('--submission', required=True, help='Path to submission_vector.json')
    parser.add_argument('--dataroot', required=True, help='NuScenes dataset root')
    parser.add_argument('--ann-file', required=True, help='Annotation pkl file')
    parser.add_argument('--roi-size', nargs=2, type=int, default=list(ROI_SIZE),
                        help='ROI size (default: 60 30)')
    parser.add_argument('--interval', type=int, default=INTERVAL,
                        help='mAS frame interval (default: 2)')
    parser.add_argument('--loc-weight', type=float, default=LOC_WEIGHT,
                        help='Localization weight in SI formula (default: 0.3)')
    parser.add_argument('--n-workers', type=int, default=N_WORKERS,
                        help='Parallel workers (default: 16)')
    args = parser.parse_args()

    roi_size = tuple(args.roi_size)
    if roi_size == (100, 50):
        thresholds = [1.0, 1.5, 2.0]
    else:
        thresholds = THRESHOLDS

    # Load submission
    print(f'Loading submission from {args.submission}')
    submission = mmcv.load(args.submission)
    results = submission['results']

    # Load GT
    gts, gt_instances_all, frame_infos = load_all_gt(args.ann_file, args.dataroot, roi_size)

    # Compute mAP
    map_result = compute_mAP(results, gts, thresholds, n_workers=args.n_workers)

    # Compute mAS
    mas_result = compute_mAS(results, gt_instances_all, frame_infos,
                             roi_size=roi_size, interval=args.interval,
                             loc_weight=args.loc_weight, n_workers=args.n_workers)

    # Print results
    print_results(map_result, mas_result, thresholds)

    return map_result, mas_result


if __name__ == '__main__':
    main()
