from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


def normalized_time_to_frame(t: float, num_frames: int) -> int:
    """Map a normalized temporal coordinate to a discrete frame index.

    FalconDet stores point time using the dataset's frame-index convention:
    frame ``i`` is encoded as ``i / num_frames``.  Predictions are therefore
    assigned to the nearest frame-index anchor, with exact half-frame ties
    resolved to the earlier frame.  Keep training validation and inference on
    this shared helper so checkpoint selection matches deployment behavior.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    value = min(max(float(t), 0.0), 1.0 - torch.finfo(torch.float32).eps)
    return max(0, min(int(math.ceil(value * num_frames - 0.5)), num_frames - 1))


def point_frame_index(t: float, num_frames: int) -> int:
    """Backward-compatible alias for normalized_time_to_frame."""
    return normalized_time_to_frame(t, num_frames)


def match_points(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    distance_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedily match same-frame points by increasing spatial distance."""
    candidates = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truth):
            if pred["frame"] != gt["frame"]:
                continue
            distance = (
                (float(pred["x"]) - float(gt["x"])) ** 2
                + (float(pred["y"]) - float(gt["y"])) ** 2
            ) ** 0.5
            if distance <= distance_threshold:
                candidates.append((distance, pred_idx, gt_idx))

    matches = []
    matched_pred = set()
    matched_gt = set()
    for distance, pred_idx, gt_idx in sorted(candidates):
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, distance))
    return (
        matches,
        sorted(set(range(len(predictions))) - matched_pred),
        sorted(set(range(len(ground_truth))) - matched_gt),
    )


def _canvas_size(sample: Dict[str, Any], num_frames: int) -> Tuple[float, float, int]:
    canvas_size = sample.get("canvas_size", (1.0, 1.0, num_frames))
    if isinstance(canvas_size, torch.Tensor):
        canvas_size = canvas_size.tolist()
    width = float(canvas_size[0]) if len(canvas_size) > 0 else 1.0
    height = float(canvas_size[1]) if len(canvas_size) > 1 else 1.0
    frames = int(canvas_size[2]) if len(canvas_size) > 2 else num_frames
    return max(width, 1.0), max(height, 1.0), max(frames, 1)


def _point_records(
    points: torch.Tensor,
    width: float,
    height: float,
    num_frames: int,
    scores: torch.Tensor | None = None,
) -> List[Dict[str, Any]]:
    records = []
    for index, point in enumerate(points):
        record = {
            "frame": normalized_time_to_frame(float(point[2]), num_frames),
            "x": float(point[0]) * width,
            "y": float(point[1]) * height,
        }
        if scores is not None:
            record["score"] = float(scores[index])
        records.append(record)
    return records


def _average_precision(tp_flags: Sequence[int], total_gt: int) -> float:
    if total_gt <= 0 or not tp_flags:
        return 0.0
    flags = np.asarray(tp_flags, dtype=np.float64)
    tp = np.cumsum(flags)
    fp = np.cumsum(1.0 - flags)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]))


def _calibration_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    num_bins: int = 10,
) -> Tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    brier = float(np.mean((score_array - label_array) ** 2))
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    for bin_index in range(num_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == num_bins - 1:
            mask = (score_array >= lower) & (score_array <= upper)
        else:
            mask = (score_array >= lower) & (score_array < upper)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(
            float(score_array[mask].mean()) - float(label_array[mask].mean())
        )
    return brier, ece


def compute_global_point_metrics(
    ground_truth: Sequence[Dict[str, torch.Tensor]],
    predictions: Sequence[Dict[str, torch.Tensor]],
    score_threshold: float,
    spatial_threshold: float,
    num_frames: int,
) -> Dict[str, float]:
    """Compute tube-independent, strict same-frame point detection metrics."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    fixed_errors = []
    scored_predictions = []
    gt_by_sample = {}
    distance_threshold_by_sample = {}

    for sample_idx, (gt, pred) in enumerate(zip(ground_truth, predictions)):
        width, height, sample_frames = _canvas_size(gt, num_frames)
        distance_threshold = float(spatial_threshold) * max(width, height)
        distance_threshold_by_sample[sample_idx] = distance_threshold

        gt_records = _point_records(
            gt["points"],
            width=width,
            height=height,
            num_frames=sample_frames,
        )
        pred_records = _point_records(
            pred["points"],
            width=width,
            height=height,
            num_frames=sample_frames,
            scores=pred["points_scores"],
        )
        gt_by_sample[sample_idx] = gt_records

        fixed_predictions = [
            record for record in pred_records if record["score"] >= score_threshold
        ]
        matches, unmatched_pred, unmatched_gt = match_points(
            fixed_predictions,
            gt_records,
            distance_threshold=distance_threshold,
        )
        total_tp += len(matches)
        total_fp += len(unmatched_pred)
        total_fn += len(unmatched_gt)
        fixed_errors.extend(distance for _, _, distance in matches)

        for record in pred_records:
            scored_predictions.append((record["score"], sample_idx, record))

    matched_gt = {sample_idx: set() for sample_idx in gt_by_sample}
    tp_flags = []
    sorted_scores = []
    for score, sample_idx, pred in sorted(
        scored_predictions, reverse=True, key=lambda item: item[0]
    ):
        best_gt_idx = None
        best_distance = None
        for gt_idx, gt in enumerate(gt_by_sample[sample_idx]):
            if gt_idx in matched_gt[sample_idx] or pred["frame"] != gt["frame"]:
                continue
            distance = (
                (float(pred["x"]) - float(gt["x"])) ** 2
                + (float(pred["y"]) - float(gt["y"])) ** 2
            ) ** 0.5
            if distance > distance_threshold_by_sample[sample_idx]:
                continue
            if best_distance is None or distance < best_distance:
                best_gt_idx = gt_idx
                best_distance = distance
        is_tp = int(best_gt_idx is not None)
        if best_gt_idx is not None:
            matched_gt[sample_idx].add(best_gt_idx)
        tp_flags.append(is_tp)
        sorted_scores.append(float(score))

    total_gt = sum(len(points) for points in gt_by_sample.values())
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap = _average_precision(tp_flags, total_gt)
    brier, ece = _calibration_metrics(sorted_scores, tp_flags)

    if fixed_errors:
        error_array = np.asarray(fixed_errors, dtype=np.float64)
        mean_error = float(error_array.mean())
        median_error = float(np.median(error_array))
        p90_error = float(np.percentile(error_array, 90))
        p95_error = float(np.percentile(error_array, 95))
    else:
        mean_error = median_error = p90_error = p95_error = 0.0

    return {
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": f1,
        "point_ap": ap,
        "point_TPs": total_tp,
        "point_FPs": total_fp,
        "point_FNs": total_fn,
        "point_mean_error_px": mean_error,
        "point_median_error_px": median_error,
        "point_p90_error_px": p90_error,
        "point_p95_error_px": p95_error,
        "point_visibility_brier": brier,
        "point_visibility_ece": ece,
    }
