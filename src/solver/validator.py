import copy
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from ._utils import box_iou_3d
from .point_metrics import compute_global_point_metrics


class Validator:
    def __init__(
            self,
            gt: List[Dict[str, torch.Tensor]],
            preds: List[Dict[str, torch.Tensor]],
            bbox_conf_thresh=0.4,
            point_conf_thresh=0.4,
            iou_thresh=0.5,
            point_spatial_thresh=0.02,
            point_temporal_thresh = 0.05,
            point_num_frames=5,
    ) -> None:
        """
        Format example:
        gt = [{'labels': tensor([0]), 'boxes': tensor([[x1, y1, z1, x2, y2, z2]])}, ...]
        preds = [{'bboxes_labels': tensor([0]), 'boxes': tensor([[x1, y1, z1, x2, y2, z2]]), 'bboxes_scores': tensor([...])}, ...]
        """
        self.gt = gt
        self.preds = preds
        self.bbox_conf_thresh = bbox_conf_thresh
        self.point_conf_thresh = point_conf_thresh
        self.iou_thresh = iou_thresh

        # point matching thresholds
        self.point_spatial_thresh = point_spatial_thresh
        self.point_temporal_thresh = point_temporal_thresh
        self.point_num_frames = point_num_frames

        self.thresholds = np.arange(0.2, 1.0, 0.05)
        self.conf_matrix = None
        self.matched_group_pairs = []  # [(sample_idx, pred_gid, gt_gid), ...]

    def compute_metrics(self, extended=False) -> Dict[str, float]:
        self.matched_group_pairs = []
        filtered_preds = filter_preds(
            copy.deepcopy(self.preds),
            self.bbox_conf_thresh,
            self.point_conf_thresh,
        )

        # tube-level metrics
        metrics = self._compute_main_metrics(filtered_preds)

        gated_metrics = self._compute_point_metrics_tube_gated(filtered_preds)
        metrics.update({f"{key}_tube_gated": value for key, value in gated_metrics.items()})
        metrics.update(
            compute_global_point_metrics(
                self.gt,
                self.preds,
                score_threshold=self.point_conf_thresh,
                spatial_threshold=self.point_spatial_thresh,
                num_frames=self.point_num_frames,
            )
        )

        if not extended:
            metrics.pop("extended_metrics", None)
        return metrics

    def _compute_main_metrics(self, preds):
        (
            self.metrics_per_class,
            self.conf_matrix,
            self.class_to_idx,
        ) = self._compute_metrics_and_confusion_matrix(preds)

        # -------- tube-level aggregation --------
        tps, fps, fns = 0, 0, 0
        ious = []
        extended_metrics = {}

        for key, value in self.metrics_per_class.items():
            tps += value["TPs"]
            fps += value["FPs"]
            fns += value["FNs"]
            ious.extend(value["IoUs"])

            extended_metrics[f"precision_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FPs"])
                if value["TPs"] + value["FPs"] > 0
                else 0
            )
            extended_metrics[f"recall_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FNs"])
                if value["TPs"] + value["FNs"] > 0
                else 0
            )

            extended_metrics[f"iou_{key}"] = np.mean(value["IoUs"])

        precision = tps / (tps + fps) if (tps + fps) > 0 else 0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = np.mean(ious).item() if ious else 0

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "TPs": tps,
            "FPs": fps,
            "FNs": fns,
            "extended_metrics": extended_metrics,
        }

    def _compute_metrics_and_confusion_matrix(self, preds):
        # Initialize per-class metrics
        metrics_per_class = defaultdict(lambda: {"TPs": 0, "FPs": 0, "FNs": 0, "IoUs": []})

        # Collect all class IDs
        all_classes = set()
        for pred in preds:
            all_classes.update(pred["bboxes_labels"].tolist())
        for gt in self.gt:
            all_classes.update(gt["bboxes_labels"].tolist())

        all_classes = sorted(list(all_classes))
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
        n_classes = len(all_classes)
        conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # +1 for background class

        for sample_idx, (pred, gt) in enumerate(zip(preds, self.gt)):
            pred_boxes, pred_labels = pred["bboxes"], pred["bboxes_labels"]
            gt_boxes, gt_labels = gt["bboxes"], gt["bboxes_labels"]

            n_preds, n_gts = len(pred_boxes), len(gt_boxes)

            matched_pred_indices = set()
            matched_gt_indices = set()

            if n_preds > 0 and n_gts > 0:
                ious = box_iou_3d(pred_boxes, gt_boxes) if n_preds > 0 and n_gts > 0 else torch.tensor([])

                # For each pred box, find the gt box with highest IoU
                ious_mask = ious >= self.iou_thresh
                pred_indices, gt_indices = torch.nonzero(ious_mask, as_tuple=True)
                iou_values = ious[pred_indices, gt_indices]

                # Sorting by IoU to match highest scores first
                sorted_indices = torch.argsort(-iou_values)
                pred_indices = pred_indices[sorted_indices]
                gt_indices = gt_indices[sorted_indices]
                iou_values = iou_values[sorted_indices]

                for pred_idx, gt_idx, iou in zip(pred_indices, gt_indices, iou_values):
                    if (
                            pred_idx.item() in matched_pred_indices
                            or gt_idx.item() in matched_gt_indices
                    ):
                        continue
                    matched_pred_indices.add(pred_idx.item())
                    matched_gt_indices.add(gt_idx.item())
                    pred_gid = pred["bboxes_group_ids"][pred_idx].item()
                    gt_gid = gt["bboxes_group_ids"][gt_idx].item()

                    self.matched_group_pairs.append((sample_idx, pred_gid, gt_gid))

                    pred_label = pred_labels[pred_idx].item()
                    gt_label = gt_labels[gt_idx].item()

                    pred_cls_idx = class_to_idx[pred_label]
                    gt_cls_idx = class_to_idx[gt_label]

                    # Update confusion matrix
                    conf_matrix[gt_cls_idx, pred_cls_idx] += 1

                    # Update per-class metrics
                    if pred_label == gt_label:
                        metrics_per_class[gt_label]["TPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(iou.item())
                    else:
                        # Misclassification
                        metrics_per_class[gt_label]["FNs"] += 1
                        metrics_per_class[pred_label]["FPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(0)
                        metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched predictions (False Positives)
            unmatched_pred_indices = set(range(n_preds)) - matched_pred_indices
            for pred_idx in unmatched_pred_indices:
                pred_label = pred_labels[pred_idx].item()
                pred_cls_idx = class_to_idx[pred_label]
                # Update confusion matrix: background row
                conf_matrix[n_classes, pred_cls_idx] += 1
                # Update per-class metrics
                metrics_per_class[pred_label]["FPs"] += 1
                metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched ground truths (False Negatives)
            unmatched_gt_indices = set(range(n_gts)) - matched_gt_indices
            for gt_idx in unmatched_gt_indices:
                gt_label = gt_labels[gt_idx].item()
                gt_cls_idx = class_to_idx[gt_label]
                # Update confusion matrix: background column
                conf_matrix[gt_cls_idx, n_classes] += 1
                # Update per-class metrics
                metrics_per_class[gt_label]["FNs"] += 1
                metrics_per_class[gt_label]["IoUs"].append(0)

        return metrics_per_class, conf_matrix, class_to_idx

    def _compute_point_metrics_tube_gated(self, preds):
        """
        Point evaluation is performed only inside matched tube pairs.
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        errors = []

        # Build GT point index
        gt_points_by_key = defaultdict(list)
        gt_groups_by_sample = defaultdict(set)
        for sample_idx, gt in enumerate(self.gt):
            gids = gt["points_group_ids"].tolist()
            for p, gid in zip(gt["points"], gids):
                key = (sample_idx, int(gid))
                gt_points_by_key[key].append(p)
                gt_groups_by_sample[sample_idx].add(int(gid))

        # Build Pred point index
        pred_points_by_key = defaultdict(list)
        pred_groups_by_sample = defaultdict(set)
        for sample_idx, pred in enumerate(preds):
            gids = pred["points_group_ids"].tolist()
            for p, gid in zip(pred["points"], gids):
                key = (sample_idx, int(gid))
                pred_points_by_key[key].append(p)
                pred_groups_by_sample[sample_idx].add(int(gid))

        # ------------------------------
        # Matched tube pairs
        # ------------------------------
        matched_gt = defaultdict(set)
        matched_pred = defaultdict(set)

        for sample_idx, pred_gid, gt_gid in self.matched_group_pairs:
            matched_pred[sample_idx].add(pred_gid)
            matched_gt[sample_idx].add(gt_gid)

        # ------------------------------
        # 1. Unmatched GT tubes -> FN
        # ------------------------------
        for sample_idx, gt_gids in gt_groups_by_sample.items():
            unmatched_gt = gt_gids - matched_gt.get(sample_idx, set())
            for gid in unmatched_gt:
                total_fn += len(gt_points_by_key.get((sample_idx, gid), []))

        # ------------------------------
        # 2. Unmatched Pred tubes -> FP
        # ------------------------------
        for sample_idx, pred_gids in pred_groups_by_sample.items():
            unmatched_pred = pred_gids - matched_pred.get(sample_idx, set())
            for gid in unmatched_pred:
                total_fp += len(pred_points_by_key.get((sample_idx, gid), []))

        # ------------------------------
        # 3. Matched tubes -> point matching
        # ------------------------------
        for sample_idx, pred_gid, gt_gid in self.matched_group_pairs:
            pred_points = pred_points_by_key.get((sample_idx, pred_gid), [])
            gt_points = gt_points_by_key.get((sample_idx, gt_gid), [])

            if len(pred_points) == 0 and len(gt_points) == 0:
                continue

            if len(pred_points) == 0:
                total_fn += len(gt_points)
                continue

            if len(gt_points) == 0:
                total_fp += len(pred_points)
                continue

            pred_points = torch.stack(pred_points)  # [Np, 3]
            gt_points = torch.stack(gt_points)  # [Ng, 3]

            # ------------------------------
            # Separate spatial / temporal distance
            # ------------------------------
            # spatial: (x, y)
            spatial_dist = torch.cdist(
                pred_points[:, :2], gt_points[:, :2]
            )  # [Np, Ng]

            # temporal: t
            temporal_dist = torch.abs(
                pred_points[:, 2:3] - gt_points[:, 2].unsqueeze(0)
            )  # [Np, Ng]

            valid_mask = (
            (spatial_dist <= self.point_spatial_thresh) &
            (temporal_dist <= self.point_temporal_thresh)
            )

            pred_indices, gt_indices = torch.nonzero(valid_mask, as_tuple=True)

            if len(pred_indices) == 0:
                total_fp += len(pred_points)
                total_fn += len(gt_points)
                continue

            # greedy matching by smallest distance
            pair_dist = spatial_dist[pred_indices, gt_indices]
            order = torch.argsort(pair_dist)

            pred_indices = pred_indices[order]
            gt_indices = gt_indices[order]
            pair_dist = pair_dist[order]

            matched_pred = set()
            matched_gt = set()

            for pi, gi, d in zip(pred_indices, gt_indices, pair_dist):
                pi = pi.item()
                gi = gi.item()
                if pi in matched_pred or gi in matched_gt:
                    continue

                matched_pred.add(pi)
                matched_gt.add(gi)

                total_tp += 1
                errors.append(d.item())

            total_fp += len(pred_points) - len(matched_pred)
            total_fn += len(gt_points) - len(matched_gt)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_error = float(np.mean(errors)) if errors else 0.0

        return {
            "point_precision": precision,
            "point_recall": recall,
            "point_f1": f1,
            "point_TPs": total_tp,
            "point_FPs": total_fp,
            "point_FNs": total_fn,
            "point_mean_error": mean_error,
        }


def filter_preds(preds, bbox_conf_thresh, point_conf_thresh):
    for pred in preds:
        # tube-level
        keep_idxs = pred["bboxes_scores"] >= bbox_conf_thresh
        pred["bboxes"] = pred["bboxes"][keep_idxs]
        pred["bboxes_scores"] = pred["bboxes_scores"][keep_idxs]
        pred["bboxes_labels"] = pred["bboxes_labels"][keep_idxs]
        pred["bboxes_group_ids"] = pred["bboxes_group_ids"][keep_idxs]

        # point-level
        valid_group_ids = pred["bboxes_group_ids"]
        if valid_group_ids.numel() == 0:
            group_keep_mask = torch.zeros_like(
                pred["points_group_ids"], dtype=torch.bool
            )
        else:
            group_keep_mask = torch.isin(pred["points_group_ids"], valid_group_ids)
        score_keep_mask = pred["points_scores"] >= point_conf_thresh
        point_keep_mask = group_keep_mask & score_keep_mask

        pred["points"] = pred["points"][point_keep_mask]
        pred["points_scores"] = pred["points_scores"][point_keep_mask]
        pred["points_labels"] = pred["points_labels"][point_keep_mask]
        pred["points_group_ids"] = pred["points_group_ids"][point_keep_mask]

    return preds
