import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .utils import box_cxcyczwhd_to_xyzxyz, relative_points_to_global


@register()
class FalconDetPostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'point_score_power',
        'tube_score_power',
        'point_tube_score_blend',
    ]

    def __init__(
            self,
            num_classes=80,
            use_focal_loss=True,
            num_top_queries=300,
            point_score_power=1.0,
            tube_score_power=0.5,
            point_tube_score_blend=1.0,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.point_score_power = point_score_power
        self.tube_score_power = tube_score_power
        self.point_tube_score_blend = point_tube_score_blend

    def extra_repr(self) -> str:
        return (
            f'use_focal_loss={self.use_focal_loss}, '
            f'num_classes={self.num_classes}, '
            f'num_top_queries={self.num_top_queries}, '
            f'point_score_power={self.point_score_power}, '
            f'tube_score_power={self.tube_score_power}, '
            f'point_tube_score_blend={self.point_tube_score_blend}'
        )

    def forward(self, outputs):
        logits = outputs['pred_logits'].detach().cpu()
        boxes = outputs['pred_boxes'].detach().cpu()
        points = outputs.get('pred_points', None)
        point_scores = outputs.get('pred_point_scores', None)
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu()
        if isinstance(point_scores, torch.Tensor):
            point_scores = point_scores.detach().cpu()

        # cxcyczwhd -> xyzxyz
        bbox_pred = box_cxcyczwhd_to_xyzxyz(boxes)

        # ---------- Level-1: tube selection ----------
        if self.use_focal_loss:
            scores = logits.sigmoid()
            top_k = min(self.num_top_queries, scores.shape[1] * self.num_classes)
            scores, index = torch.topk(
                scores.flatten(1),
                top_k,
                dim=-1
            )
            labels = index % self.num_classes
            index = index // self.num_classes
        else:
            scores = F.softmax(logits, dim=-1)[..., :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = labels.gather(1, index)
            else:
                index = torch.arange(scores.shape[1]).unsqueeze(0).expand(
                    scores.shape[0], -1
                )

        # ---------- Packaging ----------
        results = []

        for b in range(bbox_pred.shape[0]):
            # --------- tubes ----------
            bboxes = bbox_pred[b].gather(
                0,
                index[b].unsqueeze(-1).repeat(1, 6)
            )
            bboxes_labels = labels[b]
            bboxes_scores = scores[b]

            num_boxes = bboxes.shape[0]
            bboxes_group_ids = torch.arange(num_boxes)

            # --------- points (flatten) ----------
            all_points = []
            all_points_scores = []
            all_points_labels = []
            all_points_group_ids = []

            if (
                points is not None
                and point_scores is not None
                and points[b] is not None
            ):
                for gid in range(num_boxes):
                    query_idx = index[b][gid]
                    pts_rel = points[b][query_idx]  # [T, 3]
                    tube_box = bboxes[gid]  # [6] (xyzxyz)
                    pts = relative_points_to_global(
                        pts_rel,
                        tube_box
                    )
                    raw_point_scores = point_scores[b][query_idx].clamp_min(1e-6)
                    tube_quality = bboxes_scores[gid].clamp_min(1e-6)
                    tube_scale = (
                        (1.0 - self.point_tube_score_blend)
                        + self.point_tube_score_blend * tube_quality.pow(self.tube_score_power)
                    )
                    tube_point_scores = (
                        raw_point_scores.pow(self.point_score_power)
                        * tube_scale
                    )

                    all_points.append(pts)
                    all_points_scores.append(tube_point_scores)
                    all_points_labels.append(torch.full((pts.shape[0],), bboxes_labels[gid]))
                    all_points_group_ids.append(torch.full((pts.shape[0],), gid))

                all_points = torch.cat(all_points, dim=0)
                all_points_scores = torch.cat(all_points_scores, dim=0)
                all_points_labels = torch.cat(all_points_labels, dim=0)
                all_points_group_ids = torch.cat(all_points_group_ids, dim=0)
            else:
                all_points = torch.empty((0, 3))
                all_points_scores = torch.empty((0,))
                all_points_labels = torch.empty((0,), dtype=torch.long)
                all_points_group_ids = torch.empty((0,), dtype=torch.long)

            results.append({
                # tube-level
                "bboxes": bboxes.clamp(0, 1),
                "bboxes_labels": bboxes_labels,
                "bboxes_scores": bboxes_scores,
                "bboxes_group_ids": bboxes_group_ids,

                # point-level
                "points": all_points.clamp(0, 1),
                "points_scores": all_points_scores,
                "points_labels": all_points_labels,
                "points_group_ids": all_points_group_ids,
            })

        return results
