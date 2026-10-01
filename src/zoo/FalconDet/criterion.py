"""
reference:
https://github.com/facebookresearch/detr/blob/main/models/detr.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision
from .utils import box_iou_3d, generalized_box_iou_3d, box_cxcyczwhd_to_xyzxyz, global_points_to_relative
from ...misc.dist import get_world_size, is_dist_available_and_initialized
from ...core import register


class GaussHeatmap3DGenerator:
    def __init__(self, volume_size=(5, 512, 512), sigma_ratio=1.2):
        """
        Args:
            volume_size: (D, H, W) spatial size of the 3D volume.
            sigma_ratio: Scale factor for Gaussian kernel width.
        """
        self.volume_size = volume_size
        self.sigma_ratio = sigma_ratio

    def __call__(self, targets):
        """
        Args:
            targets: list of dicts
        Returns:
            heatmap: torch.Tensor, shape (1, D, H, W)
        """
        D, H, W = self.volume_size
        batch_size = len(targets)
        heatmaps = torch.zeros((batch_size, D, H, W), dtype=torch.float32)

        for i, sample in enumerate(targets):
            bboxes = sample['bboxes']  # [N, 6]
            for box in bboxes:
                cx, cy, cz, w, h, d = box
                cx, w = int(cx * W), int(w * W)
                cy, h = int(cy * H), int(h * H)
                cz, d = int(cz * D), int(d * D)

                # Sigma values per axis
                sigma_x = max(w * self.sigma_ratio, 1.0)
                sigma_y = max(h * self.sigma_ratio, 1.0)
                sigma_z = max(d * self.sigma_ratio, 1.0)

                kernel = self._gaussian_kernel_3d(sigma_x, sigma_y, sigma_z)
                if kernel.numel() == 0:
                    continue

                k_d, k_h, k_w = kernel.shape
                radius_x = k_w // 2
                radius_y = k_h // 2
                radius_z = k_d // 2

                # Clip placement to valid ranges
                x_start = max(cx - radius_x, 0)
                y_start = max(cy - radius_y, 0)
                z_start = max(cz - radius_z, 0)
                x_end = min(cx + radius_x + 1, W)
                y_end = min(cy + radius_y + 1, H)
                z_end = min(cz + radius_z + 1, D)

                # Kernel crop region
                k_start_x = max(radius_x - (cx - x_start), 0)
                k_start_y = max(radius_y - (cy - y_start), 0)
                k_start_z = max(radius_z - (cz - z_start), 0)
                k_end_x = k_w - max((cx + radius_x + 1) - x_end, 0)
                k_end_y = k_h - max((cy + radius_y + 1) - y_end, 0)
                k_end_z = k_d - max((cz + radius_z + 1) - z_end, 0)

                kernel_cropped = kernel[k_start_z:k_end_z, k_start_y:k_end_y, k_start_x:k_end_x]

                if kernel_cropped.numel() == 0:
                    continue

                patch_d = z_end - z_start
                patch_h = y_end - y_start
                patch_w = x_end - x_start

                # Ensure size alignment
                if kernel_cropped.shape != (patch_d, patch_h, patch_w):
                    kernel_cropped = kernel_cropped[:patch_d, :patch_h, :patch_w]

                # Add kernel to heatmap
                heatmaps[i, z_start:z_end, y_start:y_end, x_start:x_end] += kernel_cropped

        # Normalize
        if heatmaps.max() > 0:
            heatmaps = heatmaps / heatmaps.max()

        return heatmaps  # shape: (B, D, H, W)

    @staticmethod
    def _gaussian_kernel_3d(sigma_x, sigma_y, sigma_z):
        """
        Generate a 3D Gaussian kernel.
        """
        sigma_x = max(sigma_x, 0.1)
        sigma_y = max(sigma_y, 0.1)
        sigma_z = max(sigma_z, 0.1)

        kernel_w = int(6 * sigma_x) + 1
        kernel_h = int(6 * sigma_y) + 1
        kernel_d = int(6 * sigma_z) + 1

        # Ensure odd kernel size
        if kernel_w % 2 == 0: kernel_w += 1
        if kernel_h % 2 == 0: kernel_h += 1
        if kernel_d % 2 == 0: kernel_d += 1

        x = torch.arange(kernel_w, dtype=torch.float32) - (kernel_w // 2)
        y = torch.arange(kernel_h, dtype=torch.float32) - (kernel_h // 2)
        z = torch.arange(kernel_d, dtype=torch.float32) - (kernel_d // 2)

        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

        kernel = torch.exp(-(
                (xx ** 2) / (2 * sigma_x ** 2) +
                (yy ** 2) / (2 * sigma_y ** 2) +
                (zz ** 2) / (2 * sigma_z ** 2)
        ))

        # Normalize
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum

        return kernel


@register()
class FalconDetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self,
                 matcher,
                 weight_dict,
                 losses,
                 heatmap_size,
                 alpha=0.2,
                 gamma=2.0,
                 eos_coef=1e-4,
                 point_positive_weight=3.0,
                 point_negative_weight=1.0,
                 point_xy_weight=2.0,
                 point_t_weight=1.0,
                 point_score_alpha=0.25,
                 point_score_gamma=2.0,
                 point_score_use_quality=True,
                 point_score_positive_floor=0.0,
                 point_score_quality_power=1.0,
                 point_score_use_tube_quality=True,
                 point_tube_quality_floor=0.25,
                 point_tube_quality_blend=1.0,
                 point_quality_xy_scale=0.02,
                 point_quality_t_scale=0.05,
                 point_gt_loss_weight=0.4,
                 point_jitter_loss_weight=0.3,
                 point_pred_loss_weight=0.2,
                 point_background_loss_weight=0.1,
                 point_background_topk=3,
                 point_gt_warmup_epochs=0,
                 aux_loss_weight=0.5,
                 enc_aux_loss_weight=0.5,
                 dn_loss_weight=0.25,
                 num_classes=80
                 ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.heatmap_size = heatmap_size

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma
        self.point_positive_weight = point_positive_weight
        self.point_negative_weight = point_negative_weight
        self.point_xy_weight = point_xy_weight
        self.point_t_weight = point_t_weight
        self.point_score_alpha = point_score_alpha
        self.point_score_gamma = point_score_gamma
        self.point_score_use_quality = point_score_use_quality
        self.point_score_positive_floor = point_score_positive_floor
        self.point_score_quality_power = point_score_quality_power
        self.point_score_use_tube_quality = point_score_use_tube_quality
        self.point_tube_quality_floor = point_tube_quality_floor
        self.point_tube_quality_blend = point_tube_quality_blend
        self.point_quality_xy_scale = max(point_quality_xy_scale, 1e-6)
        self.point_quality_t_scale = max(point_quality_t_scale, 1e-6)
        self.point_gt_loss_weight = float(point_gt_loss_weight)
        self.point_jitter_loss_weight = float(point_jitter_loss_weight)
        self.point_pred_loss_weight = float(point_pred_loss_weight)
        self.point_background_loss_weight = float(point_background_loss_weight)
        self.point_background_topk = max(int(point_background_topk), 0)
        self.point_gt_warmup_epochs = max(int(point_gt_warmup_epochs), 0)
        self.aux_loss_weight = float(aux_loss_weight)
        self.enc_aux_loss_weight = float(enc_aux_loss_weight)
        self.dn_loss_weight = float(dn_loss_weight)

        self.vox_gen = GaussHeatmap3DGenerator(volume_size=heatmap_size, sigma_ratio=0.6)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["bboxes_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["bboxes_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(torch.float32)
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx].float()
        target_boxes = torch.cat([t['bboxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).float()
        ious, _ = box_iou_3d(box_cxcyczwhd_to_xyzxyz(src_boxes), box_cxcyczwhd_to_xyzxyz(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits'].float()
        target_classes_o = torch.cat([t["bboxes_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["bboxes_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx].float()
        target_boxes = torch.cat([t['bboxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).float()

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou_3d( \
            box_cxcyczwhd_to_xyzxyz(src_boxes), box_cxcyczwhd_to_xyzxyz(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _point_source_loss(self, source, targets, assignments):
        device = targets[0]['bboxes'].device
        point_loss = torch.zeros((), device=device)
        score_loss = torch.zeros((), device=device)
        positive_count = 0
        query_count = 0

        for batch_index, target in enumerate(targets):
            pred_points = source['pred_points'][batch_index]
            pred_scores = source['pred_point_scores'][batch_index]
            pred_logits = source.get(
                'pred_point_logits', [None] * len(targets)
            )[batch_index]
            source_boxes = source['boxes'][batch_index]
            if pred_points is None or source_boxes is None:
                continue

            for source_index, target_index in assignments[batch_index]:
                pred_pts = pred_points[source_index].float()
                pred_score = pred_scores[source_index].float().clamp(1e-4, 1 - 1e-4)
                pred_logit = (
                    None if pred_logits is None else pred_logits[source_index].float()
                )
                point_box = source_boxes[source_index].float()
                gt_points_rel = pred_pts.new_empty((0, 3))
                tube_quality = None

                if target_index is not None:
                    gt_box = target['bboxes'][target_index].float()
                    group_id = target['bboxes_group_ids'][target_index]
                    point_mask = target['points_group_ids'] == group_id
                    if point_mask.any():
                        gt_points_rel = global_points_to_relative(
                            target['points'][point_mask],
                            point_box.unsqueeze(0),
                        ).clamp(0.0, 1.0)
                    if self.point_score_use_tube_quality:
                        with torch.no_grad():
                            tube_quality = box_iou_3d(
                                box_cxcyczwhd_to_xyzxyz(point_box.unsqueeze(0)),
                                box_cxcyczwhd_to_xyzxyz(gt_box.unsqueeze(0)),
                            )[0].reshape(()).clamp(
                                min=self.point_tube_quality_floor,
                                max=1.0,
                            )

                target_points, target_scores = self._build_point_targets(
                    gt_points_rel,
                    num_queries=pred_pts.shape[0],
                    device=pred_pts.device,
                    dtype=pred_pts.dtype,
                )
                positive_mask = target_scores > 0
                score_targets = target_scores.clone()

                if positive_mask.any():
                    point_loss += self.point_xy_weight * F.smooth_l1_loss(
                        pred_pts[positive_mask, :2],
                        target_points[positive_mask, :2],
                        reduction='sum',
                        beta=0.02,
                    )
                    point_loss += self.point_t_weight * F.smooth_l1_loss(
                        pred_pts[positive_mask, 2],
                        target_points[positive_mask, 2],
                        reduction='sum',
                        beta=0.01,
                    )
                    positive_count += int(positive_mask.sum().item())

                    if self.point_score_use_quality:
                        with torch.no_grad():
                            xy_err = torch.norm(
                                pred_pts[positive_mask, :2]
                                - target_points[positive_mask, :2],
                                dim=-1,
                            )
                            t_err = torch.abs(
                                pred_pts[positive_mask, 2]
                                - target_points[positive_mask, 2]
                            )
                            quality = torch.exp(
                                -0.5 * (
                                    (xy_err / self.point_quality_xy_scale) ** 2
                                    + (t_err / self.point_quality_t_scale) ** 2
                                )
                            ).clamp_(0.0, 1.0)
                            if self.point_score_quality_power != 1.0:
                                quality = quality.pow(self.point_score_quality_power)
                            if self.point_score_positive_floor > 0:
                                quality = self.point_score_positive_floor + (
                                    1.0 - self.point_score_positive_floor
                                ) * quality
                        score_targets[positive_mask] = quality
                    if tube_quality is not None:
                        tube_scale = (
                            (1.0 - self.point_tube_quality_blend)
                            + self.point_tube_quality_blend * tube_quality
                        )
                        score_targets[positive_mask] *= tube_scale
                    if self.point_score_positive_floor > 0:
                        score_targets[positive_mask] = score_targets[
                            positive_mask
                        ].clamp(
                            min=self.point_score_positive_floor,
                            max=1.0,
                        )

                score_weights = torch.full_like(
                    pred_score, self.point_negative_weight
                )
                score_weights[positive_mask] = self.point_positive_weight
                if pred_logit is not None:
                    pred_prob = pred_logit.sigmoid()
                    alpha_factor = torch.full_like(
                        pred_prob, 1.0 - self.point_score_alpha
                    )
                    alpha_factor[positive_mask] = self.point_score_alpha
                    pt = (
                        score_targets * pred_prob
                        + (1.0 - score_targets) * (1.0 - pred_prob)
                    )
                    focal_weight = alpha_factor * (1.0 - pt).pow(
                        self.point_score_gamma
                    )
                    score_loss += (
                        F.binary_cross_entropy_with_logits(
                            pred_logit,
                            score_targets,
                            reduction='none',
                        )
                        * score_weights
                        * focal_weight
                    ).sum()
                else:
                    score_loss += F.binary_cross_entropy(
                        pred_score,
                        score_targets,
                        weight=score_weights,
                        reduction='sum',
                    )
                query_count += pred_score.numel()

        return (
            point_loss / max(positive_count, 1),
            score_loss / max(query_count, 1),
        )

    def loss_points(self, outputs, targets, indices, num_boxes, epoch=0):
        """Supervise matched predicted tubes and hard background tube queries."""
        assert 'pred_points' in outputs
        assert 'pred_point_scores' in outputs

        pred_source = {
            'pred_points': outputs['pred_points'],
            'pred_point_scores': outputs['pred_point_scores'],
            'pred_point_logits': outputs.get(
                'pred_point_logits', [None] * len(targets)
            ),
            'boxes': outputs.get(
                'point_pred_boxes',
                [boxes for boxes in outputs['pred_boxes']],
            ),
        }
        positive_assignments = []
        background_assignments = []
        for batch_index, (source_indices, target_indices) in enumerate(indices):
            source_indices = source_indices.to(outputs['pred_boxes'].device)
            target_indices = target_indices.to(outputs['pred_boxes'].device)
            positive_assignments.append([
                (int(source_index), int(target_index))
                for source_index, target_index in zip(
                    source_indices.tolist(), target_indices.tolist()
                )
            ])

            matched = set(source_indices.tolist())
            unmatched = [
                query_index
                for query_index in range(outputs['pred_boxes'].shape[1])
                if query_index not in matched
            ]
            if self.point_background_topk > 0 and unmatched:
                tube_scores = outputs['pred_logits'][batch_index].sigmoid().max(-1).values
                unmatched_tensor = torch.as_tensor(
                    unmatched, device=tube_scores.device, dtype=torch.long
                )
                count = min(self.point_background_topk, len(unmatched))
                selected = unmatched_tensor[
                    torch.topk(tube_scores[unmatched_tensor], count).indices
                ].tolist()
            else:
                selected = unmatched
            background_assignments.append(
                [(int(source_index), None) for source_index in selected]
            )

        zero = outputs['pred_boxes'].sum() * 0.0
        pred_point_loss, pred_score_loss = self._point_source_loss(
            pred_source, targets, positive_assignments
        )
        background_point_loss, background_score_loss = self._point_source_loss(
            pred_source, targets, background_assignments
        )

        gt_point_loss = gt_score_loss = zero
        gt_source = outputs.get('point_gt_outputs')
        if gt_source is not None:
            gt_assignments = [
                [(target_index, target_index) for target_index in range(len(target['bboxes']))]
                for target in targets
            ]
            gt_point_loss, gt_score_loss = self._point_source_loss(
                gt_source, targets, gt_assignments
            )

        jitter_point_loss = jitter_score_loss = zero
        jitter_source = outputs.get('point_jitter_outputs')
        if jitter_source is not None:
            jitter_assignments = [
                [(target_index, target_index) for target_index in range(len(target['bboxes']))]
                for target in targets
            ]
            jitter_point_loss, jitter_score_loss = self._point_source_loss(
                jitter_source, targets, jitter_assignments
            )

        warmup = int(epoch) < self.point_gt_warmup_epochs
        pred_weight = 0.0 if warmup else self.point_pred_loss_weight
        jitter_weight = 0.0 if warmup else self.point_jitter_loss_weight
        gt_weight = (
            1.0
            if warmup and gt_source is not None
            else self.point_gt_loss_weight
        )

        return {
            'loss_points': (
                gt_weight * gt_point_loss
                + jitter_weight * jitter_point_loss
                + pred_weight * pred_point_loss
                + self.point_background_loss_weight * background_point_loss
            ),
            'loss_point_scores': (
                gt_weight * gt_score_loss
                + jitter_weight * jitter_score_loss
                + pred_weight * pred_score_loss
                + self.point_background_loss_weight * background_score_loss
            ),
        }

    @staticmethod
    def _build_point_targets(gt_points_rel, num_queries, device, dtype):
        target_points = torch.zeros((num_queries, 3), device=device, dtype=dtype)
        target_scores = torch.zeros((num_queries,), device=device, dtype=dtype)

        if num_queries <= 0:
            return target_points, target_scores

        centers = (torch.arange(num_queries, device=device, dtype=dtype) + 0.5) / num_queries
        target_points[:, 2] = centers

        if gt_points_rel.numel() == 0:
            return target_points, target_scores

        temporal_dist = torch.abs(gt_points_rel[:, 2:3] - centers.unsqueeze(0))
        query_indices = temporal_dist.argmin(dim=1)
        best_per_query = {}

        for point_idx, query_idx in enumerate(query_indices.tolist()):
            dist = temporal_dist[point_idx, query_idx].item()
            if query_idx not in best_per_query or dist < best_per_query[query_idx][0]:
                best_per_query[query_idx] = (dist, gt_points_rel[point_idx])

        for query_idx, (_, point) in best_per_query.items():
            target_points[query_idx] = point
            target_scores[query_idx] = 1.0

        return target_points, target_scores

    def loss_vox_density(self, outputs, targets, indices, num_boxes, log=True):
        pred_vox_density = outputs['vox_density']
        if pred_vox_density is None:
            return {}

        true_vox_density = self.vox_gen(targets)

        # Optional visualization hooks removed for delivery build.

        pred_vox_density = pred_vox_density.float()
        true_vox_density = true_vox_density.to(pred_vox_density.device).float()
        true_vox_density_2d = F.max_pool3d(true_vox_density.unsqueeze(1), kernel_size=(true_vox_density.shape[1], 1, 1))

        loss_vox_density = F.mse_loss(pred_vox_density, true_vox_density_2d.squeeze(1))

        return {'loss_vox_density': loss_vox_density}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'points': self.loss_points,
            'cardinality': self.loss_cardinality,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["bboxes_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            loss_kwargs = (
                {'epoch': kwargs.get('epoch', 0)}
                if loss == 'points'
                else {}
            )
            l_dict = self.get_loss(
                loss, outputs, targets, indices, num_boxes, **loss_kwargs
            )
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if 'vox_density' in outputs:
            vox_density_loss = self.loss_vox_density(outputs, targets, indices, num_boxes)
            losses.update(vox_density_loss)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)['indices']
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    elif loss == 'points':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] * self.aux_loss_weight
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_aux_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_aux_outputs']):
                indices = self.matcher(enc_outputs, targets)['indices']
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    elif loss == 'points':
                        continue

                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] * self.enc_aux_loss_weight
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    elif loss == 'points':
                        continue

                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **kwargs)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] * self.dn_loss_weight
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['bboxes_labels']) for t in targets]
        device = targets[0]['bboxes_labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                                         torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
