import torch


def box_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Compute IoU (Intersection over Union) and union volumes between 3D boxes.

    Args:
        boxes1 (Tensor[N, 6]): (x1, y1, z1, x2, y2, z2)
        boxes2 (Tensor[M, 6]): (x1, y1, z1, x2, y2, z2)

    Returns:
        iou (Tensor[N, M]): pairwise IoU
        union (Tensor[N, M]): pairwise union volumes
    """
    # Compute per-box volumes.
    whd1 = (boxes1[:, 3:] - boxes1[:, :3]).clamp(min=0)
    whd2 = (boxes2[:, 3:] - boxes2[:, :3]).clamp(min=0)
    volume1 = whd1.prod(dim=1)
    volume2 = whd2.prod(dim=1)

    # Intersection corners.
    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N, M, 3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N, M, 3]

    # Intersection sizes.
    whd = (rb - lt).clamp(min=0)
    inter = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]  # [N, M]

    # Union volume.
    union = volume1[:, None] + volume2 - inter

    # IoU.
    iou = inter / union.clamp(min=1e-6)

    return iou


def box_cxcyczwhd_to_xyzxyz(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    b = [
        (x_c - 0.5 * w),  # x_min
        (y_c - 0.5 * h),  # y_min
        (z_c - 0.5 * d),  # z_min
        (x_c + 0.5 * w),  # x_max
        (y_c + 0.5 * h),  # y_max
        (z_c + 0.5 * d),  # z_max
    ]
    return torch.stack(b, dim=-1)


def scale_boxes(boxes, orig_shape, resized_shape):
    """
    Scale boxes from resized shape back to original shape.

    Args:
        boxes: Tensor with shape [N, 6] in (x1, y1, z1, x2, y2, z2).
        orig_shape: Original shape as [W, H, D].
        resized_shape: Resized shape as [W, H, D].

    Returns:
        Scaled boxes tensor.
    """
    scale_x = orig_shape[0] / resized_shape[0]
    scale_y = orig_shape[1] / resized_shape[1]
    scale_z = orig_shape[2] / resized_shape[2]
    boxes[:, 0] *= scale_x
    boxes[:, 3] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 4] *= scale_y
    boxes[:, 2] *= scale_z
    boxes[:, 5] *= scale_z
    return boxes
