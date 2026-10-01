import math
import torch
import torch.nn as nn

__all__ = [
    'mod',
    'box_iou_3d',
    'generalized_box_iou_3d',
    'box_cxcyczwhd_to_xyzxyz',
    'box_xyzxyz_to_cxcyczwhd',
    'stabilize_box_cxcyczwhd',
    'global_points_to_relative',
    'relative_points_to_global',
    'get_activation',
    'inverse_sigmoid',
    'bias_init_with_prob',
]


def mod(a, b):
    out = a - a // b * b
    return out


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

    return iou, union


def generalized_box_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Generalized IoU for 3D bounding boxes.

    Boxes are expected in [x1, y1, z1, x2, y2, z2] format.

    Args:
        boxes1 (Tensor[N, 6])
        boxes2 (Tensor[M, 6])

    Returns:
        giou (Tensor[N, M]): pairwise Generalized IoU
    """
    # Validity checks.
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all(), "boxes1 has invalid boxes"
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all(), "boxes2 has invalid boxes"

    # Step 1: compute IoU and union.
    iou, union = box_iou_3d(boxes1, boxes2)

    # Step 2: compute smallest enclosing box.
    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])  # [N, M, 3]
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])  # [N, M, 3]

    whd = (rb - lt).clamp(min=0)
    volume_c = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]  # [N, M]

    # Step 3: compute GIoU.
    giou = iou - (volume_c - union) / volume_c.clamp(min=1e-6)

    return giou


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


def box_xyzxyz_to_cxcyczwhd(x: torch.Tensor) -> torch.Tensor:
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    b = [
        (x0 + x1) / 2,  # cx
        (y0 + y1) / 2,  # cy
        (z0 + z1) / 2,  # cz
        (x1 - x0),  # w
        (y1 - y0),  # h
        (z1 - z0),  # d
    ]
    return torch.stack(b, dim=-1)


def stabilize_box_cxcyczwhd(
        boxes: torch.Tensor,
        min_size_xy: float = 1e-4,
        min_size_t: float = 1e-4,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes

    min_sizes = torch.tensor(
        [min_size_xy, min_size_xy, min_size_t],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    centers = boxes[..., :3]
    sizes = torch.maximum(boxes[..., 3:], min_sizes)
    return torch.cat([centers, sizes], dim=-1)


def global_points_to_relative(p, tube):
    """
    Convert global points to tube-relative coordinates.

    Args:
        p: Tensor with shape (..., 3).
        tube: Tensor with shape (..., 6) in cxcyczwhd.

    Returns:
        Tensor of relative points in [0, 1].
    """
    tube = stabilize_box_cxcyczwhd(tube)
    x0 = tube[..., 0] - tube[..., 3] / 2
    y0 = tube[..., 1] - tube[..., 4] / 2
    t0 = tube[..., 2] - tube[..., 5] / 2

    rel = torch.stack([
        (p[..., 0] - x0) / tube[..., 3],
        (p[..., 1] - y0) / tube[..., 4],
        (p[..., 2] - t0) / tube[..., 5],
    ], dim=-1)

    return rel

def relative_points_to_global(points_rel, boxes_xyzxyz):
    """
    points_rel: [T, 3]  (relative, usually in [0,1])
    boxes_xyzxyz: [6]   (x1, y1, z1, x2, y2, z2)
    """
    x1, y1, z1, x2, y2, z2 = boxes_xyzxyz
    w = (x2 - x1).clamp_min(1e-4)
    h = (y2 - y1).clamp_min(1e-4)
    d = (z2 - z1).clamp_min(1e-4)

    points_global = torch.stack([
        points_rel[:, 0] * w + x1,
        points_rel[:, 1] * h + y1,
        points_rel[:, 2] * d + z1,
    ], dim=-1)

    return points_global


def get_activation(act: str, inplace: bool = True):
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == "silu" or act == "swish":
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "silu":
        m = nn.SiLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError("")
    if hasattr(m, "inplace"):
        m.inplace = inplace

    return m


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


