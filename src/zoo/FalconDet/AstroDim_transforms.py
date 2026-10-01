"""Transform operators for spatiotemporal dim-moving-target detection.

All transforms are implemented as torchvision v2 `Transform` subclasses and
operate on custom `TVTensor` wrappers defined in `_tensors.py`.
"""

import random
import re
import warnings
from collections import Counter
from typing import Any, Dict, Tuple, Union, List
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.functional as F

from ._tensors import *
from ...core import register

__all__ = [
    'RandomFlip3D',
    'Resize3D',
    'RandomResize3D',
    'Crop',
    'RandomValidCrop',
    'RemoveSingletonGroups',
    'GenerateBoundingBoxes3D',
    'Normalize'
]


def _parse_torchvision_version(version: str) -> Tuple[int, int]:
    """Parse the major and minor components of a torchvision version."""
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


_TV_MAJOR, _TV_MINOR = _parse_torchvision_version(torchvision.__version__)
_USE_PRIVATE_V2_HOOKS = (_TV_MAJOR, _TV_MINOR) <= (0, 20)


class CompatTransform(v2.Transform):
    """Bridge torchvision v2 transform hooks across supported versions."""

    if _USE_PRIVATE_V2_HOOKS:
        def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
            return {}

        def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            raise NotImplementedError(
                f"{type(self).__name__} must implement 'transform'"
            )

        def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
            return self.make_params(flat_inputs)

        def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            return self.transform(inpt, params)


@register()
class RandomFlip3D(CompatTransform):
    _transformed_types = (ImageSequence, Points)

    def __init__(self, p_h=0.5, p_v=0.05):
        super().__init__()
        self.p_h = p_h
        self.p_v = p_v

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        do_h = random.random() < self.p_h
        do_v = random.random() < self.p_v
        return {"do_h": do_h, "do_v": do_v}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, ImageSequence):
            return self._transform_image_sequence(inpt, params)
        elif isinstance(inpt, Points):
            return self._transform_points(inpt, params)
        return inpt

    @staticmethod
    def _transform_image_sequence(img_seq: ImageSequence, params: Dict[str, Any]) -> ImageSequence:
        seq = img_seq.to_tensor()
        if params["do_h"]:
            seq = F.hflip(seq)
        if params["do_v"]:
            seq = F.vflip(seq)
        return ImageSequence(seq, canvas_size=img_seq.canvas_size, fps=img_seq.fps)

    @staticmethod
    def _transform_points(pts: Points, params: Dict[str, Any]) -> Points:
        w, h = pts.canvas_size[:2]
        data = pts.to_tensor().clone()
        if pts.form == "xyt":
            x, y, t = data[:, 0], data[:, 1], data[:, 2]
        else:
            x, y, t = data[:, 0], data[:, 1], None
        if params["do_h"]:
            x = w - 1 - x
        if params["do_v"]:
            y = h - 1 - y
        if t is not None:
            return Points(torch.stack([x, y, t], dim=1), form="xyt", canvas_size=pts.canvas_size)
        return Points(torch.stack([x, y], dim=1), form="xy", canvas_size=pts.canvas_size)


@register()
class Resize3D(CompatTransform):
    _transformed_types = (ImageSequence, Points)

    def __init__(self, size):
        super().__init__()

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (tuple, list)):
            if len(size) != 2:
                raise ValueError("If 'size' is a tuple or list, it must contain exactly 2 elements (height, width).")
            if not all(isinstance(s, int) for s in size):
                raise TypeError("Elements of 'size' must be integers.")
            self.size = tuple(size)
        else:
            raise TypeError(f"'size' must be an int, tuple, or list, but got {type(size).__name__}.")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return {'h': self.size[0], 'w': self.size[1]}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        target_h, target_w = params['h'], params['w']

        if isinstance(inpt, ImageSequence):
            data = F.resize(inpt.to_tensor(), [target_h, target_w])
            return ImageSequence(data, canvas_size=(target_w, target_h) + inpt.canvas_size[2:], fps=inpt.fps)

        elif isinstance(inpt, Points):
            w0, h0 = inpt.canvas_size[:2]
            scale_x = target_w / w0
            scale_y = target_h / h0
            d = inpt.to_tensor().clone()

            if inpt.form == "xyt":
                d[:, 0] *= scale_x
                d[:, 1] *= scale_y
                return Points(d, form="xyt", canvas_size=(target_w, target_h) + inpt.canvas_size[2:])
            else:
                d[:, 0] *= scale_x
                d[:, 1] *= scale_y
                return Points(d, form="xy", canvas_size=(target_w, target_h))
        return inpt


@register()
class RandomResize3D(Resize3D):
    """
    Randomly resize 3D inputs (ImageSequence or Points) to a size sampled
    from given ranges for width and height.

    Args:
        w_range: int or tuple/list of 2 ints specifying min/max width
        h_range: int or tuple/list of 2 ints specifying min/max height.
                 If None, h_range = w_range (square resize)
    """

    def __init__(self, w_range: Union[int, Tuple[int, int], list], h_range: Union[int, Tuple[int, int], list] = None):
        if h_range is None:
            h_range = w_range

        self.w_range = self._parse_range(w_range, "w_range")
        self.h_range = self._parse_range(h_range, "h_range")
        super().__init__(size=(0, 0))

    @staticmethod
    def _parse_range(r, name):
        if isinstance(r, int):
            return r, r
        elif isinstance(r, (tuple, list)):
            if len(r) != 2:
                raise ValueError(f"{name} must have exactly 2 elements (min, max)")
            if not all(isinstance(x, int) for x in r):
                raise TypeError(f"Elements of {name} must be integers")
            return tuple(r)
        else:
            raise TypeError(f"{name} must be int, tuple, or list, got {type(r).__name__}")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        target_w = random.randint(self.w_range[0], self.w_range[1])
        target_h = random.randint(self.h_range[0], self.h_range[1])
        return {'h': target_h, 'w': target_w}


@register()
class Crop(CompatTransform):
    """Deterministically crop image sequences and aligned point targets."""

    _transformed_types = (ImageSequence, Points, PointLabels, PointGroupIDs)

    def __init__(self, x0: int, y0: int, crop_size: Tuple[int, int]):
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.crop_size = crop_size

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        pts = next((item for item in flat_inputs if isinstance(item, Points)), None)
        crop_h, crop_w = self.crop_size
        keep_mask = None
        if pts is not None:
            pts_tensor = pts.to_tensor()
            keep_mask = (
                (pts_tensor[:, 0] >= self.x0)
                & (pts_tensor[:, 0] < self.x0 + crop_w)
                & (pts_tensor[:, 1] >= self.y0)
                & (pts_tensor[:, 1] < self.y0 + crop_h)
            )
        return {
            "x0": self.x0,
            "y0": self.y0,
            "w": crop_w,
            "h": crop_h,
            "keep_mask": keep_mask,
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        x0, y0, w, h = params["x0"], params["y0"], params["w"], params["h"]
        keep_mask = params["keep_mask"]

        if isinstance(inpt, ImageSequence):
            cropped = F.crop(inpt.to_tensor(), top=y0, left=x0, height=h, width=w)
            return ImageSequence(
                cropped,
                canvas_size=(w, h) + inpt.canvas_size[2:],
                fps=inpt.fps,
            )
        if isinstance(inpt, Points):
            if keep_mask is None:
                return inpt
            data = inpt.to_tensor()[keep_mask].clone()
            if data.numel() != 0:
                data[:, 0] -= x0
                data[:, 1] -= y0
            return Points(
                data,
                form=inpt.form,
                canvas_size=(w, h) + inpt.canvas_size[2:],
            )
        if isinstance(inpt, PointLabels):
            if keep_mask is None:
                return inpt
            return PointLabels(
                inpt.to_tensor()[keep_mask],
                label_map=inpt.name_to_id or inpt.id_to_name,
            )
        if isinstance(inpt, PointGroupIDs):
            if keep_mask is None:
                return inpt
            return PointGroupIDs(inpt.to_tensor()[keep_mask])
        return inpt


@register()
class RandomValidCrop(CompatTransform):
    """
    Randomly crop the image and points such that
    the cropped region contains at least one complete group of points.
    """

    _transformed_types = (ImageSequence, Points, PointLabels, PointGroupIDs)

    def __init__(
            self,
            crop_size: Tuple[int, int],
            least_target_num=3,
            max_tries: int = 50,
            margin: float = 1.0,
    ):
        """
        Args:
            crop_size: (target_h, target_w) in pixels.
            max_tries: maximum attempts to find valid crop.
            margin: extra margin around selected groups.
        """
        super().__init__()
        self.crop_size = crop_size
        self.least_target_num = least_target_num
        self.max_tries = max_tries
        self.margin = margin

    def _find_points_and_gids(self, flat_inputs: List[Any]) -> Tuple[Any, Any]:
        """Scan flat_inputs to find Points and PointGroupIDs."""
        pts = None
        group_ids = None
        for item in flat_inputs:
            if isinstance(item, Points):
                pts = item
            elif isinstance(item, PointGroupIDs):
                group_ids = item
        return pts, group_ids

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        pts, _ = self._find_points_and_gids(flat_inputs)
        if pts is None:
            raise ValueError("RandomValidCrop requires Points in input.")
        pts_tensor = pts.to_tensor()

        crop_region = self._get_valid_crop(flat_inputs)
        keep_mask = (
                (pts_tensor[:, 0] >= crop_region['x0'])
                & (pts_tensor[:, 0] < crop_region['x0'] + crop_region['w'])
                & (pts_tensor[:, 1] >= crop_region['y0'])
                & (pts_tensor[:, 1] < crop_region['y0'] + crop_region['h'])
        )
        crop_region.update({'keep_mask': keep_mask})
        return crop_region

    def _get_valid_crop(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """
        Find a valid crop region that fully contains at least one group.
        """
        pts, group_ids = self._find_points_and_gids(flat_inputs)

        if pts is None or group_ids is None:
            raise ValueError("RandomValidCrop requires both Points and GroupIDs in input.")

        w, h = pts.canvas_size[:2]
        crop_h, crop_w = self.crop_size

        d = pts.to_tensor()
        gids = group_ids.to_tensor()

        # Compute bounding boxes for each group
        boxes = {}
        for gid in torch.unique(gids):
            mask = gids == gid
            x, y = d[mask, 0], d[mask, 1]
            if x.shape[-1] < self.least_target_num:
                continue
            boxes[int(gid.item())] = [
                x.min().item(),
                y.min().item(),
                x.max().item(),
                y.max().item(),
            ]

        valid_boxes = []
        for gid, (x_min, y_min, x_max, y_max) in boxes.items():
            box_w, box_h = x_max - x_min, y_max - y_min
            if box_w < crop_w and box_h < crop_h:
                valid_boxes.append((gid, x_min, y_min, x_max, y_max))

        if not valid_boxes:
            warnings.warn("No Valid Box for Random Crop!", UserWarning)
            x0 = max(0, (w - crop_w) // 2)
            y0 = max(0, (h - crop_h) // 2)
            return {"x0": x0, "y0": y0, "w": crop_w, "h": crop_h}

        for _ in range(self.max_tries):
            gid, x_min, y_min, x_max, y_max = random.choice(valid_boxes)
            # extend with margin
            x_min = max(0, x_min - self.margin)
            y_min = max(0, y_min - self.margin)
            x_max = min(w, x_max + self.margin)
            y_max = min(h, y_max + self.margin)

            min_x0 = max(0, x_max - crop_w)
            min_y0 = max(0, y_max - crop_h)
            max_x0 = min(x_min, w - crop_w)
            max_y0 = min(y_min, h - crop_h)

            if max_x0 >= min_x0 and max_y0 >= min_y0:
                x0 = random.uniform(min_x0, max_x0)
                y0 = random.uniform(min_y0, max_y0)
                x0 = int(max(0, min(w - crop_w, x0)))
                y0 = int(max(0, min(h - crop_h, y0)))
                return {"x0": x0, "y0": y0, "w": crop_w, "h": crop_h}

        # fallback to top-left crop
        return {"x0": 0, "y0": 0, "w": crop_w, "h": crop_h}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        x0, y0, w, h = params["x0"], params["y0"], params["w"], params["h"]
        keep_mask = params["keep_mask"]

        if isinstance(inpt, ImageSequence):
            seq = inpt.to_tensor()
            cropped = F.crop(seq, top=y0, left=x0, height=h, width=w)
            return ImageSequence(
                cropped,
                canvas_size=(w, h) + inpt.canvas_size[2:],
                fps=inpt.fps,
            )

        elif isinstance(inpt, Points):
            d = inpt.to_tensor()[keep_mask].clone()
            if d.numel() == 0:  # return empty but valid Points
                return Points(d, form=inpt.form, canvas_size=(w, h) + inpt.canvas_size[2:])
            d[:, 0] -= x0
            d[:, 1] -= y0
            return Points(d, form=inpt.form, canvas_size=(w, h) + inpt.canvas_size[2:])

        elif isinstance(inpt, PointLabels):
            d = inpt.to_tensor()[keep_mask]
            return PointLabels(d, label_map=inpt.name_to_id or inpt.id_to_name)

        elif isinstance(inpt, PointGroupIDs):
            d = inpt.to_tensor()[keep_mask]
            return PointGroupIDs(d)

        return inpt


@register()
class RemoveSingletonGroups(CompatTransform):
    """
    Remove samples (points/labels/groupids) whose GroupID occurs only once.

    Example:
        GroupIDs = [9, 9, 9, 9, 11]
        --> group 11 appears once -> removed
        --> keep_mask = [True, True, True, True, False]
    """

    _transformed_types = (Points, PointLabels, PointGroupIDs)

    def __init__(self, min_group_points: int = 1):
        super().__init__()
        self.min_group_points = max(int(min_group_points), 1)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        group_ids = None
        for item in flat_inputs:
            if isinstance(item, PointGroupIDs):
                group_ids = item
                break

        if group_ids is None:
            raise ValueError("RemoveSingletonGroups requires GroupIDs in input.")

        gids = group_ids.to_tensor()
        unique_ids, counts = torch.unique(gids, return_counts=True)
        valid_ids = unique_ids[counts >= self.min_group_points]

        keep_mask = torch.isin(gids, valid_ids)
        return {"keep_mask": keep_mask}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        keep_mask = params["keep_mask"]

        if isinstance(inpt, Points):
            d = inpt.to_tensor()[keep_mask]
            return Points(d, form=inpt.form, canvas_size=inpt.canvas_size)

        elif isinstance(inpt, PointLabels):
            d = inpt.to_tensor()[keep_mask]
            return PointLabels(d, label_map=inpt.name_to_id or inpt.id_to_name)

        elif isinstance(inpt, PointGroupIDs):
            d = inpt.to_tensor()[keep_mask]
            return PointGroupIDs(d)

        return inpt


@register()
class GenerateBoundingBoxes3D(CompatTransform):
    """
    Compute 3D bounding boxes for each unique group_id.

    For each group_id, computes the (cx, cy, cz, w, h, d) box enclosing
    all points belonging to that group across time (z = frame index).

    Updates:
        - target["bboxes"]
        - target["bboxes_labels"]
        - target["bboxes_group_ids"]

    Example:
        points: (x, y, t)
        group_ids: [1, 1, 2, 2, 2]
        labels: [cat, cat, dog, dog, dog]
        --> boxes per group_id: one for 1 (cat), one for 2 (dog)
    """

    _transformed_types = (BoundingBoxes3D, Points, PointLabels, PointGroupIDs, BoxLabels, BoxGroupIDs)

    def __init__(
            self,
            spatial_padding=2.0,
            temporal_padding=0.5,
            min_size_xy=4.0,
            min_size_t=1.0,
    ):
        super().__init__()
        self.spatial_padding = float(spatial_padding)
        self.temporal_padding = float(temporal_padding)
        self.min_size_xy = float(min_size_xy)
        self.min_size_t = float(min_size_t)

    @staticmethod
    def _expand_interval(v_min, v_max, padding, min_size, lower, upper):
        v_min = float(v_min) - padding
        v_max = float(v_max) + padding

        center = 0.5 * (v_min + v_max)
        size = max(v_max - v_min, min_size)
        size = min(size, max(upper - lower, min_size))
        half = 0.5 * size

        v_min = center - half
        v_max = center + half

        if v_min < lower:
            shift = lower - v_min
            v_min += shift
            v_max += shift
        if v_max > upper:
            shift = v_max - upper
            v_min -= shift
            v_max -= shift

        v_min = max(v_min, lower)
        v_max = min(v_max, upper)

        cur_size = v_max - v_min
        if cur_size < min_size:
            deficit = min_size - cur_size
            left_room = v_min - lower
            right_room = upper - v_max
            add_left = min(deficit * 0.5, left_room)
            add_right = min(deficit - add_left, right_room)
            v_min -= add_left
            v_max += add_right

            cur_size = v_max - v_min
            if cur_size < min_size and left_room > add_left:
                extra = min(min_size - cur_size, v_min - lower)
                v_min -= extra
            cur_size = v_max - v_min
            if cur_size < min_size and right_room > add_right:
                extra = min(min_size - cur_size, upper - v_max)
                v_max += extra

        return v_min, v_max

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        points = None
        labels = None
        group_ids = None

        # Collect Points, PointLabels, and PointGroupIDs from the input.
        for item in flat_inputs:
            if isinstance(item, Points):
                points = item
            elif isinstance(item, PointLabels):
                labels = item
            elif isinstance(item, PointGroupIDs):
                group_ids = item

        if points is None or group_ids is None or labels is None:
            raise ValueError("GenerateBoundingBoxes3D requires Points, PointLabels, and PointGroupIDs in input.")

        pts = points.to_tensor()  # shape (N, 3): [x, y, t]
        gids = group_ids.to_tensor()
        lbls = labels.to_tensor()
        canvas_w, canvas_h, canvas_t = points.canvas_size

        unique_gids = torch.unique(gids)
        boxes = []
        box_labels = []
        box_group_ids = []

        for gid in unique_gids.tolist():
            mask = (gids == gid)
            pts_grp = pts[mask]
            lbl_grp = lbls[mask]

            if pts_grp.numel() == 0:
                continue

            x_min, y_min, t_min = pts_grp.min(dim=0).values
            x_max, y_max, t_max = pts_grp.max(dim=0).values

            x_min, x_max = self._expand_interval(
                x_min, x_max, self.spatial_padding, self.min_size_xy, 0.0, max(canvas_w - 1, 0.0)
            )
            y_min, y_max = self._expand_interval(
                y_min, y_max, self.spatial_padding, self.min_size_xy, 0.0, max(canvas_h - 1, 0.0)
            )
            t_min, t_max = self._expand_interval(
                t_min, t_max, self.temporal_padding, self.min_size_t, 0.0, max(canvas_t - 1, 0.0)
            )

            cx = (x_max + x_min) / 2
            cy = (y_max + y_min) / 2
            cz = (t_max + t_min) / 2
            w = x_max - x_min
            h = y_max - y_min
            d = t_max - t_min

            boxes.append(torch.tensor([cx, cy, cz, w, h, d], dtype=torch.float32))

            # Use the most common label for this group.
            most_common_label = Counter(lbl_grp.tolist()).most_common(1)[0][0]
            box_labels.append(most_common_label)
            box_group_ids.append(gid)

        if len(boxes) == 0:
            bboxes = BoundingBoxes3D.empty(format="cxcyczwhd", canvas_size=points.canvas_size)
            bboxes_labels = BoxLabels.empty(label_map=labels.name_to_id or labels.id_to_name)
            bboxes_group_ids = BoxGroupIDs.empty(id_type="int")
        else:
            bboxes = BoundingBoxes3D(torch.stack(boxes), format="cxcyczwhd", canvas_size=points.canvas_size)
            bboxes_labels = BoxLabels(torch.tensor(box_labels, dtype=torch.int64),
                                      label_map=labels.name_to_id or labels.id_to_name)
            bboxes_group_ids = BoxGroupIDs(torch.tensor(box_group_ids, dtype=torch.int64))

        return {
            "bboxes": bboxes,
            "bboxes_labels": bboxes_labels,
            "bboxes_group_ids": bboxes_group_ids,
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, BoundingBoxes3D):
            return params["bboxes"]
        elif isinstance(inpt, BoxLabels):
            return params["bboxes_labels"]
        elif isinstance(inpt, BoxGroupIDs):
            return params["bboxes_group_ids"]
        return inpt


@register()
class Normalize(CompatTransform):
    """
    Normalize ImageSequence, Points, and BoundingBoxes3D based on canvas_size.
    Compatible with torchvision v2 transforms.
    """

    _transformed_types = (ImageSequence, BoundingBoxes3D, Points)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, ImageSequence):
            return self._normalize_image_seq(inpt)
        elif isinstance(inpt, Points):
            return self._normalize_points(inpt)
        elif isinstance(inpt, BoundingBoxes3D):
            return self._normalize_boxes3d(inpt)
        return inpt

    @staticmethod
    def _normalize_image_seq(img_seq):
        """
        Normalize ImageSequence tensor values to [0, 1].
        """
        tensor = img_seq.to_tensor().float()
        if tensor.max() > 1:
            tensor = tensor / 255.0
        return ImageSequence(
            tensor,
            canvas_size=img_seq.canvas_size,
            fps=img_seq.fps
        )

    @staticmethod
    def _normalize_points(pts):
        """
        Normalize Points coordinates using canvas_size.
        Supports both 'xy' and 'xyt' forms.
        """
        data = pts.to_tensor().clone()
        if pts.canvas_size is None:
            return pts  # no normalization if size info is missing

        if pts.form == "xy":
            w, h = pts.canvas_size[:2]
            data[:, 0] /= w
            data[:, 1] /= h
        elif pts.form == "xyt":
            w, h, t = pts.canvas_size
            data[:, 0] /= w
            data[:, 1] /= h
            data[:, 2] /= t

        return Points(data, form=pts.form, canvas_size=pts.canvas_size)

    @staticmethod
    def _normalize_boxes3d(boxes):
        """
        Normalize 3D bounding boxes using canvas_size.
        """
        data = boxes.to_tensor().clone()
        if boxes.canvas_size is None:
            return boxes
        w, h, d = boxes.canvas_size

        data[:, 0] /= w
        data[:, 1] /= h
        data[:, 2] /= d
        data[:, 3] /= w
        data[:, 4] /= h
        data[:, 5] /= d

        return boxes.__class__(data, format=boxes.format, canvas_size=boxes.canvas_size)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
