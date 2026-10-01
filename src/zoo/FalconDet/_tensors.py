import torch
from typing import Optional, Tuple, Union, List, Literal
import numpy as np
from PIL import Image

from torchvision import tv_tensors

__all__ = [
    'ImageSequence',
    'Points',
    'Labels',
    'GroupIDs',
    'BoundingBoxes3D',
    'BoxLabels',
    'PointLabels',
    'BoxGroupIDs',
    'PointGroupIDs'
]

class ImageSequence(tv_tensors.TVTensor):
    """
    A tensor-like wrapper for a sequence of images (e.g., video frames).
    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("num_frames", "canvas_size", "fps")

    def __new__(
            cls,
            data: Union[torch.Tensor, List[Image.Image]],
            *,
            canvas_size: Optional[Tuple[int, int, int]] = None,
            fps: Optional[float] = None,
    ):
        """
        Args:
            data: Either a tensor of shape (C, T, H, W) or list of PIL images.
            canvas_size: (width, height) of frames.
            fps: Optional frame rate.
        """
        if isinstance(data, list):
            # Convert list of PIL images to a stacked tensor
            tensors = []
            for im in data:
                arr = np.array(im)
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=-1)
                tensor = torch.as_tensor(arr).permute(2, 0, 1)  # (C, H, W)
                tensors.append(tensor)
            data = torch.stack(tensors, dim=1)  # (T, C, H, W)
        elif not torch.is_tensor(data):
            data = torch.as_tensor(data)

        return super().__new__(cls, data)

    def __init__(
            self,
            data: Union[torch.Tensor, List[Image.Image]],
            *,
            canvas_size: Optional[Tuple[int, int, int]] = None,
            fps: Optional[float] = None,
    ):
        super().__init__()
        c, t, h, w = self.shape[-4:]
        if canvas_size is None:
            canvas_size = (w, h, t)
        self.canvas_size = canvas_size
        self.fps = fps

    @classmethod
    def empty(cls, *, canvas_size: Optional[Tuple[int, int]] = None, fps: Optional[float] = None):
        """Create an empty ImageSequence."""
        empty_tensor = torch.empty((3, 0, 0, 0))
        return cls(empty_tensor, canvas_size=canvas_size, fps=fps)

    def clone(self, *, memory_format: torch.memory_format | None = None):
        return ImageSequence(
            self.as_subclass(torch.Tensor).clone(),
            canvas_size=self.canvas_size,
            fps=self.fps,
        )

    def to_tensor(self) -> torch.Tensor:
        """Return as torch.Tensor (T, C, H, W)."""
        return self.as_subclass(torch.Tensor)

    def __getitem__(self, idx):
        """Return a single frame as torch.Tensor or ImageSequence."""
        if isinstance(idx, int):
            frame = self[:, idx:idx + 1, :, :]
            return ImageSequence(
                frame.clone(),
                canvas_size=self.canvas_size,
                fps=self.fps,
            )
        return super().__getitem__(idx)

    def __repr__(self):
        return (
            f"ImageSequence(shape={tuple(self.shape)}, "
            f"fps={self.fps}, canvas_size={self.canvas_size})"
        )

    def to(self, device: torch.device, dtype: torch.dtype = None, ** kwargs):
        data = self.as_subclass(torch.Tensor).to(device, dtype=dtype, **kwargs)
        return ImageSequence(
            data,
            canvas_size=self.canvas_size,
            fps=self.fps
        )


class Points(tv_tensors.TVTensor):
    """
    A lightweight wrapper for point coordinates in image (or sequence) space.
    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("form", "canvas_size")

    def __new__(cls, data, *, form: Literal["xy", "xyt"] = "xy",
                canvas_size: Optional[Tuple[int, int]] = None):
        # data shape: (N, 2) or (N, 3) for temporal points
        if not torch.is_tensor(data):
            data = torch.as_tensor(data, dtype=torch.float32)
        return super().__new__(cls, data)

    def __init__(self, data, *, form: Literal["xy", "xyt"] = "xy",
                 canvas_size: Optional[Tuple[int, int] | Tuple[int, int, int]] = None):
        super().__init__()
        self.form = form
        self.canvas_size = canvas_size

    @classmethod
    def empty(cls, *, form: Literal["xy", "xyt"] = "xy", canvas_size: Optional[Tuple[int, int]] = None):
        """Create an empty Points instance."""
        dim = 3 if form == "xyt" else 2
        return cls(torch.empty((0, dim)), form=form, canvas_size=canvas_size)

    def clone(self, *, memory_format: torch.memory_format | None = None):
        return Points(
            self.as_subclass(torch.Tensor).clone(memory_format=memory_format),
            form=self.form,
            canvas_size=self.canvas_size
        )

    def to_tensor(self):
        return self.as_subclass(torch.Tensor)

    def __repr__(self):
        s = f"Points(shape={tuple(self.shape)}, form={self.form}, canvas_size={self.canvas_size})"
        return s

    def to(self, device: torch.device, dtype: torch.dtype = None, ** kwargs):
        data = self.as_subclass(torch.Tensor).to(device, dtype=dtype, **kwargs)
        return Points(
            data,
            form=self.form,
            canvas_size=self.canvas_size,
        )


class Labels(tv_tensors.TVTensor):
    """
    Wrapper for class labels of detected points or objects.
    Supports bidirectional mapping between label IDs and names.
    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("num_labels", "num_classes")

    def __new__(cls, data, *, label_map: Optional[dict] = None):
        # Ensure tensor is int64
        if not torch.is_tensor(data):
            data = torch.as_tensor(data, dtype=torch.int64)
        elif data.dtype != torch.int64:
            data = data.to(torch.int64)
        return super().__new__(cls, data)

    def __init__(self, data, *, label_map: Optional[dict] = None):
        super().__init__()
        self.num_labels = int(self.numel())

        # Handle label_map: can be {name: id} or {id: name}
        self.name_to_id, self.id_to_name = self._process_label_map(label_map)
        self.num_classes = len(self.name_to_id) if self.name_to_id else None

    @classmethod
    def empty(cls, *, label_map: Optional[dict] = None):
        """Create an empty Labels instance."""
        return cls(torch.empty((0,), dtype=torch.int64), label_map=label_map)

    @staticmethod
    def _process_label_map(label_map):
        """Normalize label map into both directions."""
        if label_map is None:
            return {}, {}
        # Detect direction
        if all(isinstance(k, str) and isinstance(v, int) for k, v in label_map.items()):
            name_to_id = label_map
            id_to_name = {v: k for k, v in label_map.items()}
        elif all(isinstance(k, int) and isinstance(v, str) for k, v in label_map.items()):
            id_to_name = label_map
            name_to_id = {v: k for k, v in label_map.items()}
        else:
            raise ValueError(
                "label_map must be either {name: id} or {id: name}"
            )
        return name_to_id, id_to_name

    def clone(self, *, memory_format: torch.memory_format | None = None):
        return Labels(
            self.as_subclass(torch.Tensor).clone(memory_format=memory_format),
            label_map=self.name_to_id or self.id_to_name
        )

    def to_tensor(self):
        return self.as_subclass(torch.Tensor)

    def decode(self):
        """Return a list of human-readable label names."""
        if not self.id_to_name:
            return [int(x.item()) for x in self.flatten()]
        return [self.id_to_name.get(int(x.item()), f"unk_{x.item()}") for x in self.flatten()]

    def encode(self, names: list[str]):
        """Convert a list of label names into tensor form."""
        if not self.name_to_id:
            raise ValueError("Cannot encode names: label_map not provided.")
        ids = [self.name_to_id[n] for n in names]
        return Labels(ids, label_map=self.name_to_id)

    def __repr__(self):
        class_info = f"num_classes={self.num_classes}" if self.num_classes else "num_classes=?"
        mapping_info = (
            f", label_map(keys)={list(self.name_to_id.keys())[:3]}..."
            if self.name_to_id else ""
        )
        return f"Labels(shape={tuple(self.shape)}, {class_info}{mapping_info})"


class GroupIDs(tv_tensors.TVTensor):
    """
    Wrapper for object or track group identifiers.
    Supports either integer or string-based IDs.
    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("num_ids", "id_type")

    def __new__(cls, data):
        # Convert to tensor of strings or ints
        if isinstance(data, (list, tuple)) and len(data) == 0:
            data = torch.empty((0,), dtype=torch.int64)
            id_type = "int"
        elif isinstance(data, (list, tuple)) and isinstance(data[0], str):
            # Encode strings as bytes for tensor storage
            data = np.array(data, dtype='S')
            data = torch.from_numpy(data)
            id_type = "str"
        else:
            data = torch.as_tensor(data)
            id_type = "int"
        obj = super().__new__(cls, data)
        obj.id_type = id_type
        return obj

    def __init__(self, data):
        super().__init__()
        self.num_ids = int(self.shape[0]) if self.ndim > 0 else 1

    def clone(self, *, memory_format: torch.memory_format | None = None):
        return GroupIDs(
            self.as_subclass(torch.Tensor).clone(memory_format=memory_format)
        )

    def to_tensor(self):
        return self.as_subclass(torch.Tensor)

    @classmethod
    def empty(cls, id_type: Literal["int", "str"] = "int"):
        """Create an empty GroupIDs instance."""
        if id_type == "str":
            data = torch.from_numpy(np.array([], dtype='S'))
        else:
            data = torch.empty((0,), dtype=torch.int64)
        return cls(data)

    def __repr__(self):
        s = f"GroupIDs(shape={tuple(self.shape)}, type={self.id_type})"
        return s


class BoundingBoxes3D(tv_tensors.TVTensor):
    """
    A lightweight wrapper for 3D bounding boxes.

    Supports two formats:
        - "cx cy cz w h d": (center_x, center_y, center_z, width, height, depth)
        - "x y z x y z": (x1, y1, z1, x2, y2, z2)

    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("format", "canvas_size")

    def __new__(
            cls,
            data,
            *,
            format: Literal["cxcyczwhd", "xyzxyz"] = "xyzxyz",
            canvas_size: Optional[Tuple[int, int, int]] = None,
    ):
        # data shape: (N, 6)
        if not torch.is_tensor(data):
            data = torch.as_tensor(data, dtype=torch.float32)
        return super().__new__(cls, data)

    def __init__(
            self,
            data,
            *,
            format: Literal["cxcyczwhd", "xyzxyz"] = "xyzxyz",
            canvas_size: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.format = format
        self.canvas_size = canvas_size
        self.num_boxes = int(self.shape[0]) if self.ndim > 0 else 0

    @classmethod
    def empty(
            cls,
            *,
            format: Literal["cxcyczwhd", "xyzxyz"] = "xyzxyz",
            canvas_size: Optional[Tuple[int, int, int]] = None,
    ):
        """Create an empty 3D bounding box container."""
        return cls(torch.empty((0, 6), dtype=torch.float32), format=format, canvas_size=canvas_size)

    def clone(self, *, memory_format: torch.memory_format | None = None):
        return BoundingBoxes3D(
            self.as_subclass(torch.Tensor).clone(memory_format=memory_format),
            format=self.format,
            canvas_size=self.canvas_size,
        )

    def to_tensor(self):
        """Return as torch.Tensor (N, 6)."""
        return self.as_subclass(torch.Tensor)

    def __repr__(self):
        return (
            f"BoundingBoxes3D(shape={tuple(self.shape)}, "
            f"format={self.format}, canvas_size={self.canvas_size})"
        )

    def to(self, device: torch.device, dtype: torch.dtype = None, ** kwargs):
        data = self.as_subclass(torch.Tensor).to(device, dtype=dtype, **kwargs)
        return BoundingBoxes3D(
            data,
            format=self.format,
            canvas_size=self.canvas_size,
        )

class BoxLabels(Labels):
    """Labels corresponding to detected bounding boxes."""

    def __repr__(self):
        base = super().__repr__().replace("Labels", "BoxLabels")
        return base


class PointLabels(Labels):
    """Labels corresponding to detected keypoints or points."""

    def __repr__(self):
        base = super().__repr__().replace("Labels", "PointLabels")
        return base


class BoxGroupIDs(GroupIDs):
    """Group or track IDs corresponding to bounding boxes."""

    def __repr__(self):
        base = super().__repr__().replace("GroupIDs", "BoxGroupIDs")
        return base


class PointGroupIDs(GroupIDs):
    """Group or track IDs corresponding to points."""

    def __repr__(self):
        base = super().__repr__().replace("GroupIDs", "PointGroupIDs")
        return base
