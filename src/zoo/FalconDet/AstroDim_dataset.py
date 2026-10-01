"""Dataset loaders for AstroDim dim-moving-target sequences.

The dataset is organized as multiple subfolders, each containing paired JSON
annotations and PNG frames. Samples are generated as fixed-length frame
sequences with point-level labels and group IDs, then converted to custom
`TVTensor` wrappers for transform compatibility.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from collections import Counter

from torch.utils.data import Dataset, ConcatDataset
try:
    from torchvision.io import ImageReadMode, read_image
except Exception:  # pragma: no cover - torchvision io is optional at runtime
    ImageReadMode = None
    read_image = None

from ...core import register
from ._tensors import *

LABEL_DICT = {"debris": 0}


@register()
class AstroDimDataset:
    """A concatenated dataset over all valid AstroDim sub-datasets.

    This class scans each child directory under `main_folder`, tries to build
    an `AstroDimSubDataset`, and concatenates all successfully loaded subsets.
    """
    __inject__ = ['transforms']

    def __init__(
            self,
            main_folder: str,
            seq_len: int,
            transforms,
            use_torchvision_io: bool = True,
            image_channels: int = 3,
            min_group_points: int = 2,
            include_empty_windows: bool = False,
            negative_window_ratio: float = 0.0,
    ) -> None:
        """
        Create a dataset by concatenating all sub-datasets under a root folder.

        Args:
            main_folder: Root folder containing sequence subfolders.
            seq_len: Number of frames per sample.
            transforms: Transform pipeline applied per sample.
            use_torchvision_io: Whether to use torchvision IO for image loading.
            image_channels: Number of image channels to read (1 or 3).
            min_group_points: Minimum number of points required for at least
                one group in a valid sequence window.
            include_empty_windows: Whether to sample windows without supported
                target annotations.
            negative_window_ratio: Maximum number of negative windows relative
                to the number of positive windows.

        Returns:
            None.
        """
        if image_channels not in (1, 3):
            raise ValueError("image_channels must be 1 or 3")

        self.datasets = []
        for dataset_folder in sorted(os.listdir(main_folder)):
            full_path = os.path.join(main_folder, dataset_folder)
            if os.path.isdir(full_path):
                try:
                    dataset = AstroDimSubDataset(
                        full_path,
                        seq_len,
                        transforms,
                        use_torchvision_io=use_torchvision_io,
                        image_channels=image_channels,
                        min_group_points=min_group_points,
                        include_empty_windows=include_empty_windows,
                        negative_window_ratio=negative_window_ratio,
                    )
                    self.datasets.append(dataset)
                except Exception as e:
                    print(f"Failed to load dataset {dataset_folder}: {e}")

        self.combined_dataset = ConcatDataset(self.datasets)

    def __len__(self) -> int:
        return len(self.combined_dataset)

    def __getitem__(self, idx: int):
        return self.combined_dataset[idx]


class AstroDimSubDataset(Dataset):
    def __init__(
            self,
            data_folder: str,
            seq_len: int,
            transforms,
            use_torchvision_io: bool = True,
            image_channels: int = 3,
            min_group_points: int = 2,
            include_empty_windows: bool = False,
            negative_window_ratio: float = 0.0,
    ) -> None:
        """
        Load a single AstroDim sequence dataset.

        Args:
            data_folder: Sequence folder containing json/ and images/.
            seq_len: Number of frames per sample.
            transforms: Transform pipeline applied per sample.
            use_torchvision_io: Whether to use torchvision IO for image loading.
            image_channels: Number of image channels to read (1 or 3).
            min_group_points: Minimum number of points required for at least
                one group in a valid sequence window.
            include_empty_windows: Whether to sample windows without supported
                target annotations.
            negative_window_ratio: Maximum number of negative windows relative
                to the number of positive windows.

        Returns:
            None.
        """
        super().__init__()
        if image_channels not in (1, 3):
            raise ValueError("image_channels must be 1 or 3")

        self.dataset_folder = data_folder
        self.seq_len = seq_len
        self.label_dict = LABEL_DICT
        self._transforms = transforms
        self.image_channels = image_channels
        self.use_torchvision_io = bool(use_torchvision_io and read_image is not None)
        self.min_group_points = max(int(min_group_points), 1)
        self.include_empty_windows = bool(include_empty_windows)
        self.negative_window_ratio = max(float(negative_window_ratio), 0.0)

        self.json_folder = os.path.join(self.dataset_folder, 'json')
        self.png_folder = os.path.join(self.dataset_folder, 'images')
        self.json_files = [
            os.path.join(self.json_folder, f)
            for f in sorted(os.listdir(self.json_folder))
            if f.endswith(".json")
        ]
        self.png_files = [
            os.path.join(self.png_folder, f)
            for f in sorted(os.listdir(self.png_folder))
            if f.endswith(".png")
        ]
        assert len(self.json_files) == len(self.png_files), "Mismatch between JSON and PNG files"

        self.frame_annotations = self._load_all_annotations()
        self.valid_items_list = []
        self._filter_valid_item()

    def __len__(self) -> int:
        return len(self.valid_items_list)

    def __getitem__(self, idx: int):
        """
        Fetch a sample and apply transforms.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_sequence, target_dict).
        """
        img_seq, target = self.load_item(idx)
        if self._transforms is not None:
            img_seq, target, _ = self._transforms(img_seq, target, self)
        return img_seq, target

    def _filter_valid_item(self) -> None:
        """Build the list of valid sequence windows for training."""
        positive_items = []
        negative_items = []
        for idx in range(len(self.json_files) - self.seq_len + 1):
            img_files_seq = self.png_files[idx:idx + self.seq_len]
            target_dict = {"points": [], "labels": [], "group_ids": [], "img_seq": []}

            for i in range(self.seq_len):
                data = self.frame_annotations[idx + i]
                if "img_w" not in target_dict and "img_h" not in target_dict:
                    target_dict["img_w"] = data.get('imageWidth', 0)
                    target_dict["img_h"] = data.get('imageHeight', 0)
                else:
                    if target_dict["img_w"] != data.get('imageWidth', 0) or target_dict["img_h"] != data.get(
                            'imageHeight', 0):
                        raise ValueError(
                            "Inconsistent image dimensions: existing 'img_w' or 'img_h' in target_dict does not match the values provided in data")

                for shape in data.get('shapes', []):
                    label = shape.get("label", "")
                    gid = shape.get("group_id", "")
                    if label not in self.label_dict:
                        continue
                    xy = shape["points"][0]
                    target_dict["points"].append(xy)
                    target_dict["labels"].append(self.label_dict[label])
                    target_dict["group_ids"].append(gid)
                    target_dict["img_seq"].append(i)

            if len(target_dict["group_ids"]) > 0:
                max_count = max(Counter(target_dict["group_ids"]).values())
                if max_count >= self.min_group_points:
                    target_dict["is_negative"] = False
                    positive_items.append((idx, (tuple(img_files_seq), target_dict)))
            elif self.include_empty_windows:
                target_dict["is_negative"] = True
                negative_items.append((idx, (tuple(img_files_seq), target_dict)))

        selected_negative_items = []
        if negative_items and self.negative_window_ratio > 0:
            target_negative_count = max(
                1,
                int(np.ceil(len(positive_items) * self.negative_window_ratio)),
            )
            target_negative_count = min(target_negative_count, len(negative_items))
            selected_indices = np.linspace(
                0,
                len(negative_items) - 1,
                num=target_negative_count,
                dtype=int,
            )
            selected_negative_items = [
                negative_items[int(index)] for index in np.unique(selected_indices)
            ]

        all_items = sorted(
            positive_items + selected_negative_items,
            key=lambda item: item[0],
        )
        self.valid_items_list = [item for _, item in all_items]

    def _load_all_annotations(self):
        """Load all JSON annotations for the sequence."""
        annotations = []
        for json_path in self.json_files:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations.append(json.load(f))
        return annotations

    def _read_image(self, image_path: str) -> torch.Tensor:
        """Read one image as a CHW tensor."""
        if self.use_torchvision_io:
            mode = ImageReadMode.GRAY if self.image_channels == 1 else ImageReadMode.RGB
            return read_image(image_path, mode=mode)

        with Image.open(image_path) as img:
            if self.image_channels == 1:
                arr = np.array(img.convert('L'))
                tensor = torch.from_numpy(arr).unsqueeze(0)
            else:
                arr = np.array(img.convert('RGB'))
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor.contiguous()

    def load_item(self, idx: int):
        """
        Build a sample with image sequence and target dictionary.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_sequence, target_dict).
        """
        img_files_seq, target_dict = self.valid_items_list[idx]
        frames = [self._read_image(image_path) for image_path in img_files_seq]

        img_w = target_dict["img_w"]
        img_h = target_dict["img_h"]
        canvas_size = (img_w, img_h, self.seq_len)

        point_values = [
            xy + [t] for xy, t in zip(target_dict["points"], target_dict["img_seq"])
        ]
        points = (
            np.asarray(point_values, dtype=np.float32).reshape(-1, 3)
            if point_values
            else np.empty((0, 3), dtype=np.float32)
        )

        labels = PointLabels(
            np.array(target_dict["labels"], dtype=np.int64),
            label_map=self.label_dict
        )
        group_ids = (
            PointGroupIDs(target_dict["group_ids"])
            if target_dict["group_ids"]
            else PointGroupIDs.empty(id_type="int")
        )

        img_seq = ImageSequence(torch.stack(frames, dim=1), canvas_size=canvas_size)
        pts = Points(points, form="xyt", canvas_size=canvas_size)

        target = {
            "points": pts,
            "points_labels": labels,
            "points_group_ids": group_ids,
            "seq_len": self.seq_len,
            "image_files": img_files_seq,
            "dataset_root": self.dataset_folder,
            "index": idx,
            "is_negative": bool(target_dict.get("is_negative", False)),
            "bboxes": BoundingBoxes3D.empty(format="cxcyczwhd"),
            "bboxes_labels": BoxLabels.empty(label_map=self.label_dict),
            "bboxes_group_ids": BoxGroupIDs.empty(id_type="int"),
        }

        return img_seq, target

    def extra_repr(self) -> str:
        s = f" dataset_folder: {self.dataset_folder}\n seq_len: {self.seq_len}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s
