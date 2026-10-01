import torch 
import torch.utils.data as data

from ..core import register


__all__ = ['DataLoader']


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    """Thin wrapper around PyTorch DataLoader for config-based construction."""

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        """Store the current epoch for distributed samplers."""
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1


@register()
def default_collate_fn(items):
    """
    Default collate function for FalconDet sequences.

    Args:
        items: List of (image_sequence, target) tuples.

    Returns:
        Tuple of (batch_tensor, target_list).
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


