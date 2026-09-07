"""Fixed HeadLite architecture and its tensor input contract."""
import torch
from types import MappingProxyType
from ._network import Encoder40ChMeta

MODEL_CONFIG = MappingProxyType({
    "acc_gyr_dim1": 32, "prs_dim1": 48, "dense_dim": 256,
    "dropout_rate": 0.21469034765110603,
    "meta_feature_dim": 6, "meta_encode_dim": 64, "proj_dim": 128,
})


def validate_inputs(acc, gyr, prs, meta):
    """Validate already-prepared tensors; do not transform their values."""
    values = (("acc", acc, (3, 300)), ("gyr", gyr, (3, 300)),
              ("prs", prs, (40, 300)), ("meta", meta, (6,)))
    for name, value, tail in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != len(tail) + 1 or tuple(value.shape[1:]) != tail:
            raise ValueError(f"{name} must have shape (B, {', '.join(map(str, tail))})")
        if value.shape[0] < 1 or not value.is_floating_point():
            raise ValueError(f"{name} requires a nonempty batch and floating dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
    if any(v.shape[0] != acc.shape[0] or v.device != acc.device or v.dtype != acc.dtype
           for _, v, _ in values):
        raise ValueError("All inputs must share batch size, dtype and device")


class HeadLite(Encoder40ChMeta):
    """Construct the fixed 1,732,146-parameter architecture (random weights)."""
    def __init__(self):
        super().__init__(**MODEL_CONFIG)

    def forward(self, acc, gyr, prs, meta, return_embedding=False):
        validate_inputs(acc, gyr, prs, meta)
        if self.training and acc.shape[0] < 2:
            raise ValueError("BatchNorm requires at least two samples in training mode; use eval() for single-stride inference")
        return super().forward(acc, gyr, prs, meta, return_embedding=return_embedding)


def build_headlite():
    """Return a HeadLite network; no weights are downloaded or loaded."""
    return HeadLite()
