"""Arithmetic mean of five HeadLite prediction means."""
from collections.abc import Sequence
import torch
from torch import nn


def _extract_mean(output, index):
    """Take the mean tensor out of a member's return value.

    Members are ordinary ``nn.Module`` objects, so the wrapper cannot assume anything about what
    they return. A model is expected to return ``(mean, variance)``; a bare tensor, a dictionary
    or a short sequence is refused by name rather than failing later inside ``torch.stack``.
    """
    if isinstance(output, torch.Tensor):
        raise TypeError(
            f"member {index} returned a bare tensor; a member must return (mean, variance)")
    if not isinstance(output, (tuple, list)) or len(output) < 1:
        raise TypeError(
            f"member {index} returned {type(output).__name__}; a member must return "
            "(mean, variance)")
    mean = output[0]
    if not isinstance(mean, torch.Tensor):
        raise TypeError(
            f"member {index} returned a {type(mean).__name__} as its mean; a tensor is required")
    return mean


def _check_mean(mean, index, batch):
    """Refuse a mean that cannot be averaged into a per-sample prediction.

    The shape is checked against the *input* batch, not merely against the other members. Five
    members that all ignore the batch and return one row agree with each other, so a
    member-to-member comparison alone accepts them and the wrapper then returns one row for a
    many-sample input.
    """
    if mean.ndim != 2 or mean.shape[1] != 1:
        raise ValueError(
            f"member {index} returned a mean of shape {tuple(mean.shape)}; (B, 1) is required")
    if mean.shape[0] != batch:
        raise ValueError(
            f"member {index} returned {mean.shape[0]} row(s) for an input batch of {batch}. "
            "Each member must predict one value per input sample; a member that ignores the "
            "batch would otherwise silently change the number of predictions.")
    if not mean.is_floating_point():
        raise TypeError(
            f"member {index} returned a mean of dtype {mean.dtype}; a floating dtype is required "
            "to average. The dtype itself is not constrained further, so autocast and reduced "
            "precision are allowed.")
    if not torch.isfinite(mean).all():
        raise ValueError(
            f"member {index} returned a non-finite mean. A single NaN or Inf would propagate "
            "through the average and produce a non-finite prediction without any error.")


class HeadLiteEnsemble(nn.Module):
    """Five-network mean predictor; no dataset or evaluation-fold logic.

    This wrapper returns only the mean prediction, not an aggregate variance.
    It does not certify how the members were trained or selected.

    Members may be any ``nn.Module`` that returns ``(mean, variance)``. Each mean must be a
    floating tensor of shape ``(B, 1)`` where ``B`` is the batch of the inputs passed in, must be
    finite, and must sit on one device shared by all five. The result is the arithmetic mean of
    the five member means, in the order given.
    """
    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        members = list(models)
        if len(members) != 5:
            raise ValueError("HeadLiteEnsemble requires exactly five models")
        if not all(isinstance(m, nn.Module) for m in members):
            raise TypeError("All members must be torch.nn.Module objects")
        if len({id(m) for m in members}) != 5:
            raise ValueError("The same model object cannot be included twice")
        self.models = nn.ModuleList(members)

    def forward(self, acc, gyr, prs, meta):
        if not isinstance(acc, torch.Tensor) or acc.ndim < 1:
            raise TypeError("acc must be a tensor with a leading batch dimension")
        batch = acc.shape[0]
        means = []
        for index, model in enumerate(self.models):
            mean = _extract_mean(model(acc, gyr, prs, meta), index)
            _check_mean(mean, index, batch)
            means.append(mean)
        devices = {m.device for m in means}
        if len(devices) != 1:
            raise ValueError(
                f"member means are on {len(devices)} different devices: "
                f"{sorted(str(d) for d in devices)}")
        return torch.stack(means, dim=0).mean(dim=0)
