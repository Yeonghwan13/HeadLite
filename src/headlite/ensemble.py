"""Arithmetic mean of five HeadLite prediction means."""
from collections.abc import Sequence
import torch
from torch import nn


class HeadLiteEnsemble(nn.Module):
    """Five-network mean predictor; no dataset or evaluation-fold logic.

    This wrapper returns only the mean prediction, not an aggregate variance.
    It does not certify how the members were trained or selected.
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
        means = [model(acc, gyr, prs, meta)[0] for model in self.models]
        if any(x.shape != means[0].shape or x.ndim != 2 or x.shape[1] != 1
               for x in means):
            raise ValueError("Each model must return a mean tensor of shape (B, 1)")
        return torch.stack(means, dim=0).mean(dim=0)
