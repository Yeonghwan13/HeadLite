import torch
import pytest
from headlite import HeadLiteEnsemble


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__(); self.register_buffer("value", torch.tensor(float(value)))
    def forward(self, acc, gyr, prs, meta):
        mean = self.value.expand(acc.shape[0], 1)
        return mean, torch.ones_like(mean)


def test_prediction_average():
    e = HeadLiteEnsemble([Constant(i) for i in range(1, 6)]).eval()
    x = torch.zeros(3, 1)
    y = e(x, x, x, x)
    assert y.shape == (3, 1)
    assert torch.equal(y, torch.full((3, 1), 3.0))
    assert not any(m.training for m in e.models)


@pytest.mark.parametrize("count", [0, 1, 4, 6])
def test_wrong_size(count):
    with pytest.raises(ValueError, match="five"):
        HeadLiteEnsemble([Constant(i) for i in range(count)])


def test_duplicate_object():
    m = Constant(1)
    with pytest.raises(ValueError, match="same model"):
        HeadLiteEnsemble([m] * 5)
