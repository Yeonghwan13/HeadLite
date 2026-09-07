# HeadLite

PyTorch model definitions for **HeadLite: Compact Multi-Modal Sensor Fusion for Stride Length Estimation from Smart Insoles** (IEEE Sensors Journal, 2026).

[Paper](https://doi.org/10.1109/JSEN.2026.3724195)

HeadLite encodes acceleration, angular velocity and plantar pressure in separate convolutional branches. Their pooled features are combined with a metadata embedding and passed to mean and auxiliary variance heads. One network has **1,732,146 parameters**.

## Scope

This release contains the model architecture, a five-network mean-prediction wrapper and generated-input examples. Study data, preprocessing, dataset loaders, evaluation splits, training pipelines and pretrained weights are not included. The examples check the software interface; they do not reproduce the study results.

## Installation

Python 3.10 or newer. Install a PyTorch build suitable for your platform, then run from the checkout:

```bash
python -m pip install -e .
```

Installing the package provides the `headlite` library. `examples/` and `tests/` are not part of
the installed package, so those commands are run from a checkout.

Verified on CPU with Python 3.11 and PyTorch 2.9.1, and in continuous integration on Python 3.10
and 3.11. The `>=` bounds in `pyproject.toml` are the supported range, not a claim that every
combination inside it was tested. No GPU path and no trained weights were exercised.

## Quick start

```bash
python examples/forward.py
python examples/forward.py --ensemble
```

Both examples use generated tensors and random weights. No data or checkpoints are downloaded, and nothing is written to disk.

```python
import torch
from headlite import HeadLite

model = HeadLite().eval()
acc = torch.randn(2, 3, 300)
gyr = torch.randn(2, 3, 300)
prs = torch.rand(2, 40, 300)
meta = torch.randn(2, 6)

with torch.inference_mode():
    mean, variance = model(acc, gyr, prs, meta)

print(mean.shape, variance.shape)  # torch.Size([2, 1]), torch.Size([2, 1])
```

These are interface examples, not meaningful stride-length predictions without trained weights.

## Inputs and outputs

| Tensor | Shape | Meaning |
|---|---|---|
| `acc` | `(B, 3, 300)` | Prepared acceleration channels |
| `gyr` | `(B, 3, 300)` | Prepared angular-velocity channels |
| `prs` | `(B, 40, 300)` | Prepared pressure channels |
| `meta` | `(B, 6)` | Prepared metadata features |

Inputs must share a floating dtype, device and batch size. The temporal dimension is a **length-normalized stride grid**, not a fixed physical duration. Metadata order is sex, shoe size, height, weight, age and BMI; the model does not encode or scale these fields. Any real use must supply the value scaling and metadata conventions appropriate to the trained weights.

A single network returns `(mean, variance)`, each shaped `(B, 1)`. The variance output is not a guarantee of calibration. `HeadLiteEnsemble` averages five model means in PyTorch and returns `(B, 1)`; it does not produce aggregate uncertainty or verify an evaluation protocol.

The exact head layout is documented in [Implementation notes](docs/IMPLEMENTATION_NOTES.md).

## Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Citation

Use the article metadata in [CITATION.cff](CITATION.cff).

## License

No software license has been selected for this review copy. No additional reuse terms are granted here.
