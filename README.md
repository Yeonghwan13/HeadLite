# HeadLite

PyTorch implementation of the model architecture in **HeadLite: Compact Multi-Modal Sensor Fusion
for Stride Length Estimation from Smart Insoles** (IEEE Sensors Journal, 2026).

[Paper](https://doi.org/10.1109/JSEN.2026.3724195) · [Usage](#usage) · [Citation](#citation)

HeadLite encodes acceleration, angular velocity and plantar pressure in three convolutional
branches, pools each to a fixed-width feature, and combines them with a metadata embedding before
a mean head and an auxiliary variance head. One network has 1,732,146 parameters. The package also
provides a five-network mean-prediction ensemble.

Study data, preprocessing, dataset loaders, evaluation splits, training code and trained weights
are not distributed. The examples run on generated tensors with randomly initialized weights, so
they exercise the interface and the arithmetic rather than reproducing any result from the article.

## Installation

Python 3.10 or newer and a PyTorch build for your platform. The lower bound `torch>=2.13.0` comes
from published PyTorch security advisories, not from an API this package needs.

```bash
git clone https://github.com/Yeonghwan13/HeadLite.git
cd HeadLite
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
```

The `--index-url` applies only to the PyTorch line, which is what selects the CPU build. Installing
the requirements brings in the dependency; installing the project is what makes `headlite`
importable.

Platform specifics, the reasoning behind the version bound, and the environments that were actually
tested are in [Installation details](docs/INSTALLATION.md).

## Usage

```bash
python examples/forward.py              # single network
python examples/forward.py --ensemble   # five-network mean
python examples/verify_model.py         # numerical self-checks
```

`forward.py` prints `parameters: 1732146` and a mean of shape `(2, 1)`; with `--ensemble` it prints
`parameters: 8660730` and the same output shape. `verify_model.py` checks that the computation is
self-consistent: the same inputs twice, the same weights loaded into a fresh model, a state
dictionary written to a temporary directory and read back, and the wrapper's output against the
average computed directly from its five members. It reports the largest difference each check saw
and exits non-zero on failure.

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

print(mean.shape, variance.shape)   # torch.Size([2, 1]) torch.Size([2, 1])
```

Nothing is downloaded and no study data or trained checkpoint is read. `examples/` and `tests/` are
not installed by the wheel, so run them from a checkout; the source archive contains them along
with the files the tests read.

## Inputs and outputs

| Tensor | Shape | Meaning |
|---|---|---|
| `acc` | `(B, 3, 300)` | Prepared acceleration channels |
| `gyr` | `(B, 3, 300)` | Prepared angular-velocity channels |
| `prs` | `(B, 40, 300)` | Prepared pressure channels |
| `meta` | `(B, 6)` | Prepared metadata features |

The four inputs must be finite floating tensors sharing a dtype, device and batch size. The
temporal axis is a length-normalized stride grid, not a fixed duration or sampling rate. Value
scaling, sensor calibration and metadata encoding happen before this code: use the feature order
and scaling that belong to the weights you load.

A single network returns `(mean, variance)`, each `(B, 1)`; the variance head is auxiliary and is
not a calibration guarantee. `HeadLiteEnsemble` averages the means of five networks and returns
`(B, 1)`. It produces no aggregate uncertainty and does not validate an evaluation protocol.

Use `eval()` with `torch.inference_mode()` for inference. Training mode needs a batch of at least
two samples because of BatchNorm.

The head layout, the exact dropout probability and the one place the implementation differs from
Figure 3 are in [Implementation notes](docs/IMPLEMENTATION_NOTES.md).

## Tests

In an environment that already has a supported PyTorch:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest
```

This deliberately does not apply `constraints-tested-cpu.txt`, which pins the exact version this
release was tested with and would downgrade a newer PyTorch you installed on purpose. To reproduce
that exact environment, opt into the pin in a fresh virtual environment:

```bash
python -m pip install -c constraints-tested-cpu.txt -r requirements-dev.txt
python -m pip install -c constraints-tested-cpu.txt -e .
python -m pytest
```

## Citation

Cite the article. The machine-readable metadata is in [CITATION.cff](CITATION.cff).

> Y. Kim, C. Choi, K. Yoo, S. Kim and S.-I. Choi, "HeadLite: Compact Multi-Modal Sensor Fusion for
> Stride Length Estimation from Smart Insoles," *IEEE Sensors Journal*, 2026,
> doi: 10.1109/JSEN.2026.3724195.

## License

No software license is included with this source release, so no reuse terms are granted here.
