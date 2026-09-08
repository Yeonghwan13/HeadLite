# HeadLite

PyTorch model definitions for **HeadLite: Compact Multi-Modal Sensor Fusion for Stride Length Estimation from Smart Insoles** (IEEE Sensors Journal, 2026).

[Paper](https://doi.org/10.1109/JSEN.2026.3724195)

HeadLite encodes acceleration, angular velocity and plantar pressure in separate convolutional branches. Their pooled features are combined with a metadata embedding and passed to mean and auxiliary variance heads. One network has **1,732,146 parameters**.

## Scope

This release contains the model architecture, a five-network mean-prediction wrapper and generated-input examples. Study data, preprocessing, dataset loaders, evaluation splits, training pipelines and pretrained weights are not included. The examples check the software interface; they do not reproduce the study results.

## Installation

### What you need

| | Requirement | Why |
|---|---|---|
| Interpreter | Python >= 3.10 | Runs the package |
| Runtime dependency | `torch>=2.13.0` | The model and every tensor operation |
| Test dependency | `pytest>=8` | Runs the test suite |
| Build backend | `setuptools>=68` | Installs the project from `pyproject.toml`; pip fetches it |

Python, pip and Git are tools you install yourself, not pip packages. `torch` is the only
third-party library the model imports; everything else it uses is in the standard library.
The `>=` bounds in `pyproject.toml` are the supported range, not a claim that every combination
inside it was tested.

The PyTorch lower bound comes from published security advisories rather than from anything this
package needs: GHSA-63cw-57p8-fm3p and GHSA-qfhq-4f3w-5fph affect releases below 2.10.0, and
GHSA-rrmf-rvhw-rf47 affects releases up to and including 2.12.1. 2.13.0 is the first release
outside all three. The examples here never read a checkpoint from outside the process, so this is
about the environment you install, not about anything in this repository.

### Get the repository

```bash
git clone https://github.com/Yeonghwan13/HeadLite-review.git
cd HeadLite-review
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

This assumes Python 3.11 is already installed and on your `PATH`. Any supported interpreter works;
3.11 is used here because it matches the verified environment below.

On Windows, use PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If your execution policy blocks the activation script, run `.\.venv\Scripts\python.exe` directly
instead of activating.

Without Git, download the source archive, unpack it, and start from the top-level folder.

### Install: Linux or Windows, CPU

The PyTorch command below is the CPU one from the official installation instructions for 2.9.1.
It was run on Linux for this release; the Windows form of the same command was not run here.

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-tested-cpu.txt -r requirements.txt
python -m pip install -c constraints-tested-cpu.txt -e .
python -m pip check
```

The `--index-url` applies to that one command, which selects the CPU PyTorch build. Do not make it
the default index for everything else. `pip install -e .` also resolves the declared dependencies,
so the two install lines are shown for clarity rather than because both are mechanically required;
what matters is that installing the requirements alone does not install HeadLite.

`constraints-tested-cpu.txt` pins the PyTorch version this release was tested with. It constrains
versions, it does not select a CPU or CUDA build, and it is not a full environment lock. Leave it
out if you already have a supported newer PyTorch that you do not want downgraded.

### Install: macOS

PyTorch publishes macOS wheels on PyPI, so the CPU index URL above is not used. From 2.13.0
onwards the arm64 wheels are built for macOS 14 and newer; on an older macOS, or on an Intel Mac,
no wheel in the supported range exists and this package cannot be installed there:

```bash
python -m pip install torch==2.14.0
python -m pip install -c constraints-tested-cpu.txt -r requirements.txt
python -m pip install -c constraints-tested-cpu.txt -e .
python -m pip check
```

### Another platform, or PyTorch already installed

Install a PyTorch build that your operating system, architecture and Python version support, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Pick the build from the official PyTorch instructions for your platform. CUDA, ROCm and Apple MPS
builds are outside what was tested here; the model runs on CPU and does not require a GPU. If no
wheel exists for your architecture and Python version, install a combination that has one rather
than changing the package.

### Check the installation

```bash
python -c "import headlite, torch; print(headlite.__version__, torch.__version__, headlite.__file__)"
python examples/forward.py
python examples/forward.py --ensemble
python examples/verify_model.py
```

`examples/forward.py` prints `parameters: 1732146` and a mean of shape `(2, 1)`; with `--ensemble`
it prints `parameters: 8660730` and a mean of shape `(2, 1)`. The printed values come from random
weights and are not stride-length predictions.

`examples/` and `tests/` are not installed by the wheel, so those commands are run from a checkout.
The source archive contains them, together with the requirements files the tests read.

### Verified environments

| Platform | Python | PyTorch | Ran |
|---|---|---|---|
| Ubuntu (GitHub Actions, `ubuntu-latest`) | 3.10, 3.11 | 2.14.0 CPU | Install, `pip check`, tests, all three examples |
| Ubuntu (GitHub Actions, `ubuntu-latest`) | 3.11 | 2.14.0 CPU | Wheel and sdist build, install into a clean environment, import from outside the checkout, the sdist's own tests |
| macOS 26 (arm64) | 3.11 | 2.14.0 | Install, `pip check`, tests, all three examples, wheel and sdist install |

The unit-test rows and the packaging row are separate runs, not one combined matrix.

Not exercised: Windows, any GPU path (CUDA, ROCm, Apple MPS), and trained weights. The Windows
commands above are the documented form, not a run recorded here. Other operating systems and
Python versions are supported by declaration.

The model also runs under PyTorch 2.13.0, which is the declared lower bound.

## Quick start

```bash
python examples/forward.py
python examples/forward.py --ensemble
```

```bash
python examples/verify_model.py --seed 77
```

The examples use generated tensors and random weights. Nothing is downloaded and no study data or
trained checkpoint is read.

`verify_model.py` checks that the computation is self-consistent: the same inputs through the same
model twice, the same weights loaded into a fresh model, a state dictionary written to a temporary
directory and read back, and the wrapper's output against the average computed directly from the
five members. It prints the largest difference each check observed and exits non-zero if any check
fails. It writes a temporary state dictionary while it runs and removes it afterwards; running the
tests also creates pytest's own temporary files.

This verifies that the model computes consistently. It is not a reproduction of any measurement in
the article: with random weights the numbers carry no information about stride length.

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
python -m pip install -c constraints-tested-cpu.txt -r requirements-dev.txt
python -m pytest -q
```

`requirements-dev.txt` includes `requirements.txt`, so this adds `pytest` to an environment that
already has the runtime dependency. Running the tests creates pytest's own temporary files.

## Citation

Use the article metadata in [CITATION.cff](CITATION.cff).

## License

No software license is included with this source release. No additional reuse terms are granted here.
