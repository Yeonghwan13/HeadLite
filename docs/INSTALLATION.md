# Installation details

The [README](../README.md) covers the common case. This page holds the platform specifics, the
reason for the PyTorch version bound, and what was actually tested.

## Requirements

| | Requirement | Role |
|---|---|---|
| Interpreter | Python >= 3.10 | Runs the package |
| Runtime dependency | `torch>=2.13.0` | The model and every tensor operation |
| Test dependency | `pytest>=8` | Runs the test suite |
| Build backend | `setuptools>=68` | Builds the project from `pyproject.toml`; pip fetches it |

`torch` is the only third-party library the model imports; everything else comes from the standard
library. Python, pip and Git are tools you install yourself, not pip packages. The `>=` bounds in
`pyproject.toml` describe the supported range, not a matrix that was tested end to end.

### Why the lower bound is 2.13.0

The bound comes from published PyTorch security advisories, not from anything this package needs.
GHSA-63cw-57p8-fm3p and GHSA-qfhq-4f3w-5fph affect releases below 2.10.0, and GHSA-rrmf-rvhw-rf47
affects releases up to and including 2.12.1, so 2.13.0 is the first release outside all three.
Being outside these three ranges is not a claim that a version has no other issue. The examples
here never read a checkpoint from outside the process, so this concerns the environment you
install rather than anything in this repository.

## Linux and Windows, CPU

This follows the official CPU installation form with the version this release was tested against.
It was run on Linux; the Windows form of the same command was not run here.

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-tested-cpu.txt -r requirements.txt
python -m pip install -c constraints-tested-cpu.txt -e .
python -m pip check
```

The `--index-url` applies to that one command, which is what selects the CPU build; do not make it
the default index for everything else. `constraints-tested-cpu.txt` pins a version, it does not
select a CPU or CUDA build and it is not a full environment lock. Leave it out if you already have
a newer supported PyTorch you do not want downgraded.

On Windows, create and activate the environment with PowerShell first:

```powershell
git clone https://github.com/Yeonghwan13/HeadLite.git
cd HeadLite
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If the execution policy blocks the activation script, call `.\.venv\Scripts\python.exe` directly
instead of activating.

## macOS

PyTorch publishes macOS wheels on PyPI, so the CPU index URL is not used. From 2.13.0 the arm64
wheels target macOS 14 and newer; on an older macOS, or on an Intel Mac, no wheel exists inside the
supported range and the package cannot be installed there.

```bash
python -m pip install torch==2.14.0
python -m pip install -c constraints-tested-cpu.txt -r requirements.txt
python -m pip install -c constraints-tested-cpu.txt -e .
python -m pip check
```

## Another platform, or PyTorch already installed

Install a build your operating system, architecture and Python version support, following the
official PyTorch instructions, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The model runs on CPU and does not need a GPU. CUDA, ROCm and Apple MPS builds are outside what was
tested here. If no wheel exists for your architecture and Python version, install a combination
that has one rather than modifying the package.

## Tested environments

| Platform | Python | PyTorch | What ran |
|---|---|---|---|
| Ubuntu (GitHub Actions, `ubuntu-latest`) | 3.10, 3.11 | 2.14.0 CPU | Install, `pip check`, tests, all three examples |
| Ubuntu (GitHub Actions, `ubuntu-latest`) | 3.11 | 2.14.0 CPU | Wheel and sdist build, install into a clean environment, import from outside the checkout, the sdist's own tests |
| macOS 26 (arm64) | 3.11 | 2.14.0 | Install, `pip check`, tests, all three examples, wheel and sdist install |
| macOS 26 (arm64) | 3.11 | 2.13.0 | The declared lower bound: full test suite and all three examples |

The unit-test rows and the packaging row are separate runs rather than one combined matrix.

Not exercised: Windows and any GPU path. Other operating systems and Python versions are supported
by declaration rather than by a recorded run.

## Loading a checkpoint

The repository ships no trained weights. If you load one of your own, read it as a tensor state
dictionary:

```python
state = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
```

`weights_only=True` is the right default but is not a guarantee against every malformed file, and
it has itself been the subject of PyTorch security advisories. Load only checkpoints you produced
or otherwise trust, on a PyTorch inside the supported range.
