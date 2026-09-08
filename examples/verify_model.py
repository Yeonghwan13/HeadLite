"""Check that the model computes consistently: same inputs, same weights, same result.

This exercises the software on generated tensors. It does not use study data, does not load
trained weights, and does not reproduce any result from the article. A prediction made with
randomly initialized weights carries no information about stride length.

    python examples/verify_model.py
    python examples/verify_model.py --seed 77 --device cpu

Every check prints the largest difference it observed. The process exits non-zero if any check
fails, so it is usable as a smoke test in a pipeline.
"""
import argparse
import platform
import sys
import tempfile
from pathlib import Path

import torch

from headlite import MODEL_CONFIG, HeadLite, HeadLiteEnsemble

BATCHES = (1, 2, 5)


def make_inputs(batch, device):
    """Generated tensors in the documented shapes. Not measurements."""
    return (
        torch.randn(batch, 3, 300, device=device),
        torch.randn(batch, 3, 300, device=device),
        torch.rand(batch, 40, 300, device=device),
        torch.randn(batch, 6, device=device),
    )


class Checks:
    """Collects results so that one failure does not hide the remaining checks."""

    def __init__(self):
        self.rows = []

    def record(self, name, ok, detail):
        self.rows.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    def equal(self, name, a, b, note=""):
        """Bit-exact comparison. Two runs of the same code on the same device must agree."""
        same_shape = a.shape == b.shape
        difference = (a - b).abs().max().item() if same_shape else float("nan")
        self.record(name, same_shape and difference == 0.0,
                    f"max difference {difference:.3e}{note}" if same_shape
                    else f"shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")

    @property
    def failed(self):
        return [name for name, ok, _ in self.rows if not ok]


def check_repeat(checks, model, inputs):
    """The same inputs through the same model twice. Dropout is inactive in eval mode."""
    with torch.inference_mode():
        first_mean, first_variance = model(*inputs)
        second_mean, second_variance = model(*inputs)
    checks.equal("repeated evaluation, mean", first_mean, second_mean)
    checks.equal("repeated evaluation, variance", first_variance, second_variance)
    return first_mean, first_variance


def check_reload(checks, model, inputs, reference):
    """A fresh model given the same weights must compute the same thing.

    The state dictionary is written to a temporary directory and read back with
    ``weights_only=True``. Only tensors this process just wrote are read; never a file from
    elsewhere. See the note on trusted checkpoints in the README.
    """
    replica = HeadLite().eval().to(reference.device)
    replica.load_state_dict(model.state_dict(), strict=True)
    with torch.inference_mode():
        replica_mean, _ = replica(*inputs)
    checks.equal("strict load into a new model", replica_mean, reference)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.pt"
        torch.save(model.state_dict(), path)
        restored = HeadLite().eval().to(reference.device)
        restored.load_state_dict(torch.load(path, map_location=reference.device, weights_only=True),
                                 strict=True)
        with torch.inference_mode():
            restored_mean, _ = restored(*inputs)
        exists_after_save = path.exists()
    checks.equal("save and reload", restored_mean, reference)
    checks.record("temporary checkpoint removed", not path.exists(),
                  f"written: {exists_after_save}, present after cleanup: {path.exists()}")


def check_ensemble(checks, inputs, device):
    """The wrapper's output must equal the average computed directly, in the same order."""
    members = [HeadLite().eval().to(device) for _ in range(5)]
    wrapper = HeadLiteEnsemble(members).eval()
    with torch.inference_mode():
        from_wrapper = wrapper(*inputs)
        directly = torch.stack([member(*inputs)[0] for member in members], dim=0).mean(dim=0)
    checks.equal("ensemble equals the direct average", from_wrapper, directly)
    checks.record("ensemble output shape", from_wrapper.shape == (inputs[0].shape[0], 1),
                  f"{tuple(from_wrapper.shape)} for a batch of {inputs[0].shape[0]}")
    total = sum(p.numel() for member in members for p in member.parameters())
    checks.record("five networks hold five times the parameters",
                  total == 5 * 1_732_146, f"{total:,}")


def check_shapes(checks, model, device):
    """Shapes and finiteness across several batch sizes."""
    for batch in BATCHES:
        inputs = make_inputs(batch, device)
        with torch.inference_mode():
            mean, variance = model(*inputs)
        ok = (mean.shape == (batch, 1) and variance.shape == (batch, 1)
              and torch.isfinite(mean).all() and torch.isfinite(variance).all())
        checks.record(f"batch {batch}", bool(ok),
                      f"mean {tuple(mean.shape)}, variance {tuple(variance.shape)}, both finite")


def check_configuration(checks, model):
    """The fixed configuration and the resulting parameter and state-dict counts."""
    parameters = sum(p.numel() for p in model.parameters())
    checks.record("parameter count", parameters == 1_732_146, f"{parameters:,}")
    state = model.state_dict()
    buffers = len(list(model.named_buffers()))
    tensors = len(list(model.named_parameters()))
    checks.record("state dictionary", len(state) == 115 and tensors == 76 and buffers == 39,
                  f"{len(state)} entries = {tensors} parameter tensors + {buffers} buffers")
    dropouts = [(name, module.p) for name, module in model.named_modules()
                if isinstance(module, torch.nn.Dropout)]
    expected_p = MODEL_CONFIG["dropout_rate"]
    checks.record("dropout layers", len(dropouts) == 2 and all(p == expected_p for _, p in dropouts),
                  f"{[name for name, _ in dropouts]} at p={expected_p!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0,
                        help="seed applied before the weights and the inputs are generated")
    parser.add_argument("--device", default="cpu",
                        help="torch device; cpu is the tested path")
    arguments = parser.parse_args(argv)

    device = torch.device(arguments.device)
    print(f"seed {arguments.seed} | python {platform.python_version()} | "
          f"torch {torch.__version__} | device {device}")
    print("Generated tensors and random weights. This verifies computation, not accuracy.\n")

    torch.manual_seed(arguments.seed)
    model = HeadLite().eval().to(device)
    inputs = make_inputs(2, device)

    checks = Checks()
    mean, _ = check_repeat(checks, model, inputs)
    check_reload(checks, model, inputs, mean)
    check_ensemble(checks, inputs, device)
    check_shapes(checks, model, device)
    check_configuration(checks, model)

    failed = checks.failed
    print()
    if failed:
        print(f"{len(failed)} of {len(checks.rows)} checks failed: {', '.join(failed)}")
        return 1
    print(f"all {len(checks.rows)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
