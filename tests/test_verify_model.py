"""The verification example must actually verify, and must fail when the model is wrong.

`examples/verify_model.py` is the one command a reader runs to convince themselves the model
computes consistently. A script that always prints PASS would be worse than none, so its checks
are exercised here against a deliberately broken model as well as the real one.
"""
import runpy
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "verify_model.py"


def run(argv, monkeypatch=None):
    """Run the example as a script and return (exit code, stdout)."""
    module = _load()
    return module.main(argv)


def _load():
    if not EXAMPLE.exists():
        pytest.skip("examples/ is not present; it ships in the source tree, not in the wheel")
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_model_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_example_passes_on_the_real_model(capsys):
    assert run(["--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "all 14 checks passed" in out
    assert "FAIL" not in out
    assert "1,732,146" in out


def test_the_seed_is_reported_and_honoured(capsys):
    run(["--seed", "12345"])
    assert "seed 12345" in capsys.readouterr().out


def test_the_example_says_it_is_not_a_performance_claim(capsys):
    run([])
    out = capsys.readouterr().out.lower()
    assert "not accuracy" in out or "verifies computation" in out


def test_reported_differences_are_zero_on_one_device(capsys):
    """Two runs of the same code on one device must agree exactly, not approximately."""
    run([])
    for line in capsys.readouterr().out.splitlines():
        if "max difference" in line:
            assert "0.000e+00" in line, line


def test_the_example_fails_when_the_ensemble_average_is_wrong(capsys, monkeypatch):
    """Break the averaging and the script must report a failure and return non-zero."""
    module = _load()

    class Skewed(module.HeadLiteEnsemble):
        def forward(self, acc, gyr, prs, meta):
            return super().forward(acc, gyr, prs, meta) + 1.0

    monkeypatch.setattr(module, "HeadLiteEnsemble", Skewed)
    assert module.main([]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "ensemble equals the direct average" in out


def test_the_example_fails_when_the_parameter_count_changes(capsys, monkeypatch):
    """The count is asserted, not merely printed."""
    module = _load()
    original = module.HeadLite

    class Extra(original):
        def __init__(self):
            super().__init__()
            self.unused = torch.nn.Linear(4, 4)

    monkeypatch.setattr(module, "HeadLite", Extra)
    assert module.main([]) == 1
    assert "FAIL  parameter count" in capsys.readouterr().out


def test_an_unknown_argument_exits_non_zero():
    module = _load()
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--not-an-option"])
    assert exit_info.value.code != 0


def test_the_example_writes_nothing_into_the_checkout(capsys):
    """The temporary state dictionary must not be left behind in the repository."""
    root = EXAMPLE.resolve().parent.parent
    before = {p for p in root.rglob("*") if p.suffix in {".pt", ".pth", ".ckpt"}}
    run([])
    after = {p for p in root.rglob("*") if p.suffix in {".pt", ".pth", ".ckpt"}}
    assert before == after, f"left behind: {sorted(after - before)}"
