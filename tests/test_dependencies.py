"""The declared dependencies, the requirements files and the imports must agree.

pyproject.toml is the single source of truth for what this package depends on. The requirements
files exist so that a reader can see which libraries to install; that only helps if they cannot
drift away from pyproject.toml, so the agreement is checked here rather than by hand.
"""
import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
# tomllib joined the standard library in 3.11, so an interpreter older than that does not
# list it even though it is not a third-party package on any version.
STDLIB = set(sys.stdlib_module_names) | {"tomllib"}


def _entries(name):
    """Requirement lines from a requirements file: no comments, no blanks, no pip options."""
    lines = []
    for raw in (ROOT / name).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            lines.append(line)
    return lines


def _options(name):
    """The pip option lines, such as ``-r requirements.txt``."""
    return [raw.split("#", 1)[0].strip() for raw in (ROOT / name).read_text().splitlines()
            if raw.split("#", 1)[0].strip().startswith("-")]


def _split(requirement):
    """Break ``torch>=2.9`` into ``("torch", ">=", (2, 9))``."""
    for operator in ("==", ">=", "<=", "~=", ">", "<"):
        if operator in requirement:
            name, _, version = requirement.partition(operator)
            return name.strip(), operator, tuple(int(p) for p in version.strip().split("."))
    return requirement.strip(), "", ()


# tomllib entered the standard library in Python 3.11. Adding a TOML parser to the package just to
# read its own metadata would be a real dependency bought for a check, so on 3.10 these comparisons
# are skipped and the rest of the suite still runs there.
requires_tomllib = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="tomllib requires Python 3.11; the TOML comparison runs on the newer matrix interpreters")


def _toml():
    import tomllib
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _imported_third_party(directory):
    root = ROOT / directory
    assert root.is_dir(), f"{directory}/ is missing, so this check would pass without scanning it"
    modules = set()
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return {m for m in modules if m not in STDLIB and m != "headlite"}


@requires_tomllib
def test_runtime_requirements_match_pyproject():
    assert _entries("requirements.txt") == _toml()["project"]["dependencies"]


@requires_tomllib
def test_dev_requirements_match_the_dev_extra():
    assert "-r requirements.txt" in _options("requirements-dev.txt"), (
        "requirements-dev.txt must pull in the runtime requirements")
    assert _entries("requirements-dev.txt") == _toml()["project"]["optional-dependencies"]["dev"]


@requires_tomllib
def test_the_tested_constraint_satisfies_the_declared_range():
    constraints = _entries("constraints-tested-cpu.txt")
    declared = {_split(d)[0]: _split(d) for d in _toml()["project"]["dependencies"]}
    assert constraints, "the constraints file must pin something"
    for pin in constraints:
        name, operator, version = _split(pin)
        assert operator == "==", f"{name} must be pinned exactly in a constraints file"
        assert name in declared, f"{name} is constrained but is not a declared dependency"
        _, declared_operator, declared_version = declared[name]
        assert declared_operator == ">=", f"unhandled specifier for {name}: {declared_operator}"
        assert version >= declared_version, (
            f"the tested {name} {version} is below the declared minimum {declared_version}")


def test_every_third_party_import_is_declared():
    """A new import must not slip in without appearing in the dependency list."""
    runtime = _imported_third_party("src")
    assert runtime == {"torch"}, f"unexpected runtime imports: {sorted(runtime - {'torch'})}"
    for module in _imported_third_party("examples"):
        assert module in runtime, f"the examples import {module}, which is not a runtime dependency"
    tests_only = _imported_third_party("tests") - runtime
    assert tests_only == {"pytest"}, f"unexpected test imports: {sorted(tests_only)}"


def test_the_package_itself_is_not_listed_as_a_requirement():
    """`pip install -r requirements.txt` installs the dependencies; the package is installed
    separately, and the README says so. A self-referencing entry would hide that step."""
    for name in ("requirements.txt", "requirements-dev.txt"):
        for entry in _entries(name) + _options(name):
            assert entry not in {"-e .", ".", "-e .[dev]", ".[dev]"}, (
                f"{name} installs the project itself via {entry!r}; keep that step explicit")
