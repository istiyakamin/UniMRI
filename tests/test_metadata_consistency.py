"""Keep the version and packaging metadata single-sourced.

The version lives only in ``src/unimri/__init__.py``. These tests fail loudly if
another file drifts out of sync.
"""

from __future__ import annotations

from pathlib import Path

import unimri

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).parent.parent


def test_citation_cff_version_matches() -> None:
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    line = next(ln for ln in cff.splitlines() if ln.strip().startswith("version:"))
    cff_version = line.split(":", 1)[1].strip().strip("'\"")
    assert cff_version == unimri.__version__, (
        f"CITATION.cff version {cff_version!r} != unimri.__version__ "
        f"{unimri.__version__!r} (see RELEASING.md)"
    )


def test_pyproject_version_is_dynamic() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert "version" not in project, "pin removed: version must stay dynamic"
    assert "version" in project.get("dynamic", []), "project.dynamic must include 'version'"
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/unimri/__init__.py"


def test_readme_and_license_agree_on_license() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = pyproject["project"]["classifiers"]
    assert any("MIT" in c for c in classifiers)
    assert "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
