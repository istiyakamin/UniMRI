"""Run the runnable examples end-to-end so they cannot silently rot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("[0-8]*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,  # examples may write files; keep them out of the repo
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_aspirational_example_is_not_counted() -> None:
    # The 99_* example is intentionally non-runnable; make sure we did not pick it up.
    assert all(not p.name.startswith("99") for p in EXAMPLES)
