"""Contract tests for the reader registry and dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from unimri.exceptions import UnsupportedFormatError
from unimri.io import (
    ISMRMRDReader,
    Reader,
    TwixReader,
    available_readers,
    read,
    register_reader,
)
from unimri.io.registry import _READERS


@pytest.fixture(autouse=True)
def _restore_registry():
    saved = list(_READERS)
    yield
    _READERS[:] = saved


def test_builtin_readers_registered() -> None:
    kinds = {type(r) for r in available_readers()}
    assert {ISMRMRDReader, TwixReader}.issubset(kinds)


def test_unknown_extension_raises(tmp_path: Path) -> None:
    f = tmp_path / "scan.xyz"
    f.write_bytes(b"not a real format")
    with pytest.raises(UnsupportedFormatError, match="no registered reader"):
        read(f)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read(tmp_path / "does_not_exist.mrd")


def test_custom_reader_dispatch(tmp_path: Path) -> None:
    calls: list[Path] = []

    class DummyReader(Reader):
        format_name = "Dummy"
        extensions = (".dummy",)

        def can_read(self, path) -> bool:
            return Path(path).suffix == ".dummy"

        def read(self, path, **options):
            calls.append(Path(path))
            return "dummy-result"

    register_reader(DummyReader(), prepend=True)
    f = tmp_path / "x.dummy"
    f.write_bytes(b"\x00")
    assert read(f) == "dummy-result"
    assert calls == [f]


def test_twix_magic_bytes_sniff(tmp_path: Path) -> None:
    reader = TwixReader()
    good = tmp_path / "meas.dat"
    good.write_bytes((0).to_bytes(4, "little") + (1).to_bytes(4, "little") + b"\x00" * 16)
    assert reader.can_read(good)

    bad = tmp_path / "other.dat"
    bad.write_bytes(b"\xff" * 24)
    assert not reader.can_read(bad)


def test_twix_read_not_implemented_yet(tmp_path: Path) -> None:
    f = tmp_path / "meas.dat"
    f.write_bytes((0).to_bytes(4, "little") + (1).to_bytes(4, "little") + b"\x00" * 16)
    with pytest.raises(NotImplementedError):
        read(f)
