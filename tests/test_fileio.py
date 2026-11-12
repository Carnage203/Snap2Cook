"""Q6 text tools and Q8 SafeFile."""
from __future__ import annotations

from pathlib import Path

import pytest

from snap2cook.exceptions import FileOperationError
from snap2cook.fileio.safe_io import SafeFile
from snap2cook.fileio.text_tools import FileTextProcessor


def test_q6_sample_word_order_matches_syllabus() -> None:
    assert FileTextProcessor("Python Programming").reverse_word_order_per_line() == "Programming Python"


def test_q6_each_word_reversed_keeps_spacing() -> None:
    assert FileTextProcessor("ab cd  ef").reverse_each_word() == "ba dc  fe"


def test_q6_line_order_and_strip_whitespace() -> None:
    proc = FileTextProcessor("one\ntwo\nthree")
    assert proc.reverse_line_order() == "three\ntwo\none"
    assert FileTextProcessor(" a b\tc\n d ").strip_whitespace() == "abcd"


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_q6_empty_and_whitespace_only(text: str) -> None:
    proc = FileTextProcessor(text)
    assert proc.strip_whitespace() == ""
    assert proc.reverse_word_order_per_line().strip() == ""
    assert proc.reverse_line_order().strip() == ""


def test_q6_from_missing_file_raises_custom_error(tmp_path: Path) -> None:
    with pytest.raises(FileOperationError):
        FileTextProcessor.from_file(tmp_path / "missing.txt")


def test_safefile_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileOperationError) as info:
        SafeFile(tmp_path / "nope.txt").read_text()
    assert info.value.op == "read_text" and "nope.txt" in info.value.path


def test_safefile_corrupt_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(FileOperationError, match="Invalid JSON"):
        SafeFile(bad).read_json()


def test_safefile_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "secret.txt"
    target.write_text("x")

    def deny(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", deny)
    with pytest.raises(FileOperationError, match="Permission denied"):
        SafeFile(target).read_text()


def test_safefile_not_utf8(tmp_path: Path) -> None:
    binary = tmp_path / "b.txt"
    binary.write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(FileOperationError, match="UTF-8"):
        SafeFile(binary).read_text()


def test_safefile_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    target = SafeFile(tmp_path / "sub" / "data.json")
    target.write_json({"a": 1})
    target.write_json({"a": 2})
    assert target.read_json() == {"a": 2}
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["data.json"]
