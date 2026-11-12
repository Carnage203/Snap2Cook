# SYLLABUS Q8 - handle run-time errors in file handling
"""SafeFile: every disk access in the app goes through this class."""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from snap2cook.exceptions import FileOperationError


class SafeFile:
    """Wraps a path; all failures are re-raised as :class:`FileOperationError`."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """The wrapped path."""
        return self._path

    def __repr__(self) -> str:
        return f"SafeFile({str(self._path)!r})"

    @contextmanager
    def _guard(self, op: str) -> Iterator[None]:
        """Translate low-level errors into FileOperationError with context."""
        def fail(msg: str, friendly: str, exc: BaseException) -> FileOperationError:
            return FileOperationError(f"{msg}: {self._path}", self._path, op, friendly)

        try:
            yield
        except FileNotFoundError as exc:
            raise fail("File not found", "That file was not found.", exc) from exc
        except PermissionError as exc:
            raise fail("Permission denied", "We are not allowed to access that file.", exc) from exc
        except IsADirectoryError as exc:
            raise fail("Path is a directory", "That path is a folder, not a file.", exc) from exc
        except json.JSONDecodeError as exc:
            raise fail("Invalid JSON", "That file is not valid JSON.", exc) from exc
        except UnicodeDecodeError as exc:
            raise fail("Not valid UTF-8 text", "That file is not a readable text file.", exc) from exc
        except OSError as exc:
            raise fail(f"OS error ({exc.strerror or exc})", "A disk error occurred.", exc) from exc

    def exists(self) -> bool:
        """Whether the path exists."""
        return self._path.exists()

    def read_bytes(self) -> bytes:
        """Read raw bytes."""
        with self._guard("read_bytes"):
            return self._path.read_bytes()

    def read_text(self, encoding: str = "utf-8") -> str:
        """Read text."""
        with self._guard("read_text"):
            return self._path.read_text(encoding=encoding)

    def read_json(self) -> Any:
        """Read and parse JSON."""
        with self._guard("read_json"):
            return json.loads(self._path.read_text(encoding="utf-8"))

    def write_bytes(self, data: bytes) -> None:
        """Atomically write bytes (temp file in same folder, then os.replace)."""
        with self._guard("write_bytes"):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=self._path.parent, prefix=".tmp_")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, self._path)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise

    def write_text(self, text: str) -> None:
        """Atomically write UTF-8 text."""
        self.write_bytes(text.encode("utf-8"))

    def write_json(self, data: Any) -> None:
        """Atomically write JSON."""
        self.write_text(json.dumps(data, ensure_ascii=False))

    @staticmethod
    def ensure_dir(path: str | os.PathLike[str]) -> Path:
        """Create a directory (and parents) if missing."""
        folder = Path(path)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise FileOperationError(f"Cannot create directory: {folder}", folder, "mkdir", "A folder could not be created.") from exc
        return folder


def demo() -> None:
    """Show each failure mode being converted (viva demo)."""
    with tempfile.TemporaryDirectory() as tmp:
        bad_json = Path(tmp, "bad.json")
        bad_json.write_text("{not json", encoding="utf-8")
        cases = [
            ("missing file", lambda: SafeFile(Path(tmp, "nope.txt")).read_text()),
            ("directory", lambda: SafeFile(tmp).read_text()),
            ("corrupt JSON", lambda: SafeFile(bad_json).read_json()),
        ]
        for label, action in cases:
            try:
                action()
            except FileOperationError as err:
                print(f"{label:14} -> op={err.op} | {err.user_message}")
        target = SafeFile(Path(tmp, "ok.json"))
        target.write_json({"atomic": True})
        print("atomic write ->", target.read_json())


if __name__ == "__main__":
    demo()
