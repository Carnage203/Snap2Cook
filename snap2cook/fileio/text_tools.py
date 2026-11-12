# SYLLABUS Q6 - read a file; reverse words / lines; strip whitespace
"""FileTextProcessor: the four text transformations from the syllabus question."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from snap2cook.fileio.safe_io import SafeFile


class FileTextProcessor:
    """Holds text (usually read from a file) and transforms it."""

    def __init__(self, text: str) -> None:
        self._text = text

    @classmethod
    def from_file(cls, path: str | Path) -> "FileTextProcessor":
        """Read a file safely and wrap its content."""
        return cls(SafeFile(path).read_text())

    @property
    def text(self) -> str:
        """The original text."""
        return self._text

    def __len__(self) -> int:
        return len(self._text)

    def __str__(self) -> str:
        return self._text

    def reverse_each_word(self) -> str:
        """Reverse the letters of every word, keeping spacing and lines."""
        return re.sub(r"\S+", lambda match: match.group()[::-1], self._text)

    def reverse_word_order_per_line(self) -> str:
        """Reverse the word order within each line ('Python Programming' -> 'Programming Python')."""
        return "\n".join(" ".join(reversed(line.split())) for line in self._text.splitlines())

    def reverse_line_order(self) -> str:
        """Reverse the order of the lines."""
        return "\n".join(reversed(self._text.splitlines()))

    def strip_whitespace(self) -> str:
        """Return the content with every whitespace character removed."""
        return "".join(self._text.split())


def demo() -> None:
    """Run all four operations on the syllabus sample (viva demo)."""
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp, "sample.txt")
        SafeFile(sample).write_text("Python Programming\nis fun to learn\n")
        proc = FileTextProcessor.from_file(sample)
        print("original               :", repr(proc.text))
        print("(i)  each word reversed:", repr(proc.reverse_each_word()))
        print("(ii) word order / line :", repr(proc.reverse_word_order_per_line()))
        print("     line order        :", repr(proc.reverse_line_order()))
        print("(iii) no whitespace    :", repr(proc.strip_whitespace()))


if __name__ == "__main__":
    demo()
