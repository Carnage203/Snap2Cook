"""Print every syllabus demo (for the viva):  python -m assignments.run_all"""
from __future__ import annotations

from snap2cook import exceptions
from snap2cook.analytics import series
from snap2cook.fileio import safe_io, text_tools

DEMOS = (
    ("Q6  - Text processing (fileio/text_tools.py)", text_tools.demo),
    ("Q8  - Safe file handling (fileio/safe_io.py)", safe_io.demo),
    ("Q9  - User-defined exceptions (exceptions.py)", exceptions.demo),
    ("Q10 - Pandas Series with labels (analytics/series.py)", series.demo),
)


def main() -> None:
    """Run the four demos in order."""
    for title, demo in DEMOS:
        print(f"\n{'=' * 8} {title} {'=' * 8}")
        demo()


if __name__ == "__main__":
    main()
