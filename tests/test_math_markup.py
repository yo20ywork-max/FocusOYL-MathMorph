"""Guard current documentation against a macro rejected by GitHub's renderer.

This static regression check does not replace live browser rendering tests.
Historical source and experimental evidence are intentionally excluded.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MathMarkupTests(unittest.TestCase):
    def test_current_docs_avoid_rejected_operator_macro(self):
        paths = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
        self.assertGreater(len(paths), 1)
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"\\operatorname\b", text))


if __name__ == "__main__":
    unittest.main()
