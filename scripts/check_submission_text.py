"""Check distributable text for result-reporting or paper-first wording."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("README.md", "docs", "adaptiveguard", "configs", "examples", "tests")
FORBIDDEN_PATTERNS = (
    r"\bpost[- ]?hoc\b",
    r"\bhistorical (?:result|score|run|artifact)\b",
    r"\breported (?:result|score|number|value)\b",
    r"\b(?:experiment|experimental|benchmark) results?\b",
    r"\bbaseline results?\b",
    r"\bpaper[- ]to[- ]code\b",
    r"\bcamera[- ]ready\b",
    r"\bupon publication\b",
    r"\breproduc(?:e|ed|ibility|ing|tion)\b",
    r"\breplicat(?:e|ed|ion|ing)\b",
    r"论文完成后",
    r"实验结果",
    r"复现",
)


def iter_text_files() -> list[Path]:
    files: list[Path] = []
    for relative in SCAN_ROOTS:
        path = ROOT / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
    return sorted(files)


def main() -> int:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_PATTERNS]
    violations: list[str] = []
    for path in iter_text_files():
        if path.suffix not in {".md", ".py", ".yaml", ".yml", ".json", ".txt", ".toml"}:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(pattern.search(line) for pattern in compiled):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")
    if violations:
        print("Forbidden submission wording detected:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("Submission text check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
