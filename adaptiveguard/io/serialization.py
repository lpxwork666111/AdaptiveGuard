"""Atomic JSON/JSONL persistence utilities."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_json(path: str | os.PathLike[str], payload: Any, *, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent, default=str)
            handle.write("\n")
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
