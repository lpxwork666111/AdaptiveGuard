"""Run manifest generation for deterministic bookkeeping."""

from __future__ import annotations

import platform
import random
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import import_module
from typing import Any


def make_manifest(config: Mapping[str, Any], seed: int | None = None) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "config": dict(config),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        np = import_module("numpy")
        np.random.seed(seed)
    except ImportError:
        pass
