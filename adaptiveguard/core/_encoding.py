"""Internal encoding helpers for structured controller records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any


def encode_value(value: Any) -> Any:
    """Convert nested runtime values into JSON-compatible Python values."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: encode_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return encode_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return {key: encode_value(item) for key, item in value.__dict__.items()}
    return value
