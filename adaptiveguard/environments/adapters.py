"""Small helpers for converting arbitrary observations to prompt text."""

from __future__ import annotations

import json
from typing import Any


def observation_to_text(observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    if hasattr(observation, "to_dict"):
        observation = observation.to_dict()
    try:
        return json.dumps(observation, ensure_ascii=False, default=str, sort_keys=True)
    except TypeError:
        return str(observation)
