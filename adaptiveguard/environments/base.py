"""Environment protocol for interactive agent control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ..core.types import EnvironmentTransition


class InteractiveEnvironment(ABC):
    score_range: float = 100.0

    @abstractmethod
    def reset(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> EnvironmentTransition:
        raise NotImplementedError

    def available_actions(self) -> Sequence[str]:
        return ()

    def close(self) -> None:
        return None

    def __enter__(self) -> InteractiveEnvironment:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
