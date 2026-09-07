from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class IdProvider(Protocol):
    def generate(self, prefix: str = "") -> str:
        """Generate a new unique identifier."""
        ...


class Clock(Protocol):
    def now(self) -> datetime:
        """Return the current timezone-aware UTC timestamp."""
        ...


class DefaultIdProvider:
    """Default implementation using uuid4."""

    def __init__(self) -> None:
        from uuid import uuid4
        self._uuid4 = uuid4

    def generate(self, prefix: str = "") -> str:
        return f"{prefix}{self._uuid4().hex[:8]}"


class DefaultClock:
    """Default implementation using datetime.now(timezone.utc)."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class DeterministicIdProvider:
    """Deterministic ID provider for testing/replay.

    Generates sequential IDs with a given prefix and namespace.
    """

    def __init__(self, *, namespace: str = "default") -> None:
        self._namespace = namespace
        self._counters: dict[str, int] = {}

    def generate(self, prefix: str = "") -> str:
        key = f"{self._namespace}:{prefix}"
        self._counters[key] = self._counters.get(key, 0) + 1
        return f"{prefix}{key}_{self._counters[key]:08d}"

    def reset(self) -> None:
        self._counters.clear()


class DeterministicClock:
    """Deterministic clock for testing/replay.

    Returns a fixed base time plus an incrementing offset.
    """

    def __init__(self, *, base: datetime | None = None, tick_seconds: float = 1.0) -> None:
        self._base = base or datetime(2000, 1, 1, tzinfo=timezone.utc)
        self._tick = tick_seconds
        self._counter = 0

    def now(self) -> datetime:
        from datetime import timedelta
        result = self._base + timedelta(seconds=self._counter * self._tick)
        self._counter += 1
        return result

    def reset(self) -> None:
        self._counter = 0