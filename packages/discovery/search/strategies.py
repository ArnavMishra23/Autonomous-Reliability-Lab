from __future__ import annotations

import random
from typing import Iterable, Protocol

from .models import ScenarioCandidate


class SearchStrategy(Protocol):
    def choose_next(
        self,
        candidates: Iterable[ScenarioCandidate],
        evaluated: frozenset[str] = frozenset(),
    ) -> ScenarioCandidate | None:
        """Choose one unseen candidate or return None when exhausted."""


class RandomSearchStrategy:
    """Seeded random permutation over a finite candidate space."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._random = random.Random(seed)
        self._remaining: list[ScenarioCandidate] | None = None

    def choose_next(
        self,
        candidates: Iterable[ScenarioCandidate],
        evaluated: frozenset[str] = frozenset(),
    ) -> ScenarioCandidate | None:
        if self._remaining is None:
            self._remaining = [
                candidate
                for candidate in candidates
                if candidate.candidate_id not in evaluated
            ]
            self._random.shuffle(self._remaining)
        else:
            self._remaining = [
                candidate
                for candidate in self._remaining
                if candidate.candidate_id not in evaluated
            ]
        if not self._remaining:
            return None
        return self._remaining.pop()
