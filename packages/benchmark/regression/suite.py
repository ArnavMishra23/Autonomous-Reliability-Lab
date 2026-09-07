from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from packages.agent.target import TargetAgent
from packages.environment.service import EnvironmentService
from packages.environment.state import EnvironmentState

from .models import RegressionRunResult, RegressionStatus, RegressionTest
from .runner import RegressionTestRunner


@dataclass(frozen=True, slots=True)
class RegressionSuiteResult:
    results: tuple[RegressionRunResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"results": [result.to_dict() for result in self.results]}


class RegressionSuite:
    def __init__(self, *, runner: RegressionTestRunner | None = None) -> None:
        self._tests: dict[str, RegressionTest] = {}
        self._results: dict[str, tuple[RegressionRunResult, ...]] = {}
        self._runner = runner or RegressionTestRunner()

    def add_test(self, regression_test: RegressionTest) -> None:
        if regression_test.regression_test_id in self._tests:
            raise ValueError(
                f"Regression test already exists: {regression_test.regression_test_id}"
            )
        self._tests[regression_test.regression_test_id] = RegressionTest.from_dict(
            regression_test.to_dict()
        )

    def list_tests(self) -> tuple[RegressionTest, ...]:
        return tuple(
            RegressionTest.from_dict(test.to_dict()) for test in self._tests.values()
        )

    def get_result(self, regression_test_id: str) -> tuple[RegressionRunResult, ...]:
        return tuple(self._results.get(regression_test_id, ()))

    def run_all(
        self,
        environment: EnvironmentService,
        agent: TargetAgent,
        starting_state: EnvironmentState,
    ) -> RegressionSuiteResult:
        results: list[RegressionRunResult] = []
        for regression_test in self._tests.values():
            original = self._runner.run_original(
                regression_test, environment, agent, starting_state
            )
            repaired = self._runner.run_repaired(
                regression_test, environment, agent, starting_state
            )
            pair = (original, repaired)
            self._results[regression_test.regression_test_id] = pair
            results.extend(pair)
        return RegressionSuiteResult(results=tuple(results))
