from __future__ import annotations

from packages.analysis.models import CausalAnalysis
from packages.counterfactual.models import (
    CounterfactualCandidate,
    CounterfactualResult,
    CounterfactualStatus,
)
from packages.discovery.failure.models import FailureRecord, ScenarioOutcome
from packages.discovery.scenario.models import Scenario

from .models import Expectation, Provenance, RegressionTest


class RegressionTestGenerator:
    """Creates declarative regression artifacts only from validated evidence."""

    def generate(
        self,
        scenario: Scenario,
        failure: FailureRecord,
        analysis: CausalAnalysis | None,
        counterfactual: CounterfactualResult,
        candidate: CounterfactualCandidate,
        *,
        discovery_run_id: str | None = None,
    ) -> RegressionTest:
        if failure is None:
            raise ValueError("A failure record is required")
        if counterfactual.status != CounterfactualStatus.VALIDATED:
            raise ValueError("A validated counterfactual result is required")
        if not counterfactual.environment_restored:
            raise ValueError("Counterfactual environment was not restored")
        if candidate.candidate_id != counterfactual.candidate_id:
            raise ValueError("Counterfactual candidate does not match its result")
        if failure.outcome not in {
            ScenarioOutcome.FAILURE,
            ScenarioOutcome.UNSAFE_ACTION,
        }:
            raise ValueError("Regression generation requires a failure or unsafe action")

        return RegressionTest(
            regression_test_id=f"regression_{counterfactual.result_id}",
            name="Reproduce and prevent subscription cancellation failure",
            description="Preserves the stale-billing failure and validated repair.",
            scenario=Scenario.from_dict(scenario.to_dict()),
            expected_original_outcome=Expectation(
                outcome=failure.outcome,
                description="The original unsafe or failed behavior must reproduce.",
            ),
            validated_intervention=candidate.intervention,
            expected_repaired_outcome=Expectation(
                outcome=ScenarioOutcome.SUCCESS,
                description="The validated intervention must resolve the task safely.",
            ),
            provenance=Provenance(
                discovery_run_id=discovery_run_id,
                scenario_id=scenario.scenario_id,
                failure_id=failure.failure_id,
                analysis_id=analysis.analysis_id if analysis else None,
                counterfactual_result_id=counterfactual.result_id,
                intervention_id=candidate.intervention.intervention_id,
            ),
            metadata={"source": "validated_counterfactual"},
        )
