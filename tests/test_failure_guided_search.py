import json

from packages.agent.target import MockProvider, TargetAgent
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.tools import build_tool_registry
from packages.discovery.failure import FailureDetector
from packages.discovery.scenario import ScenarioRunner
from packages.discovery.search import (
    DiscoveryEngine,
    DiscoveryStatus,
    FailureGuidedSearchStrategy,
    RandomSearchStrategy,
    ScenarioCandidate,
    ScenarioGenerator,
    ScenarioParameter,
    ScenarioSpace,
    SearchDecision,
    candidate_distance,
)
from packages.environment.service import EnvironmentService


def candidate(candidate_id: str, billing_status: str, trigger_step: int) -> ScenarioCandidate:
    return ScenarioCandidate(
        candidate_id,
        {"billing_status": billing_status, "trigger_step": trigger_step},
    )


def record(candidate_value: ScenarioCandidate, outcome: str) -> dict[str, object]:
    return {"candidate": candidate_value.to_dict(), "outcome": outcome}


def create_environment() -> tuple[EnvironmentService, str, str, str]:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    return environment, customer.id, subscription.id, billing.id


def create_space() -> ScenarioSpace:
    return ScenarioSpace(
        parameters=(
            ScenarioParameter("billing_status", "discrete", ("active", "cancelled")),
            ScenarioParameter("trigger_step", "discrete", (1, 2, 3, 4, 5)),
        )
    )


def test_candidate_distance_counts_differing_parameters() -> None:
    assert candidate_distance(candidate("a", "cancelled", 4), candidate("b", "cancelled", 5)) == 1
    assert candidate_distance(candidate("a", "cancelled", 4), candidate("b", "active", 1)) == 2


def test_outcome_weights_prioritize_unsafe_over_failure() -> None:
    strategy = FailureGuidedSearchStrategy()
    candidates = (
        candidate("near_failure", "active", 1),
        candidate("near_unsafe", "cancelled", 4),
    )
    history = (
        record(candidate("failure", "cancelled", 3), "failure"),
        record(candidate("unsafe", "cancelled", 4), "unsafe_action"),
    )

    selected = strategy.choose_next(candidates, frozenset({"failure", "unsafe"}), history)

    assert selected is not None
    assert selected.candidate_id == "near_unsafe"


def test_interesting_neighborhood_and_degraded_guidance() -> None:
    strategy = FailureGuidedSearchStrategy()
    interesting = candidate("degraded", "cancelled", 3)
    near = candidate("near", "cancelled", 4)
    far = candidate("far", "active", 1)

    selected = strategy.choose_next(
        (far, near),
        frozenset({interesting.candidate_id}),
        (record(interesting, "degraded"),),
    )

    assert selected == near
    assert strategy.last_decision is not None
    assert any(reason.startswith("near_degraded") for reason in strategy.last_decision.reasons)


def test_boundary_detection_and_parameter_relevance() -> None:
    strategy = FailureGuidedSearchStrategy()
    success = candidate("success", "active", 3)
    failure = candidate("failure", "cancelled", 3)
    boundary_neighbor = candidate("neighbor", "cancelled", 4)
    far = candidate("far", "active", 1)

    selected = strategy.choose_next(
        (far, boundary_neighbor),
        frozenset({"success", "failure"}),
        (record(success, "success"), record(failure, "failure")),
    )

    assert selected == boundary_neighbor
    assert strategy.last_decision is not None
    assert "boundary_bonus" in strategy.last_decision.reasons
    assert "relevant_parameter_billing_status" in strategy.last_decision.reasons


def test_interaction_and_novelty_contributions() -> None:
    strategy = FailureGuidedSearchStrategy()
    interesting = ScenarioCandidate(
        "interesting",
        {"billing_status": "cancelled", "trigger_step": 4, "mode": "strict"},
    )
    same_region = ScenarioCandidate(
        "same",
        {"billing_status": "cancelled", "trigger_step": 5, "mode": "strict"},
    )
    novel = ScenarioCandidate(
        "novel",
        {"billing_status": "active", "trigger_step": 1, "mode": "relaxed"},
    )

    selected = strategy.choose_next(
        (novel, same_region),
        frozenset({"interesting"}),
        (record(interesting, "unsafe_action"),),
    )

    assert selected == same_region
    assert strategy.last_decision is not None
    assert "interaction_bonus" in strategy.last_decision.reasons
    assert "novelty_bonus" in strategy.last_decision.reasons


def test_deterministic_tie_breaking_and_no_history_behavior() -> None:
    candidates = (
        candidate("z", "active", 1),
        candidate("a", "active", 1),
    )
    first = FailureGuidedSearchStrategy()
    second = FailureGuidedSearchStrategy()

    assert first.choose_next(candidates) == second.choose_next(candidates)
    assert first.last_decision is not None
    assert first.last_decision.candidate_id == "a"


def test_strategy_returns_none_when_all_candidates_evaluated() -> None:
    strategy = FailureGuidedSearchStrategy()
    candidates = (candidate("a", "active", 1),)

    assert strategy.choose_next(candidates, frozenset({"a"})) is None


def test_search_decision_is_serializable() -> None:
    decision = SearchDecision(
        "candidate_a",
        3.5,
        ("novelty_bonus",),
        {"novelty": 0.25, "total": 0.25},
    )

    assert json.dumps(decision.to_dict())


def test_history_changes_score_and_components_are_visible() -> None:
    strategy = FailureGuidedSearchStrategy()
    near = candidate("near", "cancelled", 3)
    far = candidate("far", "active", 1)
    interesting = candidate("interesting", "cancelled", 2)

    strategy.choose_next((near,), frozenset(), ())
    empty_score = strategy.last_decision.score
    strategy.choose_next(
        (near, far),
        frozenset({interesting.candidate_id}),
        (record(interesting, "unsafe_action"),),
    )

    assert strategy.last_decision is not None
    assert strategy.last_decision.candidate_id == "near"
    assert strategy.last_decision.score > empty_score
    assert strategy.last_decision.score_components["neighborhood"] > 0
    assert strategy.last_decision.score_components["novelty"] == 0.25


def test_engine_supports_failure_guided_strategy_and_records_decisions() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    space = create_space()
    generator = ScenarioGenerator(
        customer_id=customer_id,
        subscription_id=subscription_id,
        billing_id=billing_id,
    )
    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        MockProvider(
            customer_id=customer_id,
            subscription_id=subscription_id,
            billing_id=billing_id,
        ),
    )
    result = DiscoveryEngine().run(
        space,
        generator,
        FailureGuidedSearchStrategy(),
        ScenarioRunner(),
        environment,
        agent,
        FailureDetector(),
        max_scenarios=10,
    )

    assert result.status == DiscoveryStatus.MAX_SCENARIOS_REACHED
    assert result.scenarios_evaluated == 10
    assert len(result.search_decisions) == 10
    assert len({item["candidate"]["candidate_id"] for item in result.results}) == 10
    assert json.dumps(result.to_dict())


def test_random_and_guided_runs_are_deterministic_and_budgeted() -> None:
    def run(strategy: object) -> tuple[tuple[str, ...], int]:
        environment, customer_id, subscription_id, billing_id = create_environment()
        generator = ScenarioGenerator(
            customer_id=customer_id,
            subscription_id=subscription_id,
            billing_id=billing_id,
        )
        agent = TargetAgent(
            TargetAgentToolInterface(build_tool_registry(environment)),
            MockProvider(
                customer_id=customer_id,
                subscription_id=subscription_id,
                billing_id=billing_id,
            ),
        )
        result = DiscoveryEngine().run(
            create_space(),
            generator,
            strategy,
            ScenarioRunner(),
            environment,
            agent,
            FailureDetector(),
            max_scenarios=5,
        )
        return tuple(item["candidate"]["candidate_id"] for item in result.results), result.scenarios_evaluated

    random_first = run(RandomSearchStrategy(seed=12))
    random_second = run(RandomSearchStrategy(seed=12))
    guided_first = run(FailureGuidedSearchStrategy())
    guided_second = run(FailureGuidedSearchStrategy())

    assert random_first == random_second
    assert guided_first == guided_second
    assert random_first[1] == guided_first[1] == 5
    assert len(set(random_first[0])) == 5
    assert len(set(guided_first[0])) == 5
