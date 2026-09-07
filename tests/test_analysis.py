import json
from copy import deepcopy

from packages.agent import (
    MockProvider,
    TargetAgent,
    TargetAgentToolInterface,
    build_tool_registry,
)
from packages.analysis import (
    CausalAnalysis,
    CausalAnalyzer,
    CausalEdge,
    CausalNode,
    EvidenceReference,
    FailureHypothesis,
)
from packages.discovery.failure import DeterministicVerifier, FailureDetector
from packages.discovery.scenario import (
    Perturbation,
    PerturbationType,
    Scenario,
    ScenarioRunner,
    Trigger,
    TriggerType,
)
from packages.environment.service import EnvironmentService


def create_flagship() -> tuple[EnvironmentService, Scenario, object, str, str]:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    scenario = Scenario(
        scenario_id="flagship-analysis",
        name="Stale billing analysis",
        description="Billing changes during cancellation.",
        task=__import__("packages.agent", fromlist=["AgentTask"]).AgentTask(
            task_id="task-flagship",
            goal="Process Maya's subscription cancellation.",
        ),
        perturbations=(
            Perturbation(
                perturbation_id="billing-drift",
                type=PerturbationType.STATE_CHANGE,
                target=f"billing.{billing.id}.status",
                value="cancelled",
                trigger=Trigger(TriggerType.AGENT_STEP, step=4),
            ),
        ),
    )
    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        MockProvider(
            customer_id=customer.id,
            subscription_id=subscription.id,
            billing_id=billing.id,
        ),
    )
    return environment, scenario, agent, subscription.id, billing.id


def test_causal_models_are_json_serializable() -> None:
    evidence = EvidenceReference("e1", "tool_call", "call1", "A call")
    node = CausalNode("n1", "observation", None, "test", value={"status": "active"})
    edge = CausalEdge("e1", "n1", "n2", "observed", (evidence,))
    hypothesis = FailureHypothesis("h1", "A stale observation", 0.8, (evidence,))
    analysis = CausalAnalysis("a1", "s1", "r1", "failure", "n1", "n2", __import__("packages.analysis.models", fromlist=["CausalGraph"]).CausalGraph((node,), (edge,)), (hypothesis,), (evidence,))

    json.dumps(evidence.to_dict())
    json.dumps(node.to_dict())
    json.dumps(edge.to_dict())
    json.dumps(hypothesis.to_dict())
    json.dumps(analysis.to_dict())


def test_flagship_causal_analysis_extracts_evidence_graph() -> None:
    environment, scenario, agent, subscription_id, billing_id = create_flagship()
    original = environment.snapshot()
    run = ScenarioRunner().run(scenario, environment, agent)
    verification = DeterministicVerifier().verify(run)
    failure = FailureDetector().detect(run, verification)

    analysis = CausalAnalyzer().analyze(run, verification, failure)

    node_types = {node.node_type for node in analysis.graph.nodes}
    relations = {edge.relation for edge in analysis.graph.edges}
    assert "reality_change" in node_types
    assert "observation" in node_types
    assert "action" in node_types
    assert "tool_result" in node_types
    assert "verification" in node_types
    assert "outcome" in node_types
    assert "contradicted" in relations
    assert analysis.first_reality_divergence is not None
    assert analysis.first_consequential_action is not None
    assert analysis.hypotheses
    assert any("no longer current" in hypothesis.statement for hypothesis in analysis.hypotheses)
    assert any(
        node.entity_id == billing_id and node.node_type == "reality_change"
        for node in analysis.graph.nodes
    )
    assert any(
        node.entity_id == subscription_id and node.node_type == "action"
        for node in analysis.graph.nodes
    )
    json.dumps(analysis.to_dict())
    assert environment.snapshot() == original


def test_timeline_is_ordered_and_contains_tool_artifacts() -> None:
    environment, scenario, agent, _, _ = create_flagship()
    run = ScenarioRunner().run(scenario, environment, agent)

    timeline = CausalAnalyzer().extract_timeline(run)

    assert [item["sequence"] for item in timeline] == sorted(item["sequence"] for item in timeline)
    assert any(item["kind"] == "environment_event" for item in timeline)
    assert any(item["kind"] == "tool_call" for item in timeline)
    assert any(item["kind"] == "tool_result" for item in timeline)


def test_insufficient_evidence_returns_no_divergence_or_hypothesis() -> None:
    environment, scenario, agent, _, _ = create_flagship()
    run = ScenarioRunner().run(scenario, environment, agent)
    run.agent_run.trace.entries.clear()
    run.final_environment = {"customers": {}, "subscriptions": {}, "billing_accounts": {}, "events": []}

    analysis = CausalAnalyzer().analyze(run)

    assert analysis.first_reality_divergence is None
    assert analysis.first_consequential_action is None
    assert analysis.hypotheses == ()


def test_analysis_does_not_mutate_run_or_environment() -> None:
    environment, scenario, agent, _, _ = create_flagship()
    run = ScenarioRunner().run(scenario, environment, agent)
    before_run = deepcopy(run.to_dict())
    before_environment = environment.snapshot()

    CausalAnalyzer().analyze(run)

    assert run.to_dict() == before_run
    assert environment.snapshot() == before_environment


def test_causal_timeline_orders_step4_perturbation_after_step1_2_observations() -> None:
    """H3: Causal timeline must correctly order perturbation at step 4 after step 1-2 observations."""
    environment, scenario, agent, subscription_id, billing_id = create_flagship()
    run = ScenarioRunner().run(scenario, environment, agent)

    timeline = CausalAnalyzer().extract_timeline(run)

    # Find sequence numbers for key events
    sequences = {}
    for item in timeline:
        kind = item["kind"]
        if kind == "environment_event":
            event = item["value"]
            if event.get("target") == billing_id and event.get("field") == "status":
                sequences["perturbation_step4"] = item["sequence"]
        elif kind == "tool_result":
            result = item["value"]
            call = next((t for t in timeline if t["kind"] == "tool_call" and t["sequence"] == item["sequence"] - 1), {})
            tool_name = call.get("value", {}).get("tool_name", "")
            if tool_name == "get_customer" and "customer" not in sequences:
                sequences["get_customer"] = item["sequence"]
            elif tool_name == "get_subscription" and "get_subscription" not in sequences:
                sequences["get_subscription"] = item["sequence"]
            elif tool_name == "get_billing_status" and "get_billing_status_1" not in sequences:
                sequences["get_billing_status_1"] = item["sequence"]

    # Perturbation at step 4 should occur after step 1 and 2 observations
    # (MockProvider does get_customer, get_subscription, get_billing_status in steps 1-3)
    # The step 4 perturbation happens before the agent's 4th decision cycle
    assert sequences["get_customer"] < sequences["perturbation_step4"]
    assert sequences["get_subscription"] < sequences["perturbation_step4"]
    # The first get_billing_status is at step 3, before the step 4 perturbation
    assert sequences["get_billing_status_1"] < sequences["perturbation_step4"]

    # Verify the timeline is strictly ordered
    all_sequences = [item["sequence"] for item in timeline]
    assert all_sequences == sorted(all_sequences)
