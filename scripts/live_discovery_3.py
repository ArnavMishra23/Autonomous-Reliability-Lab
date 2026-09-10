#!/usr/bin/env python
"""Live DiscoveryEngine smoke test: 3 real Nemotron scenarios.

This script runs exactly THREE controlled scenarios through the existing
DiscoveryEngine using the real Nebius Token Factory + NVIDIA Nemotron provider.

Usage:
    NEBIUS_API_KEY=xxx NEBIUS_MODEL=nvidia/nemotron-3-super-120b-a12b \
    python scripts/live_discovery_3.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Bootstrap: add repository root to sys.path so 'packages' imports works.
# This allows running the script directly from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Load repository-root .env before reading environment variables.
load_dotenv(_REPO_ROOT / ".env")

from packages.agent.target import TargetAgent
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.tools import build_tool_registry
from packages.agent.providers.nebius import NebiusNemotronProvider
from packages.environment.service import EnvironmentService
from packages.discovery.search import (
    DiscoveryEngine,
    RandomSearchStrategy,
    ScenarioGenerator,
    ScenarioParameter,
    ScenarioSpace,
)
from packages.discovery.failure import FailureDetector
from packages.discovery.scenario.runner import ScenarioRunner
from packages.discovery.scenario.models import ScenarioRun
from packages.agent.target import AgentRun


class CapturingRunner:
    """Wrapper that delegates to ScenarioRunner but captures each ScenarioRun."""

    def __init__(self, runner: ScenarioRunner) -> None:
        self._runner = runner
        self.captured_runs: list[ScenarioRun] = []

    def run(self, scenario, environment, agent) -> ScenarioRun:
        scenario_run = self._runner.run(scenario, environment, agent)
        self.captured_runs.append(scenario_run)
        return scenario_run


def extract_scenario_diagnostics(scenario_run: ScenarioRun) -> dict:
    """Extract structured diagnostic data from a ScenarioRun.
    
    Pure function for testing - returns data that print_scenario_diagnostics formats.
    """
    agent_run = scenario_run.agent_run

    diagnostics = {
        "scenario_id": scenario_run.scenario_id,
        "run_id": scenario_run.run_id,
        "scenario_status": scenario_run.status.value,
        "scenario_error": scenario_run.error,
        "applied_perturbations": [
            {
                "perturbation_id": p.perturbation_id,
                "target": p.target,
                "value": p.value,
                "step": p.step,
                "event_id": p.event_id,
            }
            for p in scenario_run.applied_perturbations
        ],
        "agent_run": None,
        "final_subscription_count": len(scenario_run.final_environment.get("subscriptions", {})),
        "final_billing_count": len(scenario_run.final_environment.get("billing_accounts", {})),
    }

    if agent_run is not None:
        tool_sequence = []
        for entry in agent_run.trace.entries:
            call = entry.call
            result = entry.result
            tool_sequence.append({
                "tool_name": call.tool_name,
                "arguments": dict(call.arguments),
                "success": result.success,
                "error": result.error,
                "error_code": result.error_code.value if result.error_code else None,
            })

        diagnostics["agent_run"] = {
            "status": agent_run.status.value,
            "error": agent_run.error,
            "steps": len(agent_run.trace.entries),
            "final_result": agent_run.final_result,
            "tool_sequence": tool_sequence,
        }

    return diagnostics


def print_scenario_diagnostics(index: int, scenario_run: ScenarioRun) -> None:
    """Print concise diagnostic info for one scenario run."""
    diag = extract_scenario_diagnostics(scenario_run)

    print(f"\n  Scenario {index}:")
    print(f"    Scenario ID: {diag['scenario_id']}")
    print(f"    Run ID: {diag['run_id']}")
    print(f"    Scenario Status: {diag['scenario_status']}")
    print(f"    Scenario Error: {diag['scenario_error'] if diag['scenario_error'] else 'None'}")

    if diag["agent_run"] is not None:
        ar = diag["agent_run"]
        print(f"    Agent Run Status: {ar['status']}")
        print(f"    Agent Run Error: {ar['error'] if ar['error'] else 'None'}")
        print(f"    Agent Steps: {ar['steps']}")
        print(f"    Agent Final Result: {ar['final_result'][:200] if ar['final_result'] else 'None'}")

        print(f"    Tool Sequence:")
        for j, tool in enumerate(ar["tool_sequence"], 1):
            print(f"      {j}. {tool['tool_name']}({tool['arguments']})")
            print(f"         -> success={tool['success']}, error={tool['error']}, error_code={tool['error_code']}")
    else:
        print(f"    Agent Run: None (scenario failed before agent execution)")

    # Applied perturbations
    if diag["applied_perturbations"]:
        print(f"    Applied Perturbations:")
        for p in diag["applied_perturbations"]:
            print(f"      {p['perturbation_id']}: target={p['target']}, value={p['value']}, step={p['step']}, event_id={p['event_id']}")
    else:
        print(f"    Applied Perturbations: None")

    # Environment restored check
    print(f"    Final Subscription Count: {diag['final_subscription_count']}")
    print(f"    Final Billing Count: {diag['final_billing_count']}")


class FixedThreeStrategy:
    """Deterministic strategy that picks exactly three predefined candidates."""

    def __init__(self, target_candidates: list) -> None:
        self._target_candidates = target_candidates
        self._index = 0

    def choose_next(
        self,
        candidates: list,
        evaluated: set = frozenset(),
        history: list = None,
    ):
        """Return the next target candidate that hasn't been evaluated yet."""
        # Build a lookup for fast matching
        candidate_map = {c.candidate_id: c for c in candidates}

        while self._index < len(self._target_candidates):
            target_id = self._target_candidates[self._index]
            self._index += 1
            if target_id not in evaluated and target_id in candidate_map:
                return candidate_map[target_id]
        return None


def create_space() -> ScenarioSpace:
    """Create the ScenarioSpace matching test definitions."""
    return ScenarioSpace(
        parameters=(
            ScenarioParameter("billing_status", "discrete", ("active", "cancelled")),
            ScenarioParameter("trigger_step", "discrete", (1, 2, 3, 4, 5)),
        )
    )


def main() -> int:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY not set. Skipping live discovery smoke test.")
        print("Set NEBIUS_API_KEY and NEBIUS_MODEL to run.")
        return 0

    model = os.getenv("NEBIUS_MODEL")
    if not model:
        print("NEBIUS_MODEL not set. Please specify a model identifier.")
        return 1

    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

    print(f"Live DiscoveryEngine smoke test with:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  Scenarios: 3 fixed candidates")

    # Set up the Customer Operations environment
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    print(f"\nEnvironment setup:")
    print(f"  Customer: {customer.id} ({customer.name})")
    print(f"  Subscription: {subscription.id} (plan: {subscription.plan})")
    print(f"  Billing: {billing.id} (status: {billing.status.value})")

    # Build tool registry and agent
    tool_registry = build_tool_registry(environment)
    tool_interface = TargetAgentToolInterface(tool_registry)

    provider = NebiusNemotronProvider(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=60.0,
    )

    agent = TargetAgent(tool_interface, provider, max_steps=10)

    # Create the three fixed candidates we want to evaluate
    space = create_space()
    all_candidates = space.enumerate_candidates()

    # Find the exact three candidates by their parameters
    target_candidate_ids = []
    for c in all_candidates:
        params = c.parameter_values
        if (params.get("billing_status") == "active" and params.get("trigger_step") == 1):
            target_candidate_ids.append(c.candidate_id)
        elif (params.get("billing_status") == "cancelled" and params.get("trigger_step") == 1):
            target_candidate_ids.append(c.candidate_id)
        elif (params.get("billing_status") == "cancelled" and params.get("trigger_step") == 3):
            target_candidate_ids.append(c.candidate_id)

    print(f"\nTarget candidates:")
    for cid in target_candidate_ids:
        c = next(c for c in all_candidates if c.candidate_id == cid)
        print(f"  {cid}: billing_status={c.parameter_values['billing_status']}, trigger_step={c.parameter_values['trigger_step']}")

    if len(target_candidate_ids) != 3:
        print(f"ERROR: Expected 3 target candidates, found {len(target_candidate_ids)}")
        return 1

    # Use fixed-order strategy for exactly these three
    strategy = FixedThreeStrategy(target_candidate_ids)

    # Create generator, runner, detector
    generator = ScenarioGenerator(
        customer_id=customer.id,
        subscription_id=subscription.id,
        billing_id=billing.id,
    )
    base_runner = ScenarioRunner()
    capturing_runner = CapturingRunner(base_runner)
    detector = FailureDetector()
    engine = DiscoveryEngine()

    # Run discovery with max_scenarios=3
    print(f"\nStarting DiscoveryEngine run (max_scenarios=3)...")
    try:
        result = engine.run(
            space=space,
            generator=generator,
            strategy=strategy,
            runner=capturing_runner,
            environment=environment,
            agent=agent,
            failure_detector=detector,
            max_scenarios=3,
        )
    except Exception as exc:
        print(f"\nDiscoveryEngine FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    # Print results
    print(f"\n=== DiscoveryResult ===")
    print(f"  Discovery Run ID: {result.discovery_run_id}")
    print(f"  Status: {result.status.value}")
    print(f"  Scenarios Evaluated: {result.scenarios_evaluated}")
    print(f"  Outcome Counts: {dict(result.outcome_counts)}")
    print(f"  Unique Failures: {result.unique_failure_count}")
    print(f"  Failure IDs: {list(result.failure_ids)}")
    print(f"  Started: {result.started_at}")
    print(f"  Finished: {result.finished_at}")
    if result.error:
        print(f"  Error: {result.error}")

    # Print detailed diagnostic output for each captured scenario run
    print(f"\n=== Per-Scenario Diagnostic Details ===")
    for i, scenario_run in enumerate(capturing_runner.captured_runs, 1):
        print_scenario_diagnostics(i, scenario_run)

    # Verify serialization
    try:
        result_dict = result.to_dict()
        json.dumps(result_dict)
        print(f"\nDiscoveryResult JSON serialization: OK")
    except Exception as exc:
        print(f"\nDiscoveryResult JSON serialization FAILED: {exc}")
        return 1

    # Persist DiscoveryResult as JSON artifact
    artifact_path = write_discovery_artifact(result_dict, _REPO_ROOT)
    if artifact_path:
        print(f"\nArtifact:")
        print(f"  {artifact_path.relative_to(_REPO_ROOT)}")

    # Verify environment restored (should be back to original state)
    final_env = environment.snapshot()
    print(f"\n=== Final Environment State (after all scenarios) ===")
    for sub in final_env.subscriptions.values():
        print(f"  Subscription {sub.id}: {sub.status.value}")
    for bill in final_env.billing_accounts.values():
        print(f"  Billing {bill.id}: {bill.status.value}")

    # Check if original entities are still active
    sub_ok = final_env.subscriptions.get(subscription.id) is not None
    bill_ok = final_env.billing_accounts.get(billing.id) is not None
    if sub_ok and bill_ok:
        print("\nEnvironment restoration: OK")
    else:
        print("\nEnvironment restoration: FAILED - entities missing")

    print("\nLive DiscoveryEngine smoke test completed.")
    return 0


def print_scenario_diagnostics(index: int, scenario_run: ScenarioRun) -> None:
    """Print concise diagnostic info for one scenario run."""
    agent_run = scenario_run.agent_run

    print(f"\n  Scenario {index}:")
    print(f"    Scenario ID: {scenario_run.scenario_id}")
    print(f"    Run ID: {scenario_run.run_id}")
    print(f"    Scenario Status: {scenario_run.status.value}")
    print(f"    Scenario Error: {scenario_run.error if scenario_run.error else 'None'}")

    # Extract candidate params from scenario metadata
    meta = scenario_run.applied_perturbations
    # The scenario metadata is not directly on ScenarioRun, but we can infer from perturbations
    # For now, we'll rely on the order matching our target candidates

    if agent_run is not None:
        print(f"    Agent Run Status: {agent_run.status.value}")
        print(f"    Agent Run Error: {agent_run.error if agent_run.error else 'None'}")
        print(f"    Agent Steps: {len(agent_run.trace.entries)}")
        print(f"    Agent Final Result: {agent_run.final_result[:200] if agent_run.final_result else 'None'}")

        print(f"    Tool Sequence:")
        for j, entry in enumerate(agent_run.trace.entries, 1):
            call = entry.call
            result = entry.result
            print(f"      {j}. {call.tool_name}({call.arguments})")
            print(f"         -> success={result.success}, error={result.error}, error_code={result.error_code}")
    else:
        print(f"    Agent Run: None (scenario failed before agent execution)")

    # Applied perturbations
    if scenario_run.applied_perturbations:
        print(f"    Applied Perturbations:")
        for p in scenario_run.applied_perturbations:
            print(f"      {p.perturbation_id}: target={p.target}, value={p.value}, step={p.step}, event_id={p.event_id}")
    else:
        print(f"    Applied Perturbations: None")

    # Environment restored check
    final_subs = scenario_run.final_environment.get("subscriptions", {})
    final_bills = scenario_run.final_environment.get("billing_accounts", {})
    print(f"    Final Subscription Count: {len(final_subs)}")
    print(f"    Final Billing Count: {len(final_bills)}")


def write_discovery_artifact(result_dict: dict, repo_root: Path) -> Path | None:
    """Write DiscoveryResult to a timestamped JSON file atomically."""
    try:
        # Create artifacts directory
        artifact_dir = repo_root / "artifacts" / "live-discovery"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: discovery_<timestamp>Z_<run_id>.json
        # Use the discovery_run_id from result if available
        run_id = result_dict.get("discovery_run_id", "unknown")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"discovery_{timestamp}_{run_id}.json"
        artifact_path = artifact_dir / filename

        # Atomic write: write to temp file then rename
        temp_path = artifact_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        temp_path.replace(artifact_path)

        return artifact_path
    except Exception as exc:
        print(f"WARNING: Failed to write discovery artifact: {exc}")
        return None


if __name__ == "__main__":
    sys.exit(main())