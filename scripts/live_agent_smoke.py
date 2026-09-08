#!/usr/bin/env python
"""Live TargetAgent smoke test for end-to-end Nebius integration.

This script only runs when NEBIUS_API_KEY is set in the environment.
It runs a single real TargetAgent task through Nebius against the
Customer Operations environment.

Usage:
    NEBIUS_API_KEY=xxx NEBIUS_MODEL=... python scripts/live_agent_smoke.py
"""

from __future__ import annotations

import os
import sys

from packages.agent.target import TargetAgent, AgentTask
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.tools import build_tool_registry
from packages.agent.providers.nebius import NebiusNemotronProvider
from packages.environment.service import EnvironmentService


def main() -> int:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY not set. Skipping live agent smoke test.")
        print("Set NEBIUS_API_KEY and NEBIUS_MODEL to run.")
        return 0

    model = os.getenv("NEBIUS_MODEL")
    if not model:
        print("NEBIUS_MODEL not set. Please specify a model identifier.")
        return 1

    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

    print(f"Live TargetAgent smoke test with:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  API Key: {'*' * (len(api_key) - 4)}{api_key[-4:] if len(api_key) > 4 else '****'}")

    try:
        # Set up the Customer Operations environment
        environment = EnvironmentService()
        customer = environment.create_customer(name="Maya", email="maya@example.com")
        subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
        billing = environment.create_billing_account(subscription_id=subscription.id)

        print(f"\nEnvironment setup:")
        print(f"  Customer: {customer.id} ({customer.name})")
        print(f"  Subscription: {subscription.id} (plan: {subscription.plan})")
        print(f"  Billing: {billing.id} (status: {billing.status.value})")

        # Build tool registry
        tool_registry = build_tool_registry(environment)
        tool_interface = TargetAgentToolInterface(tool_registry)

        # Create provider
        provider = NebiusNemotronProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=60.0,  # Longer timeout for agent runs
        )

        # Create agent
        agent = TargetAgent(tool_interface, provider, max_steps=10)

        # Run the cancellation task
        task = AgentTask(
            task_id="live-cancellation",
            goal="Process Maya's subscription cancellation.",
            context={"customer_id": customer.id},
        )

        print(f"\nRunning agent task: {task.goal}")
        print(f"Max steps: 10")

        run = agent.run(task)

        print(f"\nAgentRun result:")
        print(f"  Run ID: {run.run_id}")
        print(f"  Status: {run.status.value}")
        print(f"  Steps: {len(run.trace.entries)}")
        print(f"  Final result: {run.final_result[:200] if run.final_result else 'None'}")
        print(f"  Error: {run.error}")

        print(f"\nTool call sequence:")
        for i, entry in enumerate(run.trace.entries, 1):
            result = entry.result
            print(f"  {i}. {entry.call.tool_name}({entry.call.arguments})")
            print(f"     -> success={result.success}, error={result.error}")

        # Final environment state
        final_env = environment.snapshot()
        print(f"\nFinal environment state:")
        for sub in final_env.subscriptions.values():
            print(f"  Subscription {sub.id}: {sub.status.value}")
        for bill in final_env.billing_accounts.values():
            print(f"  Billing {bill.id}: {bill.status.value}")

        if run.status.value == "succeeded":
            print("\n✓ Live TargetAgent smoke test PASSED")
            return 0
        else:
            print(f"\n✗ Live TargetAgent smoke test FAILED: {run.error}")
            return 1

    except Exception as exc:
        print(f"\n✗ Live TargetAgent smoke test FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())