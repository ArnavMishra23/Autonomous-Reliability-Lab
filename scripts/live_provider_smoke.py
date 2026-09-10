#!/usr/bin/env python
"""Live provider smoke test for NebiusNemotronProvider.

This script only runs when NEBIUS_API_KEY is set in the environment.
It makes a single real request to the Nebius API and prints safe diagnostics.

Usage:
    NEBIUS_API_KEY=xxx NEBIUS_MODEL=... python scripts/live_provider_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Bootstrap: add repository root to sys.path so 'packages' imports work.
# This allows running the script directly from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Load repository-root .env before reading environment variables.
load_dotenv(_REPO_ROOT / ".env")

from packages.agent.target import AgentTask
from packages.agent.providers.nebius import NebiusNemotronProvider


def main() -> int:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY not set. Skipping live provider smoke test.")
        print("Set NEBIUS_API_KEY and NEBIUS_MODEL to run.")
        return 0

    model = os.getenv("NEBIUS_MODEL")
    if not model:
        print("NEBIUS_MODEL not set. Please specify a model identifier.")
        return 1

    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

    print(f"Testing NebiusNemotronProvider with:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")

    try:
        provider = NebiusNemotronProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=30.0,
        )

        # Simple test request without tools
        task = AgentTask(
            task_id="smoke-test",
            goal="Reply with a short confirmation that you are working.",
            context={},
        )

        response = provider.complete(
            system_instructions="You are a helpful assistant. Keep responses brief.",
            task=task,
            available_tools=(),
            previous_results=(),
        )

        print(f"\nResponse:")
        print(f"  Completed: {response.completed}")
        print(f"  Content: {response.content[:200]}")
        if response.tool_calls:
            print(f"  Tool calls: {[(tc.tool_name, tc.arguments) for tc in response.tool_calls]}")

        if response.completed and response.content:
            print("\nLive provider smoke test PASSED")
            return 0
        else:
            print("\nLive provider smoke test FAILED: Unexpected response")
            return 1

    except Exception as exc:
        print(f"\nLive provider smoke test FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())