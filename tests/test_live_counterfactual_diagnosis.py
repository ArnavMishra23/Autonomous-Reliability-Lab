"""Offline diagnosis of live counterfactual early-stop behavior."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import MagicMock

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from packages.agent.target import AgentRun, AgentRunStatus, AgentTask, ModelProvider, ModelResponse, ToolRequest, TargetAgent, ToolCall, ToolResult, ToolTrace, ToolTraceEntry, generate_id
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.tools import build_tool_registry
from packages.agent.models import ToolDefinition
from packages.environment.service import EnvironmentService


class EarlyStopProvider:
    """Provider that returns completed=True after get_customer (mimics observed Nemotron behavior)."""
    
    def __init__(self, stop_after_tool: str = "get_customer"):
        self.stop_after_tool = stop_after_tool
        self.call_count = 0
        self.customer_id = "cust_test"
        self.subscription_id = "sub_test"
        self.billing_id = "bill_test"
    
    def complete(
        self,
        *,
        system_instructions: str,
        task: AgentTask,
        available_tools: Sequence[ToolDefinition],
        previous_trace: Sequence[ToolTraceEntry],
    ) -> ModelResponse:
        self.call_count += 1
        step = len(previous_trace)
        
        # If we've just completed the stop_after_tool, return completed=True
        if previous_trace and previous_trace[-1].call.tool_name == self.stop_after_tool:
            return ModelResponse(
                content=f"Task completed after {self.stop_after_tool}.",
                completed=True
            )
        
        # Otherwise, request the next tool in sequence
        requests = (
            ToolRequest("get_customer", {"customer_id": self.customer_id}),
            ToolRequest("get_subscription", {"subscription_id": self.subscription_id}),
            ToolRequest("get_billing_status", {"billing_id": self.billing_id}),
            ToolRequest("cancel_subscription", {"subscription_id": self.subscription_id}),
        )
        
        if step < len(requests):
            return ModelResponse(content=f"Inspecting step {step + 1}.", tool_calls=(requests[step],))
        
        return ModelResponse(content="Done.", completed=True)


class NormalProvider:
    """Provider that completes all 4 steps (mimics original run behavior)."""
    
    def __init__(self):
        self.call_count = 0
        self.customer_id = "cust_test"
        self.subscription_id = "sub_test"
        self.billing_id = "bill_test"
    
    def complete(
        self,
        *,
        system_instructions: str,
        task: AgentTask,
        available_tools: Sequence[ToolDefinition],
        previous_trace: Sequence[ToolTraceEntry],
    ) -> ModelResponse:
        self.call_count += 1
        step = len(previous_trace)
        requests = (
            ToolRequest("get_customer", {"customer_id": self.customer_id}),
            ToolRequest("get_subscription", {"subscription_id": self.subscription_id}),
            ToolRequest("get_billing_status", {"billing_id": self.billing_id}),
            ToolRequest("cancel_subscription", {"subscription_id": self.subscription_id}),
        )
        
        if step < len(requests):
            return ModelResponse(content=f"Inspecting step {step + 1}.", tool_calls=(requests[step],))
        
        return ModelResponse(content="Done.", completed=True)


def run_agent_with_provider(environment: EnvironmentService, provider: ModelProvider) -> AgentRun:
    """Run agent with given provider and return the run."""
    tool_registry = build_tool_registry(environment)
    tool_interface = TargetAgentToolInterface(tool_registry)
    
    agent = TargetAgent(tool_interface, provider, max_steps=10)
    
    task = AgentTask(
        task_id="task_test",
        goal="Process Maya's subscription cancellation.",
    )
    
    return agent.run(task)


def test_early_stop_provider_behavior():
    """Test that early-stop provider causes agent to stop after get_customer."""
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    
    provider = EarlyStopProvider(stop_after_tool="get_customer")
    provider.customer_id = customer.id
    provider.subscription_id = subscription.id
    provider.billing_id = billing.id
    
    run = run_agent_with_provider(environment, provider)
    
    # Early-stop provider should cause agent to stop after get_customer
    assert run.status == AgentRunStatus.SUCCEEDED
    assert run.error is None
    assert run.final_result == "Task completed after get_customer."
    tool_sequence = [e.call.tool_name for e in run.trace.entries]
    assert tool_sequence == ["get_customer"]


def test_normal_provider_behavior():
    """Test that normal provider completes all steps."""
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    
    provider = NormalProvider()
    provider.customer_id = customer.id
    provider.subscription_id = subscription.id
    provider.billing_id = billing.id
    
    run = run_agent_with_provider(environment, provider)
    
    # Normal provider should complete all 4 steps
    assert run.status == AgentRunStatus.SUCCEEDED
    assert run.error is None
    assert run.final_result == "Done."
    tool_sequence = [e.call.tool_name for e in run.trace.entries]
    assert tool_sequence == ["get_customer", "get_subscription", "get_billing_status", "cancel_subscription"]


def test_classify_early_stop():
    """Classify the early-stop condition."""
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    
    provider = EarlyStopProvider(stop_after_tool="get_customer")
    provider.customer_id = customer.id
    provider.subscription_id = subscription.id
    provider.billing_id = billing.id
    
    run = run_agent_with_provider(environment, provider)
    
    tool_sequence = [e.call.tool_name for e in run.trace.entries]
    
    # Classification logic - early-stop after get_customer with completed=True
    assert run.status == AgentRunStatus.SUCCEEDED
    assert len(tool_sequence) == 1
    assert tool_sequence[0] == "get_customer"
    assert run.final_result == "Task completed after get_customer."


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING NORMAL PROVIDER BEHAVIOR")
    print("=" * 60)
    test_normal_provider_behavior()
    
    print("\n" + "=" * 60)
    print("TESTING EARLY-STOP PROVIDER BEHAVIOR")
    print("=" * 60)
    test_early_stop_provider_behavior()
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION")
    print("=" * 60)
    test_classify_early_stop()