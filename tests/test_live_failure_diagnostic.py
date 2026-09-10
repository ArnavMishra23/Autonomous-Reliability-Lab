from __future__ import annotations

import json
from datetime import datetime, timezone

from packages.agent.models import ToolDefinition, ToolResult, ToolCall, ToolTraceEntry, ToolErrorCode
from packages.agent.target import AgentTask
from packages.agent.providers.nebius import NebiusNemotronProvider
from unittest.mock import MagicMock


def create_provider_with_mock_client(model="test-model", api_key="test-key", base_url="https://test.com/v1/"):
    provider = NebiusNemotronProvider.__new__(NebiusNemotronProvider)
    provider._model = model
    provider._base_url = base_url
    provider._api_key = api_key
    provider._timeout = 30.0
    provider._client = MagicMock()
    return provider


def create_trace_entry(call_id: str, tool_name: str, arguments: dict, success: bool, output: dict, error: str | None = None, error_code: ToolErrorCode | None = None, step: int = 1):
    """Helper to create a complete trace entry with both call and result."""
    now = datetime.now(timezone.utc)
    call = ToolCall(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
        timestamp=now,
        step=step,
    )
    result = ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        success=success,
        output=output,
        error=error,
        error_code=error_code,
        timestamp=now,
    )
    return ToolTraceEntry(call=call, result=result)


def test_message_history_after_cancellation():
    """
    Reproduce the exact message history the model sees after:
    1. get_customer (success)
    2. get_subscription (success)
    3. get_billing_status (success - billing active)
    4. cancel_subscription (success - subscription cancelled, billing still active)
    5. get_billing_status (success - billing still active)
    
    This is the state BEFORE step 6 (reconcile_subscription_state)
    """
    provider = create_provider_with_mock_client()
    
    system_instructions = "Use only the available tools. Do not assume unobserved state."
    task = AgentTask(task_id="test", goal="Cancel subscription", context={})
    tools = [
        ToolDefinition("get_customer", "Read customer", {"type": "object", "required": ["customer_id"], "properties": {"customer_id": {"type": "string"}}}, True),
        ToolDefinition("get_subscription", "Read subscription", {"type": "object", "required": ["subscription_id"], "properties": {"subscription_id": {"type": "string"}}}, True),
        ToolDefinition("get_billing_status", "Read billing status", {"type": "object", "required": ["billing_id"], "properties": {"billing_id": {"type": "string"}}}, True),
        ToolDefinition("cancel_subscription", "Cancel subscription", {"type": "object", "required": ["subscription_id"], "properties": {"subscription_id": {"type": "string"}}}, False),
        ToolDefinition("reconcile_subscription_state", "Reconcile subscription", {"type": "object", "required": ["subscription_id"], "properties": {"subscription_id": {"type": "string"}}}, False),
    ]
    
    # Simulate the trace after step 5 (after get_billing_status following cancellation)
    previous_trace = [
        # Step 1: get_customer
        create_trace_entry(
            call_id="call_1",
            tool_name="get_customer",
            arguments={"customer_id": "cust_b177ce18"},
            success=True,
            output={
                "id": "cust_b177ce18",
                "name": "Test Customer",
                "email": "test@example.com",
                "subscription_id": "sub_40ba5301",
                "billing_id": "bill_a525d787"
            },
            step=1,
        ),
        # Step 2: get_subscription
        create_trace_entry(
            call_id="call_2",
            tool_name="get_subscription",
            arguments={"subscription_id": "sub_40ba5301"},
            success=True,
            output={
                "id": "sub_40ba5301",
                "customer_id": "cust_b177ce18",
                "plan": "pro",
                "status": "active",
                "created_at": "2024-01-01T00:00:00+00:00"
            },
            step=2,
        ),
        # Step 3: get_billing_status (before cancellation)
        create_trace_entry(
            call_id="call_3",
            tool_name="get_billing_status",
            arguments={"billing_id": "bill_a525d787"},
            success=True,
            output={
                "id": "bill_a525d787",
                "subscription_id": "sub_40ba5301",
                "status": "active"
            },
            step=3,
        ),
        # Step 4: cancel_subscription (SUCCESS - subscription cancelled, billing still active)
        create_trace_entry(
            call_id="call_4",
            tool_name="cancel_subscription",
            arguments={"subscription_id": "sub_40ba5301"},
            success=True,
            output={
                "subscription": {
                    "id": "sub_40ba5301",
                    "customer_id": "cust_b177ce18",
                    "plan": "pro",
                    "status": "cancelled",  # <-- KEY: subscription is cancelled
                    "created_at": "2024-01-01T00:00:00+00:00"
                },
                "event": {
                    "id": "evt_123",
                    "event_type": "subscription_cancelled",
                    "timestamp": "2024-01-01T12:00:00+00:00",
                    "target": "subscription",
                    "field": "status",
                    "old_value": "active",
                    "new_value": "cancelled",
                    "metadata": {}
                }
            },
            step=4,
        ),
        # Step 5: get_billing_status (after cancellation - billing STILL ACTIVE)
        create_trace_entry(
            call_id="call_5",
            tool_name="get_billing_status",
            arguments={"billing_id": "bill_a525d787"},
            success=True,
            output={
                "id": "bill_a525d787",
                "subscription_id": "sub_40ba5301",
                "status": "active"  # <-- KEY: billing is STILL active
            },
            step=5,
        ),
    ]
    
    messages = provider._build_messages(system_instructions, task, tools, previous_trace)
    
    print("\n=== MESSAGE HISTORY BEFORE STEP 6 (reconcile_subscription_state) ===")
    for i, msg in enumerate(messages):
        print(f"\nMessage {i}:")
        print(f"  role: {msg.get('role')}")
        if msg.get('role') == 'tool':
            print(f"  tool_call_id: {msg.get('tool_call_id')}")
            print(f"  name: {msg.get('name')}")
            content = msg.get('content', '')
            print(f"  content: {content[:300]}...")
        elif msg.get('role') == 'assistant':
            print(f"  content: {msg.get('content')}")
            tool_calls = msg.get('tool_calls', [])
            for tc in tool_calls:
                print(f"  tool_call: id={tc['id']}, name={tc['function']['name']}, args={tc['function']['arguments']}")
        else:
            print(f"  content: {msg.get('content')[:100]}...")
    
    # Verify the message structure
    assistant_messages = [m for m in messages if m.get('role') == 'assistant']
    tool_messages = [m for m in messages if m.get('role') == 'tool']
    
    print(f"\n=== SUMMARY ===")
    print(f"Total messages: {len(messages)}")
    print(f"Assistant messages (with tool_calls): {len(assistant_messages)}")
    print(f"Tool messages: {len(tool_messages)}")
    
    # Check the cancel_subscription result specifically
    cancel_result = next((m for m in tool_messages if m.get('name') == 'cancel_subscription'), None)
    if cancel_result:
        print(f"\n=== CANCEL_SUBSCRIPTION RESULT (what model saw) ===")
        print(cancel_result.get('content', '')[:500])
    
    # Check the second get_billing_status result
    billing_results = [m for m in tool_messages if m.get('name') == 'get_billing_status']
    if len(billing_results) >= 2:
        print(f"\n=== SECOND GET_BILLING_STATUS RESULT (after cancellation) ===")
        print(billing_results[1].get('content', '')[:500])
    
    return messages


def test_mock_provider_behavior():
    """Test what the MockProvider would do with this trace."""
    from packages.agent.target import MockProvider, ModelResponse, ToolRequest
    from packages.agent.models import ToolDefinition
    
    provider = MockProvider(
        customer_id="cust_b177ce18",
        subscription_id="sub_40ba5301",
        billing_id="bill_a525d787",
    )
    
    task = AgentTask(task_id="test", goal="Cancel subscription", context={})
    tools = (
        ToolDefinition("get_customer", "", {}, True),
        ToolDefinition("get_subscription", "", {}, True),
        ToolDefinition("get_billing_status", "", {}, True),
        ToolDefinition("cancel_subscription", "", {}, False),
        ToolDefinition("reconcile_subscription_state", "", {}, False),
    )
    
    # Create the same trace as above
    trace = [
        create_trace_entry("call_1", "get_customer", {"customer_id": "cust_b177ce18"}, True, 
                          {"id": "cust_b177ce18", "subscription_id": "sub_40ba5301", "billing_id": "bill_a525d787"}, step=1),
        create_trace_entry("call_2", "get_subscription", {"subscription_id": "sub_40ba5301"}, True,
                          {"id": "sub_40ba5301", "status": "active"}, step=2),
        create_trace_entry("call_3", "get_billing_status", {"billing_id": "bill_a525d787"}, True,
                          {"id": "bill_a525d787", "status": "active"}, step=3),
        create_trace_entry("call_4", "cancel_subscription", {"subscription_id": "sub_40ba5301"}, True,
                          {"subscription": {"id": "sub_40ba5301", "status": "cancelled"}, 
                           "event": {"event_type": "subscription_cancelled", "old_value": "active", "new_value": "cancelled"}}, step=4),
        create_trace_entry("call_5", "get_billing_status", {"billing_id": "bill_a525d787"}, True,
                          {"id": "bill_a525d787", "status": "active"}, step=5),
    ]
    
    response = provider.complete(
        system_instructions="",
        task=task,
        available_tools=tools,
        previous_trace=trace,
    )
    
    print(f"\n=== MOCK PROVIDER RESPONSE ===")
    print(f"Content: {response.content}")
    print(f"Completed: {response.completed}")
    print(f"Tool calls: {[(tc.tool_name, tc.arguments) for tc in response.tool_calls]}")
    
    return response


def test_reconcile_failure():
    """Test what happens when reconcile is called on already-cancelled subscription."""
    from packages.environment.service import EnvironmentService
    from packages.agent.tools import build_tool_registry
    from packages.agent.protocol import TargetAgentToolInterface
    
    environment = EnvironmentService()
    customer = environment.create_customer(name="Test", email="test@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    
    # Cancel the subscription (but NOT the billing)
    environment.cancel_subscription(subscription_id=subscription.id)
    
    print(f"\n=== STATE AFTER CANCEL_SUBSCRIPTION ===")
    print(f"Subscription status: {environment.get_subscription(subscription.id).status}")
    print(f"Billing status: {environment.get_billing(billing.id).status}")
    
    # Now try to reconcile
    interface = TargetAgentToolInterface(build_tool_registry(environment))
    
    result = interface.execute("reconcile_subscription_state", {"subscription_id": subscription.id}, step=6)
    
    print(f"\n=== RECONCILE_SUBSCRIPTION_STATE RESULT ===")
    print(f"Tool: {result.tool_name}")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Error code: {result.error_code}")
    print(f"Output: {result.output}")
    
    return result


if __name__ == "__main__":
    test_message_history_after_cancellation()
    test_mock_provider_behavior()
    test_reconcile_failure()
    print("\n=== ALL DIAGNOSTIC TESTS COMPLETE ===")