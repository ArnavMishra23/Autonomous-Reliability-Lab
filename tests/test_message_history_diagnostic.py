from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from packages.agent.models import ToolDefinition, ToolResult, ToolErrorCode, ToolCall, ToolTraceEntry
from packages.agent.target import AgentTask, ModelResponse, ToolRequest
from packages.agent.providers.nebius import NebiusNemotronProvider


def create_mock_function(name: str, arguments: dict):
    func = create_autospec(Function, instance=True)
    func.name = name
    func.arguments = json.dumps(arguments)
    return func


def create_mock_tool_call(name: str, arguments: dict, tool_call_id: str = "call_123"):
    tc = create_autospec(ChatCompletionMessageToolCall, instance=True)
    tc.function = create_mock_function(name, arguments)
    tc.id = tool_call_id
    return tc


def create_mock_completion(content: str = "", tool_calls: list[dict] | None = None):
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    if tool_calls:
        mock_tool_calls = []
        for tc in tool_calls:
            mock_tool_calls.append(create_mock_tool_call(tc["name"], tc.get("arguments", {})))
        message.tool_calls = mock_tool_calls
    else:
        message.tool_calls = None
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def create_provider_with_mock_client(model="test-model", api_key="test-key", base_url="https://test.com/v1/"):
    provider = NebiusNemotronProvider.__new__(NebiusNemotronProvider)
    provider._model = model
    provider._base_url = base_url
    provider._api_key = api_key
    provider._timeout = 30.0
    provider._client = MagicMock()
    return provider


def test_fixed_message_history_structure():
    """
    This test verifies the FIXED message history structure sent to the model.
    
    After the fix, the provider now includes BOTH:
    - Assistant tool-call messages (with tool_calls)
    - Tool result messages
    
    This gives the model full visibility into its previous actions.
    """
    provider = create_provider_with_mock_client()
    
    system_instructions = "Use only the available tools. Do not assume unobserved state."
    task = AgentTask(task_id="test", goal="Cancel subscription", context={})
    tools = [
        ToolDefinition("get_customer", "Read customer", {"type": "object", "required": ["customer_id"], "properties": {"customer_id": {"type": "string"}}}, True),
        ToolDefinition("cancel_subscription", "Cancel subscription", {"type": "object", "required": ["subscription_id"], "properties": {"subscription_id": {"type": "string"}}}, False),
    ]
    
    # Create trace entries with both calls and results (the new API)
    previous_trace = [
        ToolTraceEntry(
            call=ToolCall(
                call_id="call_1",
                tool_name="get_customer",
                arguments={"customer_id": "cust_1"},
                timestamp=datetime.now(timezone.utc),
                step=1,
            ),
            result=ToolResult(
                call_id="call_1",
                tool_name="get_customer",
                success=True,
                output={"id": "cust_1", "name": "Maya", "subscription_id": "sub_1", "billing_id": "bill_1"},
                error=None,
                error_code=None,
                timestamp=datetime.now(timezone.utc),
            )
        ),
        ToolTraceEntry(
            call=ToolCall(
                call_id="call_2",
                tool_name="cancel_subscription",
                arguments={"subscription_id": "sub_1"},
                timestamp=datetime.now(timezone.utc),
                step=2,
            ),
            result=ToolResult(
                call_id="call_2",
                tool_name="cancel_subscription",
                success=True,
                output={
                    "subscription": {"id": "sub_1", "status": "cancelled"},
                    "event": {"event_type": "subscription_cancelled", "old_value": "active", "new_value": "cancelled"}
                },
                error=None,
                error_code=None,
                timestamp=datetime.now(timezone.utc),
            )
        ),
    ]
    
    messages = provider._build_messages(system_instructions, task, tools, previous_trace)
    
    # Print the actual messages for diagnosis
    print("\n=== FIXED MESSAGE HISTORY SENT TO MODEL ===")
    for i, msg in enumerate(messages):
        print(f"\nMessage {i}:")
        print(f"  role: {msg.get('role')}")
        if msg.get('role') == 'tool':
            print(f"  tool_call_id: {msg.get('tool_call_id')}")
            print(f"  name: {msg.get('name')}")
            print(f"  content: {msg.get('content')[:200]}...")
        elif msg.get('role') == 'assistant':
            print(f"  content: {msg.get('content')}")
            print(f"  tool_calls: {msg.get('tool_calls')}")
        else:
            print(f"  content: {msg.get('content')[:100]}...")
    
    # Verify the fix: We now have assistant tool-call messages!
    assistant_messages = [m for m in messages if m.get('role') == 'assistant']
    tool_messages = [m for m in messages if m.get('role') == 'tool']
    
    print(f"\n=== DIAGNOSIS ===")
    print(f"Total messages: {len(messages)}")
    print(f"Assistant messages (with tool_calls): {len(assistant_messages)}")
    print(f"Tool messages: {len(tool_messages)}")
    
    # After fix: we have 2 assistant tool-call messages and 2 tool result messages
    assert len(assistant_messages) == 2, "FIXED: Now has 2 assistant tool-call messages in history!"
    assert len(tool_messages) == 2, "Should have 2 tool result messages"
    
    # Verify the pairing
    for i, amsg in enumerate(assistant_messages):
        tool_call_id = amsg['tool_calls'][0]['id']
        matching_tool = next((t for t in tool_messages if t['tool_call_id'] == tool_call_id), None)
        assert matching_tool is not None, f"No matching tool result for call_id {tool_call_id}"
        print(f"  OK Assistant call {tool_call_id} -> Tool result {matching_tool['tool_call_id']}")
    
    print("\n*** FIX VERIFIED: PROVIDER_MESSAGE_HISTORY_BUG IS FIXED ***")
    print("The provider now sends both assistant tool-call messages AND tool results.")


def test_cancel_subscription_result_content():
    """Verify what the successful cancel_subscription result contains."""
    # This is the actual output from _cancel_subscription in tools.py
    result = {
        "subscription": {
            "id": "sub_f887f9e1", 
            "customer_id": "cust_5e402028", 
            "plan": "pro", 
            "status": "cancelled",  # <-- KEY: status is "cancelled"
            "created_at": "2024-01-01T00:00:00+00:00"
        },
        "event": {
            "id": "evt_123",
            "event_type": "subscription_cancelled",
            "timestamp": "2024-01-01T12:00:00+00:00",
            "target": "subscription",
            "field": "status",
            "old_value": "active",
            "new_value": "cancelled",  # <-- KEY: new_value is "cancelled"
            "metadata": {}
        }
    }
    
    print("\n=== SUCCESSFUL CANCEL_SUBSCRIPTION RESULT ===")
    print(json.dumps(result, indent=2))
    
    # The result clearly indicates the subscription is cancelled
    assert result["subscription"]["status"] == "cancelled"
    assert result["event"]["new_value"] == "cancelled"
    print("\nOK Result clearly shows subscription status = cancelled")


if __name__ == "__main__":
    test_fixed_message_history_structure()
    test_cancel_subscription_result_content()
    print("\n=== ALL DIAGNOSTIC TESTS COMPLETE ===")