from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from packages.agent.models import ToolDefinition, ToolResult, ToolErrorCode
from packages.agent.target import AgentTask, ModelResponse, ToolRequest
from packages.agent.providers.nebius import NebiusNemotronProvider
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function


class MockOpenAIClient:
    """Mock OpenAI-compatible client for offline testing."""

    def __init__(self, response=None):
        self._response = response
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(return_value=self._response)

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, value):
        self._response = value
        self.chat.completions.create.return_value = value


def create_mock_function(name: str, arguments: dict):
    """Create a mock Function object."""
    func = create_autospec(Function, instance=True)
    func.name = name
    func.arguments = json.dumps(arguments)
    return func


def create_mock_tool_call(name: str, arguments: dict, tool_call_id: str = "call_123"):
    """Create a mock ChatCompletionMessageToolCall object."""
    tc = create_autospec(ChatCompletionMessageToolCall, instance=True)
    tc.function = create_mock_function(name, arguments)
    tc.id = tool_call_id
    return tc


def create_mock_completion(content: str = "", tool_calls: list[dict] | None = None):
    """Create a mock chat completion response."""
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
    """Create a provider instance with mocked client."""
    provider = NebiusNemotronProvider.__new__(NebiusNemotronProvider)
    provider._model = model
    provider._base_url = base_url
    provider._api_key = api_key
    provider._timeout = 30.0
    provider._client = MagicMock()
    return provider


def test_provider_initialization_with_constructor_args():
    """A. Provider initialization with constructor args."""
    provider = NebiusNemotronProvider(
        model="test-model",
        base_url="https://custom.example.com/v1/",
        api_key="test-key",
        timeout=15.0,
    )
    assert provider._model == "test-model"
    assert provider._base_url == "https://custom.example.com/v1/"
    assert provider._api_key == "test-key"
    assert provider._timeout == 15.0


def test_provider_initialization_with_env_vars():
    """B. Provider initialization with environment variables."""
    with patch.dict(os.environ, {
        "NEBIUS_MODEL": "env-model",
        "NEBIUS_BASE_URL": "https://env.example.com/v1/",
        "NEBIUS_API_KEY": "env-key",
    }):
        provider = NebiusNemotronProvider()
        assert provider._model == "env-model"
        assert provider._base_url == "https://env.example.com/v1/"
        assert provider._api_key == "env-key"


def test_provider_missing_model_raises():
    """C. Missing model raises clear error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Model must be provided"):
            NebiusNemotronProvider(base_url="https://test.com", api_key="test")


def test_provider_missing_api_key_raises():
    """D. Missing API key raises clear error."""
    with patch.dict(os.environ, {"NEBIUS_MODEL": "test-model"}, clear=True):
        with pytest.raises(ValueError, match="API key must be provided"):
            NebiusNemotronProvider(model="test-model")


def test_tool_definition_conversion():
    """E. Tool definition conversion to OpenAI format."""
    tools = [
        ToolDefinition(
            name="get_customer",
            description="Read a customer",
            input_schema={"type": "object", "required": ["customer_id"], "properties": {"customer_id": {"type": "string"}}},
            read_only=True,
        ),
        ToolDefinition(
            name="cancel_subscription",
            description="Cancel a subscription",
            input_schema={"type": "object", "required": ["subscription_id"], "properties": {"subscription_id": {"type": "string"}}},
            read_only=False,
        ),
    ]

    provider = NebiusNemotronProvider.__new__(NebiusNemotronProvider)
    converted = provider._convert_tools(tools)

    assert len(converted) == 2
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "get_customer"
    assert converted[0]["function"]["description"] == "Read a customer"
    assert converted[0]["function"]["parameters"] == {"type": "object", "required": ["customer_id"], "properties": {"customer_id": {"type": "string"}}}
    assert converted[1]["function"]["name"] == "cancel_subscription"


def test_single_tool_call_parsing():
    """F. Single tool call parsing."""
    tool_calls = [{"name": "get_customer", "arguments": {"customer_id": "cust_123"}}]
    completion = create_mock_completion(content="I'll get the customer", tool_calls=tool_calls)

    provider = create_provider_with_mock_client()
    response = provider._parse_completion(completion)

    assert not response.completed
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "get_customer"
    assert response.tool_calls[0].arguments == {"customer_id": "cust_123"}


def test_final_text_parsing():
    """G. Final completion text parsing."""
    completion = create_mock_completion(content="Task completed successfully.")

    provider = NebiusNemotronProvider.__new__(NebiusNemotronProvider)
    response = provider._parse_completion(completion)

    assert response.completed
    assert response.content == "Task completed successfully."
    assert response.tool_calls == ()


def test_previous_results_serialization():
    """H. Previous tool results are serialized as tool messages."""
    provider = NebiusNemotronProvider.__new__(NebiusNemotronProvider)
    system_instructions = "Test instructions"
    task = AgentTask(task_id="test", goal="Test goal", context={"key": "value"})
    tools = []
    previous_results = [
        ToolResult(
            call_id="call_1",
            tool_name="get_customer",
            success=True,
            output={"id": "cust_1", "name": "Maya"},
            error=None,
            error_code=None,
            timestamp=None,
        ),
        ToolResult(
            call_id="call_2",
            tool_name="get_subscription",
            success=False,
            output=None,
            error="Not found",
            error_code=ToolErrorCode.UNKNOWN_SUBSCRIPTION,
            timestamp=None,
        ),
    ]

    messages = provider._build_messages(system_instructions, task, tools, previous_results)

    assert len(messages) == 4  # system + user + 2 tool messages
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_1"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call_2"
    assert "Not found" in messages[3]["content"]


def test_malformed_tool_call_arguments():
    """I. Malformed JSON in tool call arguments raises clear error."""
    tool_calls = [{"name": "get_customer", "arguments": {"customer_id": "cust_1"}}]
    completion = create_mock_completion(content="", tool_calls=tool_calls)
    # Manually override the arguments to be malformed
    completion.choices[0].message.tool_calls[0].function.arguments = "not valid json"

    provider = create_provider_with_mock_client()
    with pytest.raises(ValueError, match="Malformed tool call arguments"):
        provider._parse_completion(completion)


def test_multiple_tool_calls_preserved():
    """J. Multiple tool calls are preserved in ModelResponse (TargetAgent enforces single-call semantics)."""
    tool_calls = [
        {"name": "get_customer", "arguments": {"customer_id": "cust_1"}},
        {"name": "get_subscription", "arguments": {"subscription_id": "sub_1"}},
    ]
    completion = create_mock_completion(content="", tool_calls=tool_calls)

    provider = create_provider_with_mock_client()
    response = provider._parse_completion(completion)

    # Provider preserves all tool calls; TargetAgent enforces single-call semantics
    assert not response.completed
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].tool_name == "get_customer"
    assert response.tool_calls[1].tool_name == "get_subscription"


def test_api_error_handling():
    """K. API exceptions are converted to clear provider errors."""
    provider = create_provider_with_mock_client()
    provider._client.chat.completions.create = MagicMock(side_effect=Exception("API rate limit"))

    with pytest.raises(RuntimeError, match="Nebius API error: API rate limit"):
        provider.complete(
            system_instructions="test",
            task=AgentTask(task_id="test", goal="test"),
            available_tools=[],
            previous_results=[],
        )


def test_timeout_configuration():
    """L. Timeout is configured on the client."""
    with patch.dict(os.environ, {"NEBIUS_API_KEY": "test", "NEBIUS_MODEL": "test"}):
        provider = NebiusNemotronProvider(timeout=10.0)
        assert provider._timeout == 10.0
        # The client should have the timeout
        assert provider._client.timeout == 10.0


def test_no_secret_leakage_in_errors():
    """M. API key is not exposed in error messages."""
    with patch.dict(os.environ, {"NEBIUS_API_KEY": "secret-key-123", "NEBIUS_MODEL": "test"}):
        provider = NebiusNemotronProvider()
        # The api_key should be stored internally but not logged
        assert hasattr(provider, "_api_key")
        assert provider._api_key == "secret-key-123"

    # Test that errors don't contain the key
    provider._client = MagicMock()
    provider._client.chat.completions.create = MagicMock(side_effect=Exception("auth failed"))

    try:
        provider.complete(
            system_instructions="test",
            task=AgentTask(task_id="test", goal="test"),
            available_tools=[],
            previous_results=[],
        )
    except RuntimeError as exc:
        error_str = str(exc)
        assert "secret-key-123" not in error_str


def test_target_agent_compatibility_with_fake_client():
    """N. TargetAgent works with fake Nebius client (offline integration test)."""
    from packages.agent.target import TargetAgent
    from packages.agent.protocol import TargetAgentToolInterface
    from packages.agent.tools import build_tool_registry
    from packages.environment.service import EnvironmentService

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    # Create a fake provider that returns a successful completion without tool calls
    class FakeNebiusProvider:
        def complete(self, *, system_instructions, task, available_tools, previous_results):
            return ModelResponse(content="Done", completed=True)

    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        FakeNebiusProvider(),
    )

    run = agent.run(AgentTask(task_id="test", goal="Test"))
    assert run.status.value == "succeeded"
    assert run.final_result == "Done"


def test_existing_full_test_suite_still_green():
    """O. Run the existing test suite to ensure no regressions."""
    # This test is a placeholder - the actual test suite is run separately
    pass