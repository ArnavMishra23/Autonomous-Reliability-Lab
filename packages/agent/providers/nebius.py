from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence
from unittest.mock import MagicMock, create_autospec

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from packages.agent.models import ToolDefinition, ToolResult
from packages.agent.target import AgentTask, ModelProvider, ModelResponse, ToolRequest


class NebiusNemotronProvider:
    """Nebius Token Factory model provider implementing the ModelProvider protocol.

    Uses the OpenAI-compatible API exposed by Nebius Token Factory.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._model = model or os.getenv("NEBIUS_MODEL")
        if not self._model:
            raise ValueError("Model must be provided via constructor or NEBIUS_MODEL environment variable")

        self._base_url = base_url or os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
        self._api_key = api_key or os.getenv("NEBIUS_API_KEY")
        if not self._api_key:
            raise ValueError("API key must be provided via constructor or NEBIUS_API_KEY environment variable")

        self._timeout = timeout
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=timeout,
        )

    def complete(
        self,
        *,
        system_instructions: str,
        task: AgentTask,
        available_tools: Sequence[ToolDefinition],
        previous_results: Sequence[ToolResult],
    ) -> ModelResponse:
        messages = self._build_messages(system_instructions, task, available_tools, previous_results)
        tools = self._convert_tools(available_tools)

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
            )
        except Exception as exc:
            raise RuntimeError(f"Nebius API error: {exc}") from exc

        return self._parse_completion(completion)

    def _build_messages(
        self,
        system_instructions: str,
        task: AgentTask,
        available_tools: Sequence[ToolDefinition],
        previous_results: Sequence[ToolResult],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": f"Task: {task.goal}\nContext: {task.context}"},
        ]

        for result in previous_results:
            if result.success:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "name": result.tool_name,
                    "content": json.dumps(result.output),
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "name": result.tool_name,
                    "content": json.dumps({"error": result.error, "error_code": result.error_code.value if result.error_code else None}),
                })

        return messages

    def _convert_tools(self, available_tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
        tools = []
        for tool in available_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.input_schema),
                },
            })
        return tools

    def _parse_completion(self, completion) -> ModelResponse:
        choice = completion.choices[0]
        message = choice.message

        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                if not isinstance(tc, ChatCompletionMessageToolCall):
                    continue
                function = getattr(tc, "function", None)
                if function is None:
                    continue
                arguments_str = getattr(function, "arguments", "")
                try:
                    arguments = json.loads(arguments_str) if arguments_str else {}
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed tool call arguments: {exc}") from exc
                tool_calls.append(ToolRequest(tool_name=function.name, arguments=arguments))

            # Preserve all tool calls; TargetAgent will enforce single-call-per-turn semantics
            # with a clear error message if multiple calls are returned.
            content = message.content or f"Calling tool: {tool_calls[0].tool_name}"
            return ModelResponse(content=content, tool_calls=tuple(tool_calls))

        content = message.content or ""
        if not content.strip():
            raise ValueError("Empty model response")

        return ModelResponse(content=content, completed=True)