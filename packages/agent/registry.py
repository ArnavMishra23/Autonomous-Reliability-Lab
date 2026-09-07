from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Mapping
from uuid import uuid4

from .models import ToolCall, ToolDefinition, ToolResult, ToolTrace, utc_now

ToolImplementation = Callable[[dict[str, Any]], Any]


class ToolRegistry:
    """Registers trusted internal tools and records their execution trace."""

    def __init__(self, *, clock: Callable[[], datetime] = utc_now) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolImplementation]] = {}
        self._trace = ToolTrace()
        self._clock = clock

    def register(
        self,
        definition: ToolDefinition,
        implementation: ToolImplementation,
    ) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = (definition, implementation)

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name][0]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            ToolDefinition(
                name=definition.name,
                description=definition.description,
                input_schema=definition.input_schema,
                read_only=definition.read_only,
            )
            for definition, _ in self._tools.values()
        )

    @property
    def trace(self) -> ToolTrace:
        return self._trace.snapshot()

    def reset_trace(self) -> None:
        self._trace = ToolTrace()

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolResult:
        call = ToolCall(
            call_id=f"call_{uuid4().hex[:8]}",
            tool_name=tool_name,
            arguments=arguments if isinstance(arguments, Mapping) else {},
            timestamp=self._clock(),
        )
        try:
            definition, implementation = self._tools[tool_name]
        except KeyError:
            result = self._failure(call, f"Unknown tool: {tool_name}")
            self._trace.append(call, result)
            return result

        try:
            validated = self._validate_arguments(definition, arguments)
            output = implementation(validated)
            result = ToolResult(
                call_id=call.call_id,
                tool_name=tool_name,
                success=True,
                output=deepcopy(output),
                error=None,
                timestamp=self._clock(),
            )
        except (TypeError, ValueError, KeyError) as exc:
            result = self._failure(call, str(exc))
        except Exception as exc:
            result = self._failure(call, f"Tool execution failed: {exc}")

        self._trace.append(call, result)
        return self._trace.snapshot().entries[-1].result

    @staticmethod
    def _validate_arguments(
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(arguments, Mapping):
            raise TypeError("Tool arguments must be an object")

        schema = definition.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(
                f"Missing required arguments for {definition.name}: {', '.join(missing)}"
            )
        unexpected = [name for name in arguments if name not in properties]
        if unexpected:
            raise ValueError(
                f"Unexpected arguments for {definition.name}: {', '.join(unexpected)}"
            )
        for name, value in arguments.items():
            expected_type = properties[name].get("type")
            if expected_type == "string" and not isinstance(value, str):
                raise TypeError(f"Argument {name} must be a string")
        return deepcopy(dict(arguments))

    def _failure(self, call: ToolCall, error: str) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=False,
            output=None,
            error=error or "Tool execution failed",
            timestamp=self._clock(),
        )
