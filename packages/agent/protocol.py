from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import ToolDefinition, ToolResult, ToolTrace
from .registry import ToolRegistry


class TargetAgentToolInterface:
    """The only tool-facing boundary exposed to a target agent."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]:
        return self._registry.list_tools()

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolResult:
        return self._registry.execute(tool_name, arguments)

    @property
    def trace(self) -> ToolTrace:
        return self._registry.trace
