from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    read_only: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_schema",
            MappingProxyType(deepcopy(dict(self.input_schema))),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(deepcopy(dict(self.arguments))),
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    output: Any
    error: str | None
    timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", deepcopy(self.output))


@dataclass(frozen=True, slots=True)
class ToolTraceEntry:
    call: ToolCall
    result: ToolResult


@dataclass
class ToolTrace:
    entries: list[ToolTraceEntry] = field(default_factory=list)

    def append(self, call: ToolCall, result: ToolResult) -> None:
        self.entries.append(ToolTraceEntry(call=call, result=result))

    def snapshot(self) -> "ToolTrace":
        return ToolTrace(
            entries=[
                ToolTraceEntry(
                    call=ToolCall(
                        call_id=entry.call.call_id,
                        tool_name=entry.call.tool_name,
                        arguments=entry.call.arguments,
                        timestamp=entry.call.timestamp,
                    ),
                    result=ToolResult(
                        call_id=entry.result.call_id,
                        tool_name=entry.result.tool_name,
                        success=entry.result.success,
                        output=entry.result.output,
                        error=entry.result.error,
                        timestamp=entry.result.timestamp,
                    ),
                )
                for entry in self.entries
            ]
        )
