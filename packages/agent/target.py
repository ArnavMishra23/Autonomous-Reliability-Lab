from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from .models import ToolDefinition, ToolResult, ToolTrace
from .protocol import TargetAgentToolInterface


class AgentRunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AgentTask:
    task_id: str
    goal: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", deepcopy(dict(self.context)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "context": _json_value(self.context),
        }


@dataclass(frozen=True, slots=True)
class ToolRequest:
    tool_name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", deepcopy(dict(self.arguments)))


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    tool_calls: tuple[ToolRequest, ...] = ()
    completed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))


class ModelProvider(Protocol):
    def complete(
        self,
        *,
        system_instructions: str,
        task: AgentTask,
        available_tools: Sequence[ToolDefinition],
        previous_results: Sequence[ToolResult],
    ) -> ModelResponse:
        """Return the model's next text and optional tool requests."""


@dataclass
class AgentRun:
    run_id: str
    task_id: str
    status: AgentRunStatus
    started_at: datetime
    finished_at: datetime | None = None
    trace: ToolTrace = field(default_factory=ToolTrace)
    final_result: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at is not None else None
            ),
            "trace": [
                {
                    "call": {
                        "call_id": entry.call.call_id,
                        "tool_name": entry.call.tool_name,
                        "arguments": _json_value(entry.call.arguments),
                        "timestamp": entry.call.timestamp.isoformat(),
                    },
                    "result": {
                        "call_id": entry.result.call_id,
                        "tool_name": entry.result.tool_name,
                        "success": entry.result.success,
                        "output": _json_value(entry.result.output),
                        "error": entry.result.error,
                        "timestamp": entry.result.timestamp.isoformat(),
                    },
                }
                for entry in self.trace.entries
            ],
            "final_result": self.final_result,
            "error": self.error,
        }


class MockProvider:
    """Deterministic offline provider for the cancellation workflow."""

    def __init__(
        self,
        *,
        customer_id: str,
        subscription_id: str,
        billing_id: str,
    ) -> None:
        self.customer_id = customer_id
        self.subscription_id = subscription_id
        self.billing_id = billing_id
        self.calls: list[tuple[str, tuple[ToolResult, ...]]] = []

    def complete(
        self,
        *,
        system_instructions: str,
        task: AgentTask,
        available_tools: Sequence[ToolDefinition],
        previous_results: Sequence[ToolResult],
    ) -> ModelResponse:
        del system_instructions, task, available_tools
        self.calls.append(("complete", tuple(previous_results)))
        step = len(previous_results)
        requests = (
            ToolRequest("get_customer", {"customer_id": self.customer_id}),
            ToolRequest("get_subscription", {"subscription_id": self.subscription_id}),
            ToolRequest("get_billing_status", {"billing_id": self.billing_id}),
            ToolRequest("cancel_subscription", {"subscription_id": self.subscription_id}),
        )
        if step < len(requests):
            return ModelResponse(content=f"Inspecting step {step + 1}.", tool_calls=(requests[step],))
        return ModelResponse(
            content="The subscription cancellation workflow is complete.",
            completed=True,
        )


class TargetAgent:
    """Runs a replaceable model through the explicit tool protocol boundary."""

    def __init__(
        self,
        tools: TargetAgentToolInterface,
        model: ModelProvider,
        *,
        max_steps: int = 10,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._tools = tools
        self._model = model
        self._max_steps = max_steps
        self._clock = clock

    def run(self, task: AgentTask) -> AgentRun:
        return self._run(task, step_hook=None)

    def run_with_step_hook(
        self,
        task: AgentTask,
        *,
        step_hook: Callable[[int], None],
    ) -> AgentRun:
        """Run with a callback before each 1-based model decision step."""
        return self._run(task, step_hook=step_hook)

    def _run(
        self,
        task: AgentTask,
        *,
        step_hook: Callable[[int], None] | None,
    ) -> AgentRun:
        self._tools.reset_trace()
        run = AgentRun(
            run_id=f"run_{uuid4().hex[:8]}",
            task_id=task.task_id,
            status=AgentRunStatus.RUNNING,
            started_at=self._clock(),
        )
        previous_results: list[ToolResult] = []
        system_instructions = "Use only the available tools. Do not assume unobserved state."
        try:
            for step in range(1, self._max_steps + 1):
                if step_hook is not None:
                    step_hook(step)
                response = self._model.complete(
                    system_instructions=system_instructions,
                    task=task,
                    available_tools=self._tools.available_tools,
                    previous_results=tuple(previous_results),
                )
                if not isinstance(response, ModelResponse):
                    return self._fail(run, "Malformed model response")
                if response.completed:
                    if response.tool_calls:
                        return self._fail(
                            run,
                            "Malformed model response: completed response requested tools",
                        )
                    run.status = AgentRunStatus.SUCCEEDED
                    run.final_result = response.content
                    return self._finish(run)
                if len(response.tool_calls) != 1:
                    return self._fail(
                        run,
                        "Malformed model response: expected exactly one tool call",
                    )

                request = response.tool_calls[0]
                if not isinstance(request, ToolRequest):
                    return self._fail(run, "Malformed model response: invalid tool request")
                result = self._tools.execute(request.tool_name, request.arguments)
                previous_results.append(result)
                run.trace = self._tools.trace
                if not result.success:
                    return self._fail(run, result.error or "Tool execution failed")
            return self._fail(run, f"Maximum agent steps exceeded: {self._max_steps}")
        except Exception as exc:
            return self._fail(run, f"Model execution failed: {exc}")

    def _finish(self, run: AgentRun) -> AgentRun:
        run.finished_at = self._clock()
        run.trace = self._tools.trace
        return run

    def _fail(self, run: AgentRun, error: str) -> AgentRun:
        run.status = AgentRunStatus.FAILED
        run.error = error
        return self._finish(run)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value({key: getattr(value, key) for key in value.__dataclass_fields__})
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    return value
