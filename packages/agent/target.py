from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Mapping, Protocol, Sequence

from .models import ToolDefinition, ToolResult, ToolTrace, ToolTraceEntry, ToolCall, generate_id
from packages.environment.models import Clock, DefaultClock
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
        latest_billing_status = next(
            (
                result.output.get("status")
                for result in reversed(previous_results)
                if result.tool_name == "get_billing_status"
                and result.success
                and isinstance(result.output, Mapping)
            ),
            None,
        )
        if latest_billing_status == "cancelled":
            reconciled = any(
                result.tool_name == "reconcile_subscription_state"
                and result.success
                for result in previous_results
            )
            if not reconciled:
                return ModelResponse(
                    content="Billing is cancelled; reconcile the subscription state.",
                    tool_calls=(
                        ToolRequest(
                            "reconcile_subscription_state",
                            {"subscription_id": self.subscription_id},
                        ),
                    ),
                )
            return ModelResponse(
                content="The cancellation is resolved by the reconciled billing state.",
                completed=True,
            )
        if step < len(requests):
            return ModelResponse(content=f"Inspecting step {step + 1}.", tool_calls=(requests[step],))
        return ModelResponse(
            content="The subscription cancellation workflow is complete.",
            completed=True,
        )


class TargetAgent:
    """Runs a replaceable model through the explicit tool protocol boundary.

    The agent configuration (model, max_steps, clock) is reusable.
    Each run receives its own tool interface, ensuring isolation between runs.
    """

    def __init__(
        self,
        tools: TargetAgentToolInterface,
        model: ModelProvider,
        *,
        max_steps: int = 10,
        clock: Clock | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._default_tools = tools
        self._model = model
        self._max_steps = max_steps
        self._clock = clock or DefaultClock()

    @property
    def default_tools(self) -> TargetAgentToolInterface:
        return self._default_tools

    def run(self, task: AgentTask, *, tools: TargetAgentToolInterface | None = None) -> AgentRun:
        return self._run(task, step_hook=None, tools=tools)

    def run_with_step_hook(
        self,
        task: AgentTask,
        *,
        step_hook: Callable[[int], None],
        tools: TargetAgentToolInterface | None = None,
    ) -> AgentRun:
        """Run with a callback before each 1-based model decision step."""
        return self._run(task, step_hook=step_hook, tools=tools)

    def run_with_tools(
        self,
        task: AgentTask,
        tools: TargetAgentToolInterface,
    ) -> AgentRun:
        """Run with an alternate tool boundary using the same model configuration."""
        return self._run(task, step_hook=None, tools=tools)

    def run_with_tool_wrapper(
        self,
        task: AgentTask,
        wrapper: Callable[[TargetAgentToolInterface], TargetAgentToolInterface],
        *,
        tools: TargetAgentToolInterface | None = None,
    ) -> AgentRun:
        """Run with a wrapper around the current (or provided) tool boundary."""
        base_tools = tools or self._default_tools
        return self._run(task, step_hook=None, tools=wrapper(base_tools))

    def run_with_tool_wrapper_and_step_hook(
        self,
        task: AgentTask,
        wrapper: Callable[[TargetAgentToolInterface], TargetAgentToolInterface],
        *,
        step_hook: Callable[[int], None],
        tools: TargetAgentToolInterface | None = None,
    ) -> AgentRun:
        base_tools = tools or self._default_tools
        return self._run(task, step_hook=step_hook, tools=wrapper(base_tools))

    def _run(
        self,
        task: AgentTask,
        *,
        step_hook: Callable[[int], None] | None,
        tools: TargetAgentToolInterface | None = None,
    ) -> AgentRun:
        active_tools = tools or self._default_tools
        active_tools.reset_trace()
        run = AgentRun(
            run_id=generate_id("run_"),
            task_id=task.task_id,
            status=AgentRunStatus.RUNNING,
            started_at=self._clock.now(),
        )
        previous_results: list[ToolResult] = []
        run_trace_entries: list[ToolTraceEntry] = []
        system_instructions = "Use only the available tools. Do not assume unobserved state."
        try:
            for step in range(1, self._max_steps + 1):
                if step_hook is not None:
                    step_hook(step)
                response = self._model.complete(
                    system_instructions=system_instructions,
                    task=task,
                    available_tools=active_tools.available_tools,
                    previous_results=tuple(previous_results),
                )
                if not isinstance(response, ModelResponse):
                    return self._fail(run, "Malformed model response", run_trace_entries)
                if response.completed:
                    if response.tool_calls:
                        return self._fail(
                            run,
                            "Malformed model response: completed response requested tools",
                            run_trace_entries,
                        )
                    run.status = AgentRunStatus.SUCCEEDED
                    run.final_result = response.content
                    return self._finish(run, run_trace_entries)
                if len(response.tool_calls) != 1:
                    return self._fail(
                        run,
                        "Malformed model response: expected exactly one tool call",
                        run_trace_entries,
                    )

                request = response.tool_calls[0]
                if not isinstance(request, ToolRequest):
                    return self._fail(run, "Malformed model response: invalid tool request", run_trace_entries)
                executed_tool_name = request.tool_name
                result = active_tools.execute(request.tool_name, request.arguments, step=step)
                # Use the actual executed tool name from the result (may differ due to interventions)
                executed_tool_name = result.tool_name
                previous_results.append(result)
                run_trace_entries.append(ToolTraceEntry(call=ToolCall(
                    call_id=generate_id("call_"),
                    tool_name=executed_tool_name,
                    arguments=request.arguments,
                    timestamp=self._clock.now(),
                    step=step,
                ), result=result))
                if not result.success:
                    return self._fail(run, result.error or "Tool execution failed", run_trace_entries)
            return self._fail(run, f"Maximum agent steps exceeded: {self._max_steps}", run_trace_entries)
        except Exception as exc:
            return self._fail(run, f"Model execution failed: {exc}", run_trace_entries)

    def _finish(self, run: AgentRun, trace_entries: list[ToolTraceEntry]) -> AgentRun:
        run.finished_at = self._clock.now()
        run.trace = ToolTrace(entries=trace_entries)
        return run

    def _fail(self, run: AgentRun, error: str, trace_entries: list[ToolTraceEntry] | None = None) -> AgentRun:
        run.status = AgentRunStatus.FAILED
        run.error = error
        if trace_entries is not None:
            run.trace = ToolTrace(entries=trace_entries)
        run.finished_at = self._clock.now()
        return run


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
