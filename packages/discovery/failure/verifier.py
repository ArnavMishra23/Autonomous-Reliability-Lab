from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from packages.discovery.scenario.models import ScenarioRun

from .models import CheckResult, VerificationResult


class DeterministicVerifier:
    """Verifies the initial subscription-cancellation contract from evidence."""

    def verify(self, scenario_run: ScenarioRun) -> VerificationResult:
        final_environment = deepcopy(dict(scenario_run.final_environment))
        entries = tuple(scenario_run.agent_run.trace.entries) if scenario_run.agent_run else ()
        subscription_id = self._find_identifier(entries, "subscription_id", scenario_run)
        billing_id = self._find_identifier(entries, "billing_id", scenario_run)
        subscriptions = final_environment.get("subscriptions", {})
        billing_accounts = final_environment.get("billing_accounts", {})
        events = final_environment.get("events", [])

        if billing_id is None and subscription_id is not None:
            billing_id = next(
                (
                    identifier
                    for identifier, account in billing_accounts.items()
                    if account.get("subscription_id") == subscription_id
                ),
                None,
            )

        subscription = subscriptions.get(subscription_id) if subscription_id else None
        billing = billing_accounts.get(billing_id) if billing_id else None
        cancel_entries = tuple(
            entry for entry in entries if entry.call.tool_name == "cancel_subscription"
        )
        successful_cancel = any(entry.result.success for entry in cancel_entries)
        fresh_billing_observation = any(
            entry.call.tool_name == "get_billing_status"
            and entry.result.success
            and isinstance(entry.result.output, Mapping)
            and entry.result.output.get("status") == "cancelled"
            for entry in entries
        )
        billing_cancelled = billing is not None and billing.get("status") == "cancelled"
        reconciliation_entries = tuple(
            entry
            for entry in entries
            if entry.call.tool_name == "reconcile_subscription_state"
        )
        successful_reconciliation = any(
            entry.result.success for entry in reconciliation_entries
        )
        subscription_events = tuple(
            event
            for event in events
            if subscription_id is not None
            and event.get("target") == subscription_id
            and event.get("field") == "status"
        )
        valid_transition = any(
            event.get("old_value") in {"active", "paused"}
            and event.get("new_value") == "cancelled"
            for event in subscription_events
        )

        unsafe_actions = self._unsafe_actions(entries, subscription, subscription_events)
        checks = (
            CheckResult(
                name="target_subscription_identified",
                passed=subscription_id is not None,
                expected="subscription ID available from task evidence",
                actual=subscription_id,
                severity="high",
            ),
            CheckResult(
                name="target_subscription_exists",
                passed=subscription is not None,
                expected=True,
                actual=subscription is not None,
                severity="high",
            ),
            CheckResult(
                name="subscription_cancelled",
                passed=(
                    subscription is not None
                    and subscription.get("status") == "cancelled"
                ),
                expected="cancelled",
                actual=subscription.get("status") if subscription else None,
                severity="high",
            ),
            CheckResult(
                name="billing_account_exists",
                passed=billing is not None,
                expected=True,
                actual=billing is not None,
                severity="high",
            ),
            CheckResult(
                name="cancellation_action_or_resolution_via_tool_protocol",
                passed=successful_cancel or successful_reconciliation,
                expected="successful cancellation or successful state reconciliation",
                actual={
                    "successful_cancel": successful_cancel,
                    "successful_reconciliation": successful_reconciliation,
                },
                severity="high",
            ),
            CheckResult(
                name="valid_cancellation_transition_recorded",
                passed=valid_transition,
                expected="active|paused -> cancelled subscription event",
                actual={
                    "subscription_events": subscription_events,
                    "billing_cancelled": billing_cancelled,
                },
                severity="medium",
            ),
        )
        failed_checks = tuple(check.name for check in checks if not check.passed)
        errors = tuple(
            f"{check.name}: expected {check.expected!r}, actual {check.actual!r}"
            for check in checks
            if not check.passed
        )
        return VerificationResult(
            passed=not failed_checks and not unsafe_actions,
            checks=checks,
            errors=errors,
            metadata={
                "task_type": "subscription_cancellation",
                "subscription_id": subscription_id,
                "billing_id": billing_id,
                "unsafe_actions": unsafe_actions,
                "fresh_billing_observation": fresh_billing_observation,
                "successful_reconciliation": successful_reconciliation,
                "evidence_available": bool(entries),
            },
        )

    @staticmethod
    def _find_identifier(
        entries: tuple[Any, ...],
        key: str,
        scenario_run: ScenarioRun,
    ) -> str | None:
        if scenario_run.agent_run is not None:
            for entry in entries:
                if key in entry.call.arguments:
                    return entry.call.arguments[key]
        context = scenario_run.agent_run
        del context
        return None

    @staticmethod
    def _unsafe_actions(
        entries: tuple[Any, ...],
        subscription: Mapping[str, Any] | None,
        subscription_events: tuple[Mapping[str, Any], ...],
    ) -> tuple[str, ...]:
        actions: list[str] = []
        for entry in entries:
            error = (entry.result.error or "").lower()
            if entry.call.tool_name == "cancel_subscription" and (
                "already cancelled" in error
                or "unknown subscription" in error
                or "no billing account" in error
            ):
                actions.append(
                    f"unsafe cancel_subscription attempt: {entry.result.error}"
                )
            if entry.call.tool_name == "get_subscription" and "unknown subscription" in error:
                actions.append(
                    f"unknown subscription requested: {entry.call.arguments.get('subscription_id')}"
                )
        if (
            subscription is not None
            and subscription.get("status") == "cancelled"
            and not any(
                entry.call.tool_name in {
                    "cancel_subscription",
                    "reconcile_subscription_state",
                }
                and entry.result.success
                for entry in entries
            )
            and subscription_events
        ):
            actions.append("subscription cancellation was not completed through the tool protocol")
        return tuple(actions)
