"""Validated actions and observations for the opt-in bounded agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


ActionName = Literal["call", "finish", "clarify", "unsupported"]
_FORBIDDEN = {"sql", "query", "statement", "python", "code", "script", "expression"}
ACTION_SCHEMA_HINT = (
    'Legal shapes: {"action":"call","capability":"registered_capability","arguments":{}}; '
    '{"action":"finish","reason":"covered"}; '
    '{"action":"clarify","reason":"missing scope"}; '
    '{"action":"unsupported","reason":"outside capabilities"}. '
    'Use exactly call, finish, clarify, or unsupported for action, not a tool name or status. '
    'Only action, capability, arguments, reason are allowed fields. '
    'For call, replace registered_capability with an allowed name and use an arguments object. '
    'For other actions, omit capability and arguments. '
    'reason, if present, must be a string of at most 500 characters.'
)


class ActionValidationError(ValueError):
    def __init__(self, detail_code: str, message: str, *, code: str = "invalid_action_schema"):
        super().__init__(message)
        self.code = code
        self.detail_code = detail_code


def _validate_values(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN:
                raise ActionValidationError("forbidden_argument", "argument is not allowed")
            _validate_values(nested)
    elif isinstance(value, list):
        for item in value:
            _validate_values(item)
    elif isinstance(value, str) and ";" in value:
        raise ActionValidationError("statement_separator", "statement separators are not allowed")


@dataclass(frozen=True)
class AgentAction:
    action: ActionName
    capability: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or self.action not in {"call", "finish", "clarify", "unsupported"}:
            raise ActionValidationError("invalid_action", "invalid action")
        if not isinstance(self.arguments, dict):
            raise ActionValidationError("arguments_type", "arguments must be an object")
        if not isinstance(self.reason, str):
            raise ActionValidationError("reason_type", "reason must be a string")
        if len(self.reason) > 500:
            raise ActionValidationError("reason_length", "reason must be at most 500 characters")
        if self.action != "call" and self.arguments:
            raise ActionValidationError("non_call_arguments", "non-call action cannot carry arguments")
        if self.action == "call":
            if not isinstance(self.capability, str):
                raise ActionValidationError("capability_type", "call action requires a string capability")
            if not self.capability or not self.capability.replace("_", "").isalnum():
                raise ActionValidationError("capability_name", "call action requires a capability")
        if self.action != "call" and self.capability is not None:
            raise ActionValidationError("non_call_capability", "only call action may specify a capability")
        for key, value in self.arguments.items():
            # Knowledge text is data, not an executable statement. Only this
            # declared text field accepts natural-language punctuation; all
            # other fields retain the recursive execution-argument checks.
            if self.capability == "knowledge_glossary" and key == "term" and isinstance(value, str):
                continue
            _validate_values({key: value})

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "AgentAction":
        if not isinstance(value, dict):
            raise ActionValidationError("action_object_required", "action must be an object")
        if not isinstance(value.get("action"), str) or value["action"] not in {"call", "finish", "clarify", "unsupported"}:
            raise ActionValidationError("invalid_action", "invalid agent action")
        if set(value) - {"action", "capability", "arguments", "reason"}:
            raise ActionValidationError("unknown_fields", "unknown action fields")
        if "arguments" in value and not isinstance(value["arguments"], dict):
            raise ActionValidationError("arguments_type", "arguments must be an object")
        return cls(
            action=value["action"],
            capability=value.get("capability"),
            arguments=dict(value.get("arguments") or {}),
            reason=value.get("reason", ""),
        )


@dataclass(frozen=True)
class Observation:
    id: str
    capability: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    requirement_id: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    reason_code: str | None = None
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class LoopBudget:
    max_tool_calls: int = 8
    max_decisions: int = 10
    max_retries: int = 1
    max_decision_repairs: int = 1


class CoverageLedger:
    """Only validated observations for an exact requirement can satisfy coverage."""
    def __init__(self, requirement_ids: tuple[str, ...]) -> None:
        self.required_ids = requirement_ids
        self._observations: list[Observation] = []

    def observe(self, observation: Observation) -> None:
        self._observations.append(observation)

    @property
    def complete(self) -> bool:
        completed = {o.requirement_id for o in self._observations if o.status == "complete"}
        return bool(self.required_ids) and all(r in completed for r in self.required_ids)

    @property
    def missing(self) -> list[str]:
        completed = {o.requirement_id for o in self._observations if o.status == "complete"}
        return [r for r in self.required_ids if r not in completed]

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)


def action_fingerprint(action: AgentAction) -> str:
    return json.dumps(
        {"capability": action.capability, "arguments": action.arguments},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
