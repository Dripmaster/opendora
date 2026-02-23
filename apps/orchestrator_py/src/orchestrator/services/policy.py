from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class PolicyDecision:
    allow: bool
    require_hitl: bool
    reason_code: str


_RISKY_INTENT_KEYWORDS = (
    "delete",
    "drop table",
    "rm -rf",
    "exfiltrate",
    "credential",
    "credentials",
    "secret",
    "token",
    "password",
    "api key",
    "ssh key",
    "private key",
    "삭제",
    "유출",
    "자격 증명",
)


def evaluate_execution_policy(
    *,
    channel_type: Literal["dm", "guild"],
    is_allowlisted: bool,
    codex_sandbox: str,
    request_intent: str,
) -> PolicyDecision:
    risky_intent = has_risky_intent(request_intent)

    if codex_sandbox == "danger-full-access":
        if not is_allowlisted:
            if channel_type == "dm":
                return PolicyDecision(
                    allow=False,
                    require_hitl=True,
                    reason_code="danger_full_access_dm_requires_allowlist_and_hitl",
                )
            return PolicyDecision(
                allow=False,
                require_hitl=True,
                reason_code="danger_full_access_requires_allowlist_and_hitl",
            )
        return PolicyDecision(
            allow=True,
            require_hitl=True,
            reason_code="danger_full_access_requires_hitl",
        )

    if risky_intent:
        return PolicyDecision(
            allow=True,
            require_hitl=True,
            reason_code="risky_intent_requires_hitl",
        )
    return PolicyDecision(allow=True, require_hitl=False, reason_code="allowed")


def has_risky_intent(request_intent: str) -> bool:
    normalized = str(request_intent or "").lower()
    return any(keyword in normalized for keyword in _RISKY_INTENT_KEYWORDS)
