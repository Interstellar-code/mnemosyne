"""
Safety gate for Hermes memory provider write tools.

Every write tool defaults to dry_run=True in AGENT context (any LLM-driven context).
In OPERATOR context (agent_context == "cli" or None) the default flips to dry_run=False.

Destructive tools additionally require a confirm_token that is:
  - Generated during dry-run
  - UUID-derived, 16 chars, base64url
  - 5-minute TTL, single-use, session-scoped, in-memory only

Public interface (exact names are integration contract):
  WRITE_TOOLS: frozenset[str]
  DESTRUCTIVE_TOOLS: frozenset[str]
  class SafetyGate
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------

WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "mnemosyne_remember",
        "mnemosyne_update",
        "mnemosyne_forget",
        "mnemosyne_triple_add",
        "mnemosyne_graph_link",
        "mnemosyne_invalidate",
        "mnemosyne_import",
        "mnemosyne_scratchpad_write",
        "mnemosyne_scratchpad_clear",
        "mnemosyne_sync_push",
        "mnemosyne_sync_pull",
        "memory_create_page",
        "memory_update_page",
    }
)

DESTRUCTIVE_TOOLS: frozenset[str] = frozenset(
    {
        "mnemosyne_forget",
        "mnemosyne_scratchpad_clear",
        "mnemosyne_invalidate",
        "mnemosyne_import",
        "mnemosyne_sync_pull",
        "memory_create_page",
        "memory_update_page",
    }
)

# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------

_APPLY_HINT = "Re-run with dry_run=false to apply."


@dataclass
class _Token:
    value: str
    expires_at: float
    used: bool = False


class _TokenStore:
    """In-memory, session-scoped token store. Not persisted."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._tokens: dict[str, _Token] = {}

    def generate(self) -> str:
        raw = uuid.uuid4().bytes[:12]
        token = base64.urlsafe_b64encode(raw).decode("ascii")[:16]
        self._tokens[token] = _Token(value=token, expires_at=time.monotonic() + self._ttl)
        return token

    def verify_and_consume(self, token: str) -> tuple[bool, str]:
        """Returns (ok, error_message). On success the token is immediately invalidated."""
        entry = self._tokens.get(token)
        if entry is None:
            return False, "confirm_token is invalid or not found"
        if entry.used:
            return False, "confirm_token has already been used (replay rejected)"
        if time.monotonic() > entry.expires_at:
            del self._tokens[token]
            return False, "confirm_token has expired"
        entry.used = True
        return True, ""


# ---------------------------------------------------------------------------
# Context detection
# ---------------------------------------------------------------------------

_OPERATOR_CONTEXTS: frozenset[str | None] = frozenset({"cli", None})


def _is_operator(agent_context: str | None) -> bool:
    """cli or None → operator; any other LLM-driven context → agent."""
    return agent_context in _OPERATOR_CONTEXTS


# ---------------------------------------------------------------------------
# SafetyGate
# ---------------------------------------------------------------------------


class SafetyGate:
    """Wraps handle_tool_call dispatch with dry-run and confirm-token safety."""

    def __init__(self, agent_context: str | None, *, ttl_seconds: int = 300, enabled: bool = True) -> None:
        self._agent_context = agent_context
        self._store = _TokenStore(ttl_seconds=ttl_seconds)
        # When disabled, the gate is a transparent passthrough: no dry-run flip,
        # no token enforcement. Lets the fork stay a drop-in superset of upstream
        # Mnemosyne unless a host binding explicitly opts the contract in.
        self._enabled = enabled

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_operator(self) -> bool:
        return _is_operator(self._agent_context)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def needs_wrap(self, tool_name: str) -> bool:
        """True if tool is in WRITE_TOOLS and therefore subject to dry-run logic.
        Always False when the gate is disabled (transparent passthrough)."""
        return self._enabled and tool_name in WRITE_TOOLS

    # ------------------------------------------------------------------
    # Core guard
    # ------------------------------------------------------------------

    def guard(self, tool_name: str, args: dict, apply_fn: Callable[[], str]) -> str:
        """
        Main entry point. Returns a JSON string.

        Steps:
          1. Non-write tool → pass through unchanged.
          2. Resolve effective dry_run flag.
          3. dry_run=True  → build preview; for destructive tools generate token; return dry-run shape.
          4. dry_run=False, destructive → verify confirm_token; on fail return error JSON; on success
             invalidate token and call apply_fn().
          5. dry_run=False, non-destructive → call apply_fn().
        """
        # Step 0 – disabled gate is a transparent passthrough
        if not self._enabled:
            return apply_fn()

        # Step 1 – non-write passthrough
        if tool_name not in WRITE_TOOLS:
            return apply_fn()

        # Step 2 – resolve dry_run
        if "dry_run" in args:
            dry_run = bool(args["dry_run"])
        else:
            dry_run = not self.is_operator  # AGENT default=True, OPERATOR default=False

        is_destructive = tool_name in DESTRUCTIVE_TOOLS

        if dry_run:
            # Step 3 – dry-run response
            preview = {k: v for k, v in args.items() if k not in ("dry_run", "confirm_token")}
            response: dict = {
                "dry_run": True,
                "action": tool_name,
                "preview": preview,
                "apply_hint": _APPLY_HINT,
            }
            if is_destructive:
                response["confirm_token"] = self._store.generate()
            return json.dumps(response)

        # Not dry_run from here
        if is_destructive:
            # Step 4 – require and verify confirm_token
            token = args.get("confirm_token")
            if not token:
                return json.dumps(
                    {
                        "error": "confirm_token is required for destructive operations. "
                        "Perform a dry-run first to obtain a token.",
                        "action": tool_name,
                    }
                )
            ok, err = self._store.verify_and_consume(token)
            if not ok:
                return json.dumps({"error": err, "action": tool_name})
            # Token consumed – execute
            return apply_fn()

        # Step 5 – non-destructive, not dry_run
        return apply_fn()
