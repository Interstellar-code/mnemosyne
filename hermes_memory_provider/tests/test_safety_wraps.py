"""
Tests for hermes_memory_provider/safety.py

Run:
    cd /Volumes/Ext-nvme/Development/mnemosyne
    python -m pytest hermes_memory_provider/tests/test_safety_wraps.py -q
"""

from __future__ import annotations

import json
import time

import pytest

from hermes_memory_provider.safety import (
    DESTRUCTIVE_TOOLS,
    WRITE_TOOLS,
    SafetyGate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = json.dumps({"applied": True, "tool": "sentinel"})


def _apply() -> str:
    return _SENTINEL


def _gate(agent_context: str | None, ttl: int = 300) -> SafetyGate:
    return SafetyGate(agent_context, ttl_seconds=ttl)


# Stable names for parametrize
_WRITE_NON_DESTRUCTIVE = sorted(WRITE_TOOLS - DESTRUCTIVE_TOOLS)
_DESTRUCTIVE = sorted(DESTRUCTIVE_TOOLS)


# ---------------------------------------------------------------------------
# 1. Context detection
# ---------------------------------------------------------------------------


class TestContextDetection:
    def test_cli_is_operator(self):
        assert _gate("cli").is_operator is True

    def test_none_is_operator(self):
        assert _gate(None).is_operator is True

    def test_primary_is_agent(self):
        assert _gate("primary").is_operator is False

    def test_cron_is_agent(self):
        assert _gate("cron").is_operator is False

    def test_subagent_is_agent(self):
        assert _gate("subagent").is_operator is False

    def test_background_is_agent(self):
        assert _gate("background").is_operator is False

    def test_skill_loop_is_agent(self):
        assert _gate("skill_loop").is_operator is False

    def test_flush_is_agent(self):
        assert _gate("flush").is_operator is False


# ---------------------------------------------------------------------------
# 2. needs_wrap
# ---------------------------------------------------------------------------


class TestNeedsWrap:
    def test_write_tool_needs_wrap(self):
        g = _gate("cli")
        for t in WRITE_TOOLS:
            assert g.needs_wrap(t), t

    def test_read_tool_no_wrap(self):
        g = _gate("cli")
        assert g.needs_wrap("mnemosyne_search") is False
        assert g.needs_wrap("mnemosyne_recall") is False
        assert g.needs_wrap("mnemosyne_diagnostics") is False


# ---------------------------------------------------------------------------
# 3. Non-write passthrough
# ---------------------------------------------------------------------------


class TestNonWritePassthrough:
    def test_non_write_passes_through_agent(self):
        g = _gate("primary")
        result = g.guard("mnemosyne_search", {}, _apply)
        assert result == _SENTINEL

    def test_non_write_passes_through_operator(self):
        g = _gate("cli")
        result = g.guard("mnemosyne_search", {}, _apply)
        assert result == _SENTINEL

    def test_non_write_ignores_args(self):
        g = _gate("primary")
        result = g.guard("mnemosyne_diagnostics", {"dry_run": True}, _apply)
        assert result == _SENTINEL


# ---------------------------------------------------------------------------
# 4. dry_run defaults by context
# ---------------------------------------------------------------------------


class TestDryRunDefaults:
    def test_agent_defaults_dry_run_true(self):
        g = _gate("primary")
        result = json.loads(g.guard("mnemosyne_remember", {"content": "hello"}, _apply))
        assert result["dry_run"] is True

    def test_operator_defaults_dry_run_false(self):
        g = _gate("cli")
        result = g.guard("mnemosyne_remember", {"content": "hello"}, _apply)
        assert result == _SENTINEL

    def test_explicit_dry_run_false_overrides_agent(self):
        # Agent can opt out of dry-run by passing dry_run=False (plus token for destructive)
        g = _gate("primary")
        result = g.guard(
            "mnemosyne_remember",  # non-destructive
            {"content": "hello", "dry_run": False},
            _apply,
        )
        assert result == _SENTINEL

    def test_explicit_dry_run_true_overrides_operator(self):
        g = _gate("cli")
        result = json.loads(
            g.guard("mnemosyne_remember", {"content": "hello", "dry_run": True}, _apply)
        )
        assert result["dry_run"] is True

    def test_none_context_defaults_dry_run_false(self):
        g = _gate(None)
        result = g.guard("mnemosyne_remember", {"content": "x"}, _apply)
        assert result == _SENTINEL


# ---------------------------------------------------------------------------
# 5. Dry-run response shape
# ---------------------------------------------------------------------------


class TestDryRunShape:
    def test_non_destructive_dry_run_shape(self):
        g = _gate("primary")
        result = json.loads(g.guard("mnemosyne_remember", {"content": "hello"}, _apply))
        assert result["dry_run"] is True
        assert result["action"] == "mnemosyne_remember"
        assert "preview" in result
        assert result["apply_hint"] == "Re-run with dry_run=false to apply."
        assert "confirm_token" not in result

    def test_preview_excludes_dry_run_and_token_keys(self):
        g = _gate("primary")
        args = {"content": "hello", "dry_run": True, "confirm_token": "abc"}
        result = json.loads(g.guard("mnemosyne_remember", args, _apply))
        assert "dry_run" not in result["preview"]
        assert "confirm_token" not in result["preview"]
        assert result["preview"]["content"] == "hello"

    def test_destructive_dry_run_includes_token(self):
        g = _gate("primary")
        result = json.loads(g.guard("mnemosyne_forget", {"key": "k1"}, _apply))
        assert result["dry_run"] is True
        assert "confirm_token" in result
        token = result["confirm_token"]
        assert isinstance(token, str)
        assert len(token) == 16

    def test_token_is_base64url(self):
        import re
        g = _gate("primary")
        result = json.loads(g.guard("mnemosyne_forget", {"key": "k1"}, _apply))
        token = result["confirm_token"]
        assert re.fullmatch(r"[A-Za-z0-9_\-]{16}", token)


# ---------------------------------------------------------------------------
# 6. Apply blocked without token (destructive)
# ---------------------------------------------------------------------------


class TestApplyBlockedWithoutToken:
    def test_destructive_apply_without_token_returns_error(self):
        g = _gate("cli")  # operator so no default dry_run
        result = json.loads(
            g.guard("mnemosyne_forget", {"key": "k1", "dry_run": False}, _apply)
        )
        assert "error" in result
        assert result.get("action") == "mnemosyne_forget"

    def test_destructive_apply_with_wrong_token_returns_error(self):
        g = _gate("cli")
        result = json.loads(
            g.guard(
                "mnemosyne_forget",
                {"key": "k1", "dry_run": False, "confirm_token": "badtoken1234567"},
                _apply,
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# 7. Apply succeeds with valid token
# ---------------------------------------------------------------------------


class TestApplyWithValidToken:
    def _obtain_token(self, gate: SafetyGate, tool: str, args: dict) -> str:
        dry = json.loads(gate.guard(tool, {**args, "dry_run": True}, _apply))
        return dry["confirm_token"]

    def test_destructive_apply_succeeds_with_valid_token(self):
        g = _gate("primary")
        token = self._obtain_token(g, "mnemosyne_forget", {"key": "k1"})
        result = g.guard(
            "mnemosyne_forget",
            {"key": "k1", "dry_run": False, "confirm_token": token},
            _apply,
        )
        assert result == _SENTINEL

    @pytest.mark.parametrize("tool", _DESTRUCTIVE)
    def test_all_destructive_tools_accept_valid_token(self, tool):
        g = _gate("primary")
        token = self._obtain_token(g, tool, {"arg": "val"})
        result = g.guard(
            tool,
            {"arg": "val", "dry_run": False, "confirm_token": token},
            _apply,
        )
        assert result == _SENTINEL


# ---------------------------------------------------------------------------
# 8. Token single-use (replay protection)
# ---------------------------------------------------------------------------


class TestTokenSingleUse:
    def test_token_rejected_on_replay(self):
        g = _gate("primary")
        dry = json.loads(g.guard("mnemosyne_forget", {"key": "k1", "dry_run": True}, _apply))
        token = dry["confirm_token"]

        # First use – should succeed
        r1 = g.guard(
            "mnemosyne_forget",
            {"key": "k1", "dry_run": False, "confirm_token": token},
            _apply,
        )
        assert r1 == _SENTINEL

        # Second use – should be rejected
        r2 = json.loads(
            g.guard(
                "mnemosyne_forget",
                {"key": "k1", "dry_run": False, "confirm_token": token},
                _apply,
            )
        )
        assert "error" in r2
        assert "replay" in r2["error"].lower() or "used" in r2["error"].lower()


# ---------------------------------------------------------------------------
# 9. Token expiry
# ---------------------------------------------------------------------------


class TestTokenExpiry:
    def test_expired_token_rejected(self, monkeypatch):
        g = _gate("primary", ttl=10)
        dry = json.loads(g.guard("mnemosyne_forget", {"key": "k1", "dry_run": True}, _apply))
        token = dry["confirm_token"]

        # Advance time past TTL
        original = time.monotonic
        monkeypatch.setattr(time, "monotonic", lambda: original() + 20)

        result = json.loads(
            g.guard(
                "mnemosyne_forget",
                {"key": "k1", "dry_run": False, "confirm_token": token},
                _apply,
            )
        )
        assert "error" in result
        assert "expired" in result["error"].lower()


# ---------------------------------------------------------------------------
# 10. Operator context skips dry_run entirely for non-destructive
# ---------------------------------------------------------------------------


class TestOperatorSkipsDryRun:
    @pytest.mark.parametrize("tool", _WRITE_NON_DESTRUCTIVE)
    def test_operator_non_destructive_applies_directly(self, tool):
        g = _gate("cli")
        result = g.guard(tool, {"arg": "val"}, _apply)
        assert result == _SENTINEL

    def test_operator_destructive_needs_token(self):
        g = _gate("cli")
        result = json.loads(
            g.guard("mnemosyne_forget", {"key": "k1", "dry_run": False}, _apply)
        )
        assert "error" in result

    def test_operator_destructive_with_token_works(self):
        g = _gate("cli")
        # Generate token via explicit dry_run=True
        dry = json.loads(
            g.guard("mnemosyne_forget", {"key": "k1", "dry_run": True}, _apply)
        )
        token = dry["confirm_token"]
        result = g.guard(
            "mnemosyne_forget",
            {"key": "k1", "dry_run": False, "confirm_token": token},
            _apply,
        )
        assert result == _SENTINEL


# ---------------------------------------------------------------------------
# 11. Tool set membership sanity
# ---------------------------------------------------------------------------


class TestToolSets:
    def test_destructive_is_subset_of_write(self):
        assert DESTRUCTIVE_TOOLS <= WRITE_TOOLS

    def test_write_tools_count(self):
        assert len(WRITE_TOOLS) == 13

    def test_destructive_tools_count(self):
        assert len(DESTRUCTIVE_TOOLS) == 7
