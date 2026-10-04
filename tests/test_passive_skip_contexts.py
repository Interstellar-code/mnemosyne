"""hermes-agent#251: cron is a *passive* context.

Explicit tool calls (a cron job's mnemosyne_remember) must keep working, while
implicit memory (sync_turn, prefetch, auto/session-end sleep) stays off.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import hermes_memory_provider as hmp
from hermes_memory_provider import MnemosyneMemoryProvider


def _working_rows(provider) -> int:
    return provider._beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    for var in ("MNEMOSYNE_SKIP_CONTEXTS", "MNEMOSYNE_PASSIVE_SKIP_CONTEXTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(hmp, "_active_provider_count", 0)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_read_config_key", lambda self, key: None)
    made = []
    yield made
    for p in made:
        p.shutdown()
    from hermes_memory_provider.hermes_llm_adapter import unregister_hermes_host_llm

    unregister_hermes_host_llm()  # initialize() registers the process-global backend


def _init(made, context, **kwargs):
    provider = MnemosyneMemoryProvider()
    made.append(provider)
    provider.initialize(session_id=f"s-{context}", agent_context=context, **kwargs)
    return provider


def test_cron_opens_beam_but_records_nothing_implicitly(isolated):
    p = _init(isolated, "cron")
    assert p._beam is not None
    assert p._beam.agent_context == "cron"  # beam.sleep() model-refresh guard sees it
    p.sync_turn("[SYSTEM: cron preamble] check the feed", "Nothing new today.")
    assert _working_rows(p) == 0
    assert p.prefetch("check the feed") == ""
    with patch.object(hmp, "_get_beam_class") as beam_cls:
        p.on_session_end([])
    beam_cls.assert_not_called()  # no session-end consolidation


def test_cron_explicit_remember_writes(isolated):
    p = _init(isolated, "cron")
    out = json.loads(p.handle_tool_call("mnemosyne_remember", {"content": "Weekly B1 recap: Konjunktiv II"}))
    assert "error" not in out and out.get("status") != "memory_unavailable", out
    assert _working_rows(p) == 1


def test_cron_sleep_tool_still_blocked(isolated):
    p = _init(isolated, "cron")
    out = json.loads(p.handle_tool_call("mnemosyne_sleep", {}))
    assert out["reason"] == "reflect_disabled_for_cron"


def test_primary_still_records(isolated):
    p = _init(isolated, "primary")
    p.sync_turn("remember that I like blue", "Noted, you like blue.")
    assert _working_rows(p) > 0


def test_cron_in_skip_contexts_is_full_skip(isolated):
    p = _init(isolated, "cron", skip_contexts="cron,subagent")
    assert p._beam is None


@pytest.mark.parametrize(
    ("env", "config", "kwarg", "expected"),
    [
        (None, None, None, {"cron"}),                     # default
        ("background", None, None, {"background"}),       # env
        ("background", "flush", None, {"flush"}),         # config > env
        ("background", "flush", "", set()),               # kwargs > config ("" = none)
        (None, ["cron", " flush "], None, {"cron", "flush"}),  # list form
    ],
)
def test_passive_skip_contexts_precedence(monkeypatch, env, config, kwarg, expected):
    if env is None:
        monkeypatch.delenv("MNEMOSYNE_PASSIVE_SKIP_CONTEXTS", raising=False)
    else:
        monkeypatch.setenv("MNEMOSYNE_PASSIVE_SKIP_CONTEXTS", env)
    p = MnemosyneMemoryProvider()
    monkeypatch.setattr(
        p, "_read_config_key", lambda key: config if key == "passive_skip_contexts" else None)
    p._apply_provider_config({} if kwarg is None else {"passive_skip_contexts": kwarg})
    assert p._passive_skip_contexts == expected


def test_cron_shutdown_never_unregisters_host_llm(isolated):
    """Cron runs inside the gateway: its shutdown must not pull the host LLM, whether or
    not a primary instance is still active (the cron instance is the last one here)."""
    primary = _init(isolated, "primary")
    cron = _init(isolated, "cron")
    assert hmp._active_provider_count == 2
    with patch("hermes_memory_provider.hermes_llm_adapter.unregister_hermes_host_llm") as unreg:
        primary.shutdown()
        cron.shutdown()
    unreg.assert_not_called()
    assert hmp._active_provider_count == 0


def test_failed_init_keeps_host_llm_while_another_primary_is_active(isolated):
    """A second instance whose initialize() fails must not unregister the
    process-global host LLM out from under a still-active primary."""
    from mnemosyne.core.llm_backends import get_host_llm_backend

    _init(isolated, "primary")
    assert get_host_llm_backend() is not None
    bad = MnemosyneMemoryProvider()
    with patch.object(bad, "_configured_tool_schemas", side_effect=ValueError("bad tools")):
        with pytest.raises(ValueError):
            bad.initialize(session_id="s-bad", agent_context="primary")
    assert hmp._active_provider_count == 1
    assert get_host_llm_backend() is not None
