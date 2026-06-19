"""End-to-end integration tests for the matrix-memory contract (v0.2) wired
into MnemosyneMemoryProvider.handle_tool_call.

These exercise the real provider + beam engine (no embeddings) with the safety
opt-in ON, validating spec §11.5: tier=1 passthrough, tier=4 rejection, the
two-phase dry_run -> confirm_token -> apply flow, and wiki tool registration.
"""
import json
from pathlib import Path

import pytest

from hermes_memory_provider import MnemosyneMemoryProvider


def _call(provider, name, args):
    return json.loads(provider.handle_tool_call(name, args))


@pytest.fixture()
def provider(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "mnemosyne-data"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_MATRIX_SAFETY", "1")  # opt the contract in

    hermes_home = tmp_path / "hermes"
    (hermes_home / "memories").mkdir(parents=True, exist_ok=True)
    # Seed a Tier 1 MEMORY.md (§-delimited, the built-in memory tool format)
    (hermes_home / "memories" / "MEMORY.md").write_text(
        "§ Preference\nUser prefers dark mode in all tools\n\n"
        "§ Workflow\nUser commits each fix individually\n",
        encoding="utf-8",
    )

    p = MnemosyneMemoryProvider()
    p.initialize(session_id="contract-test", hermes_home=str(hermes_home))
    assert p._beam is not None
    yield p
    p.shutdown()


def test_wiki_tools_registered(provider):
    names = {s["name"] for s in provider.get_tool_schemas()}
    assert {"memory_create_page", "memory_update_page", "memory_show_page"} <= names


def test_recall_tier4_rejected(provider):
    out = _call(provider, "mnemosyne_recall", {"query": "anything", "tier": "4"})
    assert "error" in out and "tier=4" in out["error"]


def test_recall_tier1_passthrough(provider):
    out = _call(provider, "mnemosyne_recall", {"query": "dark mode", "tier": "1"})
    assert out["tiers"] == "1"
    assert out["count"] >= 1
    assert all(r["tier"] == 1 for r in out["results"])
    assert any("dark mode" in r["entry"] for r in out["results"])


def test_recall_tier_all_merges_tier1_first(provider):
    out = _call(provider, "mnemosyne_recall", {"query": "dark mode", "tier": "all"})
    assert out["tiers"] == "all"
    assert out["results"], "expected at least the Tier 1 hit"
    assert out["results"][0]["tier"] == 1  # Tier 1 prepended


def test_remember_two_phase_with_safety_on(provider):
    # Phase 1: dry-run (default in agent context when safety opt-in is ON)
    dry = _call(provider, "mnemosyne_remember", {"content": "Switch is the active profile"})
    assert dry["dry_run"] is True
    assert dry["action"] == "mnemosyne_remember"
    # remember is non-destructive -> no confirm_token required to apply
    applied = _call(provider, "mnemosyne_remember",
                    {"content": "Switch is the active profile", "dry_run": False})
    assert applied.get("status") == "stored"


def test_create_page_two_phase_and_bridge(provider):
    # Phase 1: destructive dry-run yields a confirm_token
    dry = _call(provider, "memory_create_page",
                {"path": "entities/switch.md",
                 "content": "---\ntitle: Switch\ntags: profile\n---\nSwitch is the orchestrator."})
    assert dry["dry_run"] is True
    token = dry["confirm_token"]
    assert token

    # Phase 2: apply with the token -> page created + bridged into Mnemosyne
    applied = _call(provider, "memory_create_page",
                    {"path": "entities/switch.md",
                     "content": "---\ntitle: Switch\ntags: profile\n---\nSwitch is the orchestrator.",
                     "dry_run": False, "confirm_token": token})
    assert applied["status"] == "created"
    assert applied["indexed"] is True


def test_create_page_apply_blocked_without_token(provider):
    out = _call(provider, "memory_create_page",
                {"path": "entities/x.md", "content": "body", "dry_run": False})
    assert "error" in out  # destructive apply requires a confirm_token
