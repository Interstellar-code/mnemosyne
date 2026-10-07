"""mnemosyne_recall / prefetch merge the owner's current canonical facts (step 11b)."""
from __future__ import annotations

import json

from hermes_memory_provider import MnemosyneMemoryProvider

FLIGHTS = "BLR to DEL domestic flights: Air India AI2804 outbound Aug 15, AI2817 return Aug 23."


def _provider(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    home = tmp_path / "home"
    home.mkdir()
    p = MnemosyneMemoryProvider()
    p.initialize(session_id="s1", hermes_home=str(home), agent_identity="hermes-switch")
    p._beam.canonical.remember("hermes-switch", "trip-aug-2026", "blr-del-flights", FLIGHTS)
    return p


def _recall(p, query, limit=5):
    return json.loads(p.handle_tool_call("mnemosyne_recall", {"query": query, "limit": limit}))["results"]


def test_trip_query_returns_canonical_without_working_match(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    res = _recall(p, "Which Air India flights are BLR-DEL?")
    top = res[0]
    assert top["source"] == "canonical" and top["tier"] == "canonical"
    assert top["category"] == "trip-aug-2026" and top["name"] == "blr-del-flights"
    assert isinstance(top["canonical_id"], int)
    assert "AI2804" in top["content"]


def test_canonical_ranks_above_fresh_partial_working_match(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    p._beam.remember("Booked a hotel; Air India app keeps crashing today.", source="conversation", importance=0.9)
    res = _recall(p, "Which Air India flights are BLR-DEL?")
    assert res[0]["source"] == "canonical"
    assert any(r.get("tier") == "working" for r in res[1:])
    assert sum(r["source"] == "canonical" for r in res) == 1  # deduped, capped


def test_other_owner_and_closed_facts_excluded(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    store = p._beam.canonical
    store.remember("other-profile", "trip", "olaf", "Olaf is the travel companion of someone else.")
    store.remember("hermes-switch", "people", "olaf-old", "Olaf travel companion retired fact.")
    store.forget("hermes-switch", "people", "olaf-old")
    res = _recall(p, "Who is Olaf travel companion?")
    assert not [r for r in res if r.get("source") == "canonical"]


def test_flag_off_restores_old_behaviour(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    p._recall_include_canonical = False
    assert not [r for r in _recall(p, "Which Air India flights are BLR-DEL?") if r.get("source") == "canonical"]
    assert "AI2804" not in p.prefetch("Which Air India flights are BLR-DEL?")


def test_config_key_disables(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    p._apply_provider_config({"recall_include_canonical": "false"})
    assert p._recall_include_canonical is False


def test_prefetch_injects_canonical_block(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    out = p.prefetch("Which Air India flights are BLR-DEL?")
    assert "## Mnemosyne Canonical Facts" in out
    assert "[trip-aug-2026] blr-del-flights:" in out and "AI2804" in out


def test_low_overlap_rows_not_returned(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    # one of four meaningful tokens ("india") -> below the 0.5 threshold
    assert p._beam.canonical_hits("India cooking recipes vegetarian", "hermes-switch") == []
