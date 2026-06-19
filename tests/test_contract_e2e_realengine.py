"""Phase 4 — matrix-memory contract validated against the REAL Mnemosyne engine.

Unlike hermes_memory_provider/tests/ (mocked remember/graph fns), this exercises
the live fastembed + sqlite-vec stack so we prove the contract actually lands
content in the vector store and recalls it semantically.

Requires the `embeddings` extra (fastembed, sqlite-vec). Skips cleanly if absent.
Run: uv run --with pytest python -m pytest tests/test_contract_e2e_realengine.py -q
"""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastembed", reason="embeddings extra not installed")
pytest.importorskip("sqlite_vec", reason="embeddings extra not installed")

from hermes_memory_provider import MnemosyneMemoryProvider


def _call(p, name, args):
    return json.loads(p.handle_tool_call(name, args))


@pytest.fixture()
def provider(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.setenv("MNEMOSYNE_MATRIX_SAFETY", "1")  # contract ON
    monkeypatch.delenv("MNEMOSYNE_NO_EMBEDDINGS", raising=False)  # embeddings ON

    hermes_home = tmp_path / "hermes"
    (hermes_home / "memories").mkdir(parents=True, exist_ok=True)
    (hermes_home / "memories" / "MEMORY.md").write_text(
        "§ Travel\nThe user spent a week in Kyoto and loved the temples\n",
        encoding="utf-8",
    )

    p = MnemosyneMemoryProvider()
    p.initialize(session_id="e2e", hermes_home=str(hermes_home))
    assert p._beam is not None
    yield p
    p.shutdown()


# --- core engine sanity -----------------------------------------------------

def test_recall_through_real_engine(provider):
    # NOTE (Phase 4 finding): Mnemosyne recall is a *lexically-gated* hybrid —
    # a query with zero shared tokens is filtered out before vector scoring
    # (lexical relevance gate, commits e5823ec/92d65f4). With >=1 lexical anchor,
    # vector reranking applies ("where is the tower" matches "...is in Paris").
    provider._beam.remember(content="The Eiffel Tower is in Paris", source="fact")
    out = _call(provider, "mnemosyne_recall", {"query": "where is the tower located", "tier": "2"})
    assert out["count"] >= 1
    assert any("Eiffel" in r["content"] for r in out["results"])


def test_all_tools_registered(provider):
    names = {s["name"] for s in provider.get_tool_schemas()}
    assert {"memory_create_page", "memory_update_page", "memory_show_page"} <= names
    assert len(names) == 36  # 33 Mnemosyne tools + 3 contract wiki tools


# --- Tier 1 migration into the real vector store ----------------------------

def test_migration_seeded_memory_is_semantically_recallable(provider):
    # MEMORY.md seeded "Kyoto ... temples"; migration ran in initialize().
    out = _call(provider, "mnemosyne_recall", {"query": "Kyoto temples visit", "tier": "2"})
    assert out["count"] >= 1
    assert any("Kyoto" in r["content"] for r in out["results"])


def test_tier1_and_tier_all(provider):
    t1 = _call(provider, "mnemosyne_recall", {"query": "Kyoto", "tier": "1"})
    assert t1["tiers"] == "1" and t1["count"] >= 1 and t1["results"][0]["tier"] == 1
    allt = _call(provider, "mnemosyne_recall", {"query": "Kyoto", "tier": "all"})
    assert allt["tiers"] == "all" and allt["results"][0]["tier"] == 1  # Tier 1 prepended


def test_tier4_rejected(provider):
    out = _call(provider, "mnemosyne_recall", {"query": "x", "tier": "4"})
    assert "error" in out and "tier=4" in out["error"]


# --- wiki bridge lands content in the real vector store ---------------------

def test_wiki_create_bridges_into_vector_store(provider):
    content = "---\ntitle: Atlas\ntags: project\n---\nProject Atlas ships the new billing pipeline."
    dry = _call(provider, "memory_create_page", {"path": "entities/atlas.md", "content": content})
    applied = _call(provider, "memory_create_page",
                    {"path": "entities/atlas.md", "content": content,
                     "dry_run": False, "confirm_token": dry["confirm_token"]})
    assert applied["status"] == "created" and applied["indexed"] is True
    # the page body must now be recallable via Tier 2 (proves bridge -> vector store)
    out = _call(provider, "mnemosyne_recall", {"query": "Atlas billing pipeline", "tier": "2"})
    assert any("Atlas" in r["content"] for r in out["results"])


# --- two-phase destructive flow on a real row -------------------------------

def test_two_phase_forget_on_real_memory(provider):
    mid = provider._beam.remember(content="ephemeral row to be deleted", source="fact")
    dry = _call(provider, "mnemosyne_forget", {"memory_id": mid})
    assert dry["dry_run"] is True and dry["confirm_token"]
    applied = _call(provider, "mnemosyne_forget",
                    {"memory_id": mid, "dry_run": False, "confirm_token": dry["confirm_token"]})
    assert applied.get("status") in ("deleted", "not_found")
    # replay rejected
    replay = _call(provider, "mnemosyne_forget",
                   {"memory_id": mid, "dry_run": False, "confirm_token": dry["confirm_token"]})
    assert "error" in replay


# --- admin / KG tools return gracefully (review #27) ------------------------

def test_admin_and_kg_tools_return_cleanly(provider):
    for tool, args in [
        ("mnemosyne_stats", {}),
        ("mnemosyne_diagnose", {}),
        ("mnemosyne_sync_status", {}),
        ("mnemosyne_graph_query", {"query": "anything"}),
    ]:
        out = _call(provider, tool, args)
        assert isinstance(out, dict)  # structured response, no crash
