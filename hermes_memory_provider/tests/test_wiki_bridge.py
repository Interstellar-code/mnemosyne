"""
Tests for hermes_memory_provider/wiki_bridge.py

Run:
    cd /Volumes/Ext-nvme/Development/mnemosyne
    python -m pytest hermes_memory_provider/tests/test_wiki_bridge.py -q
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_memory_provider.wiki_bridge import WikiBridge


# ---------------------------------------------------------------------------
# Fake callables
# ---------------------------------------------------------------------------


class FakeRememberFn:
    """Records calls to remember_fn(content=..., source=..., tags=...)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, *, content: str, source: str, tags=None) -> None:
        self.calls.append({"content": content, "source": source, "tags": tags})


class FakeGraphLinkFn:
    """Records calls to graph_link_fn(source_id, target_id, relationship)."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, source_id: str, target_id: str, relationship: str) -> None:
        self.calls.append((source_id, target_id, relationship))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hermes_home(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def bridge(hermes_home: Path) -> WikiBridge:
    remember = FakeRememberFn()
    graph = FakeGraphLinkFn()
    b = WikiBridge(
        hermes_home=str(hermes_home),
        remember_fn=remember,
        graph_link_fn=graph,
    )
    return b


@pytest.fixture()
def wiki_dir(hermes_home: Path) -> Path:
    d = hermes_home / "matrix-memory" / "wiki"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_extracts_title_and_scalar(self):
        text = "---\ntitle: My Page\nauthor: Alice\n---\nBody here."
        meta, body = WikiBridge.parse_frontmatter(text)
        assert meta["title"] == "My Page"
        assert meta["author"] == "Alice"
        assert body.strip() == "Body here."

    def test_extracts_tags_inline_list(self):
        text = "---\ntitle: T\ntags: [ai, memory, graph]\n---\nContent."
        meta, body = WikiBridge.parse_frontmatter(text)
        assert meta["tags"] == ["ai", "memory", "graph"]

    def test_extracts_tags_comma_separated(self):
        text = "---\ntitle: T\ntags: ai, memory, graph\n---\nContent."
        meta, body = WikiBridge.parse_frontmatter(text)
        assert meta["tags"] == ["ai", "memory", "graph"]

    def test_strips_frontmatter_keeps_body(self):
        text = "---\ntitle: T\n---\nLine1\nLine2"
        meta, body = WikiBridge.parse_frontmatter(text)
        assert "title" not in body
        assert "Line1" in body
        assert "Line2" in body

    def test_no_frontmatter_returns_empty_meta_full_body(self):
        text = "Just a plain body\nwith no frontmatter."
        meta, body = WikiBridge.parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_string(self):
        meta, body = WikiBridge.parse_frontmatter("")
        assert meta == {}
        assert body == ""

    def test_frontmatter_only_no_body(self):
        text = "---\ntitle: T\n---\n"
        meta, body = WikiBridge.parse_frontmatter(text)
        assert meta["title"] == "T"
        assert body == ""


# ---------------------------------------------------------------------------
# create_page
# ---------------------------------------------------------------------------


class TestCreatePage:
    def test_writes_file(self, bridge: WikiBridge, wiki_dir: Path):
        bridge.create_page("entities/alice.md", "# Alice\nHello.")
        assert (wiki_dir / "entities" / "alice.md").exists()

    def test_calls_remember_fn_with_source_wiki(self, bridge: WikiBridge):
        bridge.create_page("concepts/foo.md", "---\ntitle: Foo\n---\nBody text.")
        calls = bridge.remember_fn.calls  # type: ignore[union-attr]
        assert len(calls) == 1
        assert calls[0]["source"] == "wiki"

    def test_remember_fn_receives_body_not_frontmatter(self, bridge: WikiBridge):
        bridge.create_page("concepts/bar.md", "---\ntitle: Bar\ntags: x\n---\nActual body.")
        calls = bridge.remember_fn.calls  # type: ignore[union-attr]
        assert "Actual body." in calls[0]["content"]
        assert "title" not in calls[0]["content"]

    def test_remember_fn_receives_tags(self, bridge: WikiBridge):
        bridge.create_page("concepts/baz.md", "---\ntags: [a, b]\n---\nContent.")
        calls = bridge.remember_fn.calls  # type: ignore[union-attr]
        assert calls[0]["tags"] == ["a", "b"]

    def test_returns_created_status_and_indexed_true(self, bridge: WikiBridge):
        result = bridge.create_page("raw/note.md", "---\ntitle: N\n---\nHi.")
        assert result["status"] == "created"
        assert result["indexed"] is True

    def test_returns_path(self, bridge: WikiBridge):
        result = bridge.create_page("raw/note2.md", "Content.")
        assert result["path"] == "raw/note2.md"


# ---------------------------------------------------------------------------
# Wikilink resolver
# ---------------------------------------------------------------------------


class TestResolveWikilinks:
    def test_resolves_existing_page(self, bridge: WikiBridge, wiki_dir: Path):
        (wiki_dir / "alice.md").write_text("# Alice", encoding="utf-8")
        links = bridge.resolve_wikilinks("See [[Alice]] for details.")
        assert len(links) == 1
        name, relpath = links[0]
        assert name == "Alice"
        assert relpath is not None
        assert "alice.md" in relpath

    def test_missing_page_returns_none(self, bridge: WikiBridge, wiki_dir: Path):
        links = bridge.resolve_wikilinks("See [[Missing Page]] here.")
        assert len(links) == 1
        name, relpath = links[0]
        assert name == "Missing Page"
        assert relpath is None

    def test_existing_link_triggers_graph_link_fn(self, bridge: WikiBridge, wiki_dir: Path):
        (wiki_dir / "bob.md").write_text("# Bob", encoding="utf-8")
        bridge.create_page("concepts/intro.md", "Mentions [[Bob]].")
        calls = bridge.graph_link_fn.calls  # type: ignore[union-attr]
        assert len(calls) == 1
        src, tgt, rel = calls[0]
        assert src == "concepts/intro.md"
        assert "bob.md" in tgt
        assert rel == "references"

    def test_missing_link_does_not_trigger_graph_link_fn(
        self, bridge: WikiBridge, wiki_dir: Path
    ):
        bridge.create_page("concepts/intro2.md", "Mentions [[Ghost]].")
        calls = bridge.graph_link_fn.calls  # type: ignore[union-attr]
        assert calls == []

    def test_dead_link_logged_in_log_md(self, bridge: WikiBridge, wiki_dir: Path):
        bridge.create_page("concepts/intro3.md", "Mentions [[Phantom]].")
        log_text = (wiki_dir / "log.md").read_text(encoding="utf-8")
        assert "dead_link" in log_text
        assert "Phantom" in log_text

    def test_multiple_links_mixed(self, bridge: WikiBridge, wiki_dir: Path):
        (wiki_dir / "real.md").write_text("# Real", encoding="utf-8")
        links = bridge.resolve_wikilinks("[[Real]] and [[Fake]].")
        resolved = [r for _, r in links if r is not None]
        unresolved = [r for _, r in links if r is None]
        assert len(resolved) == 1
        assert len(unresolved) == 1


# ---------------------------------------------------------------------------
# update_page
# ---------------------------------------------------------------------------


class TestUpdatePage:
    def test_edits_file_content(self, bridge: WikiBridge, wiki_dir: Path):
        bridge.create_page("entities/a.md", "Hello world.")
        bridge.update_page("entities/a.md", "world", "earth")
        content = (wiki_dir / "entities" / "a.md").read_text(encoding="utf-8")
        assert "earth" in content
        assert "world" not in content

    def test_rebridges_calls_remember_fn_again(self, bridge: WikiBridge, wiki_dir: Path):
        bridge.create_page("entities/b.md", "First content.")
        calls_before = len(bridge.remember_fn.calls)  # type: ignore[union-attr]
        bridge.update_page("entities/b.md", "First", "Second")
        calls_after = len(bridge.remember_fn.calls)  # type: ignore[union-attr]
        assert calls_after == calls_before + 1

    def test_update_missing_file_returns_not_found(self, bridge: WikiBridge):
        result = bridge.update_page("noexist.md", "x", "y")
        assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# One-way: bridge never writes back to source .md
# ---------------------------------------------------------------------------


class TestOneWay:
    def test_bridge_does_not_modify_source_file(self, bridge: WikiBridge, wiki_dir: Path):
        page_path = wiki_dir / "concepts" / "stable.md"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        original_content = "---\ntitle: Stable\n---\nBody stays the same."
        page_path.write_text(original_content, encoding="utf-8")

        bridge.bridge("concepts/stable.md")

        after_content = page_path.read_text(encoding="utf-8")
        assert after_content == original_content


# ---------------------------------------------------------------------------
# log.md audit
# ---------------------------------------------------------------------------


class TestLogMd:
    def test_bridge_appends_audit_line(self, bridge: WikiBridge, wiki_dir: Path):
        page_path = wiki_dir / "raw" / "note.md"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text("Some content.", encoding="utf-8")

        bridge.bridge("raw/note.md")

        log_text = (wiki_dir / "log.md").read_text(encoding="utf-8")
        assert "[bridge]" in log_text
        assert "raw/note.md" in log_text

    def test_dead_link_appends_dead_link_line(self, bridge: WikiBridge, wiki_dir: Path):
        page_path = wiki_dir / "raw" / "note2.md"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text("See [[NoSuchPage]].", encoding="utf-8")

        bridge.bridge("raw/note2.md")

        log_text = (wiki_dir / "log.md").read_text(encoding="utf-8")
        assert "[dead_link]" in log_text
        assert "NoSuchPage" in log_text

    def test_multiple_bridge_calls_append_multiple_lines(
        self, bridge: WikiBridge, wiki_dir: Path
    ):
        p1 = wiki_dir / "a.md"
        p2 = wiki_dir / "b.md"
        p1.write_text("Content A.", encoding="utf-8")
        p2.write_text("Content B.", encoding="utf-8")

        bridge.bridge("a.md")
        bridge.bridge("b.md")

        lines = (wiki_dir / "log.md").read_text(encoding="utf-8").splitlines()
        bridge_lines = [l for l in lines if "[bridge]" in l]
        assert len(bridge_lines) == 2


# ---------------------------------------------------------------------------
# Polling thread
# ---------------------------------------------------------------------------


class TestPolling:
    def test_start_and_stop_cleanly(self, hermes_home: Path):
        b = WikiBridge(hermes_home=str(hermes_home), poll_interval=0.1)
        b.start_polling()
        assert b._poll_thread is not None
        assert b._poll_thread.is_alive()
        b.stop_polling()
        assert b._poll_thread is None

    def test_scan_detects_modified_file(self, hermes_home: Path, wiki_dir: Path):
        """Test the mtime-scan method directly for determinism."""
        remember = FakeRememberFn()
        b = WikiBridge(
            hermes_home=str(hermes_home),
            remember_fn=remember,
            poll_interval=0.1,
        )

        # Create a file and seed it into the mtime cache (simulating first discovery)
        page = wiki_dir / "watched.md"
        page.write_text("Initial content.", encoding="utf-8")
        b._mtimes[page] = page.stat().st_mtime - 1  # make it look stale

        # One scan should pick it up and bridge it
        b._scan_once()

        assert len(remember.calls) == 1

    def test_thread_bridges_new_file(self, hermes_home: Path, wiki_dir: Path):
        """Integration: start thread, create file, advance mtime, verify bridge ran."""
        remember = FakeRememberFn()
        b = WikiBridge(
            hermes_home=str(hermes_home),
            remember_fn=remember,
            poll_interval=0.05,
        )
        b.start_polling()

        # Wait for initial baseline scan to complete
        time.sleep(0.12)

        page = wiki_dir / "new_page.md"
        page.write_text("Threaded content.", encoding="utf-8")
        # Force mtime to look newer than whatever was cached (or not cached yet)
        # by seeding stale entry if it was discovered in initial scan
        b._mtimes[page] = page.stat().st_mtime - 1

        # Wait for one poll cycle
        time.sleep(0.15)
        b.stop_polling()

        assert len(remember.calls) >= 1


# ---------------------------------------------------------------------------
# Missing wiki dir (robustness)
# ---------------------------------------------------------------------------


class TestMissingWikiDir:
    def test_constructor_does_not_raise_on_absent_dir(self, tmp_path: Path):
        # hermes_home exists but matrix-memory/wiki does not
        b = WikiBridge(hermes_home=str(tmp_path / "nonexistent"))
        assert b is not None

    def test_bridge_on_absent_file_returns_safely(self, tmp_path: Path):
        b = WikiBridge(
            hermes_home=str(tmp_path / "nonexistent"),
            remember_fn=FakeRememberFn(),
        )
        result = b.bridge("does_not_exist.md")
        assert result["indexed"] is False
        assert result["links"] == 0
        assert result["dead_links"] == 0

    def test_bridge_never_raises_on_missing_dir(self, tmp_path: Path):
        b = WikiBridge(hermes_home=str(tmp_path / "ghost"))
        # Must not raise even with no remember_fn
        result = b.bridge("anything.md")
        assert isinstance(result, dict)
