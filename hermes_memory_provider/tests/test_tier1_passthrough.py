"""Tests for Tier1Passthrough — §6.1/6.2 passthrough recall + forget."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from unittest.mock import patch

from hermes_memory_provider.tier1 import Tier1Passthrough


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hermes_home(tmp_path: Path) -> Path:
    memories = tmp_path / "memories"
    memories.mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def t1(hermes_home: Path) -> Tier1Passthrough:
    return Tier1Passthrough(hermes_home=str(hermes_home))


def _write_memory(hermes_home: Path, content: str) -> Path:
    p = hermes_home / "memories" / "MEMORY.md"
    p.write_text(content, encoding="utf-8")
    return p


def _write_user(hermes_home: Path, content: str) -> Path:
    p = hermes_home / "memories" / "USER.md"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# split_entries — § delimiter
# ---------------------------------------------------------------------------


class TestSplitEntriesSection:
    def test_splits_on_section_lines(self):
        text = textwrap.dedent("""\
            § Heading One
            Content of entry one.
            § Heading Two
            Content of entry two.
        """)
        entries = Tier1Passthrough.split_entries(text)
        assert len(entries) == 2
        assert "Heading One" in entries[0]
        assert "Heading Two" in entries[1]

    def test_preamble_before_first_section_included(self):
        text = "Intro line\n§ Section A\nBody A\n"
        entries = Tier1Passthrough.split_entries(text)
        assert any("Intro" in e for e in entries)
        assert any("Section A" in e for e in entries)

    def test_single_section(self):
        text = "§ Only section\nSome content here."
        entries = Tier1Passthrough.split_entries(text)
        assert len(entries) == 1
        assert "Only section" in entries[0]


# ---------------------------------------------------------------------------
# split_entries — \n\n fallback
# ---------------------------------------------------------------------------


class TestSplitEntriesDoubleNewline:
    def test_fallback_to_double_newline(self, caplog):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        import logging
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider.tier1"):
            entries = Tier1Passthrough.split_entries(text)
        assert len(entries) == 3
        assert entries[0] == "First paragraph."
        assert "warning" in caplog.text.lower() or caplog.records  # warning logged

    def test_fallback_strips_blank_chunks(self):
        text = "Alpha\n\n\n\nBeta"
        entries = Tier1Passthrough.split_entries(text)
        assert entries == ["Alpha", "Beta"]


# ---------------------------------------------------------------------------
# split_entries — whole-file last resort
# ---------------------------------------------------------------------------


class TestSplitEntriesWholeFile:
    def test_whole_file_when_no_delimiters(self):
        text = "Just a single line of text with no delimiters at all."
        entries = Tier1Passthrough.split_entries(text)
        assert entries == [text]

    def test_empty_string_returns_empty(self):
        assert Tier1Passthrough.split_entries("") == []

    def test_whitespace_only_returns_empty(self):
        assert Tier1Passthrough.split_entries("   \n  \n") == []


# ---------------------------------------------------------------------------
# recall — substring matching
# ---------------------------------------------------------------------------


class TestRecall:
    def test_recall_finds_matching_entry(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ Fact\nThe sky is blue.\n§ Other\nCats are mammals.")
        results = t1.recall("sky")
        assert len(results) == 1
        assert "sky" in results[0]["entry"].lower()

    def test_recall_case_insensitive(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ A\nPython is great.\n§ B\nJava is verbose.")
        results = t1.recall("PYTHON")
        assert len(results) == 1
        assert "Python" in results[0]["entry"]

    def test_recall_annotates_tier1(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ X\nSome fact.")
        results = t1.recall("fact")
        assert results[0]["tier"] == 1

    def test_recall_annotates_source_memory(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ X\nMemory entry.")
        results = t1.recall("Memory entry")
        assert results[0]["source"] == "memory"

    def test_recall_annotates_source_user(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_user(hermes_home, "§ Pref\nUser preference here.")
        results = t1.recall("User preference")
        assert results[0]["source"] == "user"

    def test_recall_includes_path(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ Z\nZebra facts.")
        results = t1.recall("Zebra")
        assert "MEMORY.md" in results[0]["path"]

    def test_recall_from_both_files(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ A\nApple info.")
        _write_user(hermes_home, "§ B\nApple preference.")
        results = t1.recall("Apple")
        assert len(results) == 2
        sources = {r["source"] for r in results}
        assert sources == {"memory", "user"}

    def test_recall_no_match_returns_empty(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ A\nSomething unrelated.")
        results = t1.recall("xyzzy_nonexistent")
        assert results == []

    def test_recall_respects_limit(self, t1: Tier1Passthrough, hermes_home: Path):
        # 5 matching entries
        content = "\n".join(f"§ E{i}\nItem {i} match" for i in range(5))
        _write_memory(hermes_home, content)
        results = t1.recall("match", limit=3)
        assert len(results) == 3

    def test_recall_missing_memory_file_safe(self, t1: Tier1Passthrough, hermes_home: Path):
        # MEMORY.md absent, USER.md absent — must not raise
        results = t1.recall("anything")
        assert results == []

    def test_recall_missing_one_file_still_searches_other(
        self, t1: Tier1Passthrough, hermes_home: Path
    ):
        # only USER.md present
        _write_user(hermes_home, "§ Q\nQuiet knowledge.")
        results = t1.recall("Quiet")
        assert len(results) == 1
        assert results[0]["source"] == "user"


class TestHomeResolution:
    def test_default_home_uses_hermes_core_profile_path(self, tmp_path: Path):
        profile_home = tmp_path / ".hermes" / "profiles" / "switch"
        with patch("hermes_constants.get_hermes_home", return_value=profile_home):
            t1 = Tier1Passthrough()
        assert t1.memory_path == profile_home / "memories" / "MEMORY.md"
        assert t1.user_path == profile_home / "memories" / "USER.md"


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    def test_forget_removes_matching_entry(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(
            hermes_home,
            "§ Keep this\nI should stay.\n§ Remove this\nI should go.",
        )
        result = t1.forget("should go", kind="memory")
        assert result["status"] == "deleted"
        remaining = t1.recall("should stay")
        assert len(remaining) == 1
        gone = t1.recall("should go")
        assert len(gone) == 0

    def test_forget_leaves_other_entries_intact(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(
            hermes_home,
            "§ A\nAlpha entry.\n§ B\nBeta entry.\n§ C\nGamma entry.",
        )
        t1.forget("Beta", kind="memory")
        remaining_a = t1.recall("Alpha")
        remaining_c = t1.recall("Gamma")
        assert len(remaining_a) == 1
        assert len(remaining_c) == 1

    def test_forget_appends_audit_line(self, t1: Tier1Passthrough, hermes_home: Path):
        p = _write_memory(hermes_home, "§ Del\nDelete me.")
        t1.forget("Delete me", kind="memory")
        text = p.read_text(encoding="utf-8")
        assert "audit" in text

    def test_forget_returns_removed_text(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ X\nSpecific content to remove.")
        result = t1.forget("Specific content", kind="memory")
        assert result["status"] == "deleted"
        assert "Specific content" in result["removed"]

    def test_forget_not_found(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ A\nSome memory.")
        result = t1.forget("nonexistent_xyz", kind="memory")
        assert result["status"] == "not_found"
        assert result["removed"] is None

    def test_forget_user_kind(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_user(hermes_home, "§ U\nUser pref to remove.")
        result = t1.forget("User pref", kind="user")
        assert result["status"] == "deleted"
        assert result["kind"] == "user"

    def test_forget_missing_file_not_found(self, t1: Tier1Passthrough, hermes_home: Path):
        # File doesn't exist at all
        result = t1.forget("anything", kind="memory")
        assert result["status"] == "not_found"

    def test_forget_case_insensitive_match(self, t1: Tier1Passthrough, hermes_home: Path):
        _write_memory(hermes_home, "§ Case\nMixedCase Entry Here.")
        result = t1.forget("mixedcase", kind="memory")
        assert result["status"] == "deleted"
