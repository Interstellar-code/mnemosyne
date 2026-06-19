"""Tests for Tier1Passthrough.migrate_once — §6.3 one-time migration."""

from __future__ import annotations

from pathlib import Path

import pytest

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


class FakeRememberFn:
    """Records calls to remember_fn(content=..., source=...)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, *, content: str, source: str) -> None:
        self.calls.append({"content": content, "source": source})


# ---------------------------------------------------------------------------
# Migration — marker absent (first run)
# ---------------------------------------------------------------------------


class TestMigrateOnceFirstRun:
    def test_calls_remember_fn_for_each_entry(self, t1: Tier1Passthrough, hermes_home: Path):
        (hermes_home / "memories" / "MEMORY.md").write_text(
            "§ Fact A\nContent A.\n§ Fact B\nContent B.", encoding="utf-8"
        )
        fn = FakeRememberFn()
        result = t1.migrate_once(fn)
        assert result["status"] == "migrated"
        assert len(fn.calls) == 2

    def test_remember_fn_called_with_source_semantic(
        self, t1: Tier1Passthrough, hermes_home: Path
    ):
        (hermes_home / "memories" / "MEMORY.md").write_text(
            "§ Entry\nSome content.", encoding="utf-8"
        )
        fn = FakeRememberFn()
        t1.migrate_once(fn)
        assert all(c["source"] == "semantic" for c in fn.calls)

    def test_remember_fn_receives_entry_content(
        self, t1: Tier1Passthrough, hermes_home: Path
    ):
        (hermes_home / "memories" / "MEMORY.md").write_text(
            "§ Fact\nThe earth orbits the sun.", encoding="utf-8"
        )
        fn = FakeRememberFn()
        t1.migrate_once(fn)
        contents = [c["content"] for c in fn.calls]
        assert any("earth orbits" in c for c in contents)

    def test_marker_written_after_migration(self, t1: Tier1Passthrough, hermes_home: Path):
        (hermes_home / "memories" / "MEMORY.md").write_text(
            "§ X\nData.", encoding="utf-8"
        )
        fn = FakeRememberFn()
        result = t1.migrate_once(fn)
        marker = Path(result["marker"])
        assert marker.exists()

    def test_marker_path_returned_in_result(self, t1: Tier1Passthrough, hermes_home: Path):
        (hermes_home / "memories" / "MEMORY.md").write_text("Some data.", encoding="utf-8")
        fn = FakeRememberFn()
        result = t1.migrate_once(fn)
        assert ".migration-v0.1.done" in result["marker"]

    def test_memory_md_not_modified_after_migration(
        self, t1: Tier1Passthrough, hermes_home: Path
    ):
        original = "§ Fact\nOriginal content stays."
        p = hermes_home / "memories" / "MEMORY.md"
        p.write_text(original, encoding="utf-8")
        fn = FakeRememberFn()
        t1.migrate_once(fn)
        assert p.read_text(encoding="utf-8") == original

    def test_migrates_user_md_entries_too(self, t1: Tier1Passthrough, hermes_home: Path):
        (hermes_home / "memories" / "MEMORY.md").write_text(
            "§ M\nMemory item.", encoding="utf-8"
        )
        (hermes_home / "memories" / "USER.md").write_text(
            "§ U\nUser item.", encoding="utf-8"
        )
        fn = FakeRememberFn()
        result = t1.migrate_once(fn)
        assert result["count"] == 2
        assert len(fn.calls) == 2

    def test_count_reflects_number_of_entries(self, t1: Tier1Passthrough, hermes_home: Path):
        (hermes_home / "memories" / "MEMORY.md").write_text(
            "§ A\nA.\n§ B\nB.\n§ C\nC.", encoding="utf-8"
        )
        fn = FakeRememberFn()
        result = t1.migrate_once(fn)
        assert result["count"] == 3

    def test_missing_files_migrate_gracefully(self, t1: Tier1Passthrough, hermes_home: Path):
        # Neither MEMORY.md nor USER.md present
        fn = FakeRememberFn()
        result = t1.migrate_once(fn)
        assert result["status"] == "migrated"
        assert result["count"] == 0
        assert len(fn.calls) == 0


# ---------------------------------------------------------------------------
# Migration — marker present (idempotency / skip)
# ---------------------------------------------------------------------------


class TestMigrateOnceSkipped:
    def _write_marker(self, hermes_home: Path) -> Path:
        marker = hermes_home / "matrix-memory" / ".migration-v0.1.done"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("done\n", encoding="utf-8")
        return marker

    def test_skipped_when_marker_present(self, t1: Tier1Passthrough, hermes_home: Path):
        self._write_marker(hermes_home)
        (hermes_home / "memories" / "MEMORY.md").write_text("§ X\nData.", encoding="utf-8")
        fn = FakeRememberFn()
        result = t1.migrate_once(fn)
        assert result["status"] == "skipped"

    def test_remember_fn_not_called_when_skipped(
        self, t1: Tier1Passthrough, hermes_home: Path
    ):
        self._write_marker(hermes_home)
        (hermes_home / "memories" / "MEMORY.md").write_text("§ X\nData.", encoding="utf-8")
        fn = FakeRememberFn()
        t1.migrate_once(fn)
        assert fn.calls == []

    def test_skip_count_is_zero(self, t1: Tier1Passthrough, hermes_home: Path):
        self._write_marker(hermes_home)
        fn = FakeRememberFn()
        result = t1.migrate_once(fn)
        assert result["count"] == 0

    def test_idempotent_second_call(self, t1: Tier1Passthrough, hermes_home: Path):
        (hermes_home / "memories" / "MEMORY.md").write_text("§ Y\nContent.", encoding="utf-8")
        fn = FakeRememberFn()
        r1 = t1.migrate_once(fn)
        assert r1["status"] == "migrated"
        fn2 = FakeRememberFn()
        r2 = t1.migrate_once(fn2)
        assert r2["status"] == "skipped"
        assert fn2.calls == []
