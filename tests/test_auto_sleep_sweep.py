"""Cross-session auto-sleep sweep (hermes-agent issue #252).

Auto-sleep eligibility was counted DB-wide but sleep() only consolidated the
current session, so inactive sessions were never consolidated. The provider
now runs a bounded, non-overlapping sleep_all_sessions() sweep through the
Hermes host LLM only, and never AAAK-encodes in that sweep.
"""

import logging
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne.core import local_llm
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.llm_backends import CallableLLMBackend, set_host_llm_backend

Provider = MnemosyneMemoryProvider


@pytest.fixture(autouse=True)
def _free_sweep_lock():
    """No test may start or leave behind a held process-wide sweep lock."""
    assert not Provider._SWEEP_LOCK.locked()
    yield
    assert Provider._SWEEP_LOCK.acquire(timeout=10), "sweep lock leaked"
    Provider._SWEEP_LOCK.release()
    Provider._SWEEP_HELD_SINCE = None


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def fake_host_llm(monkeypatch):
    """Register a fake Hermes host LLM the way the gateway adapter does."""
    calls = []

    def _complete(prompt, **kwargs):
        calls.append(prompt)
        return "LLM SUMMARY of old work"

    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
    set_host_llm_backend(CallableLLMBackend("fake-host", _complete))
    try:
        yield calls
    finally:
        set_host_llm_backend(None)


def _insert(db, rows):
    """rows: (id, session_id, hours_old[, pinned])"""
    conn = sqlite3.connect(db)
    for row in rows:
        rid, sid, hours = row[:3]
        pinned = row[3] if len(row) > 3 else 0
        ts = (datetime.now() - timedelta(hours=hours)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, pinned) "
            "VALUES (?, ?, 'conversation', ?, ?, ?)",
            (rid, f"content of {rid}", ts, sid, pinned),
        )
    conn.commit()
    conn.close()


def _consolidated(db):
    conn = sqlite3.connect(db)
    out = {r[0]: r[1] is not None for r in conn.execute("SELECT id, consolidated_at FROM working_memory")}
    conn.close()
    return out


def _count(db, table):
    conn = sqlite3.connect(db)
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return n


def _wait_for_sweep(provider):
    if provider._sweep_thread is not None:
        provider._sweep_thread.join(timeout=30)
        assert not provider._sweep_thread.is_alive(), "sweep never finished"
    assert Provider._SWEEP_LOCK.acquire(timeout=30), "sweep lock never released"
    Provider._SWEEP_LOCK.release()


# --- BeamMemory.sleep / sleep_all_sessions -----------------------------------

def test_sweep_caps_sessions_oldest_first_and_drains_over_runs(temp_db):
    beam = BeamMemory(session_id="current", db_path=temp_db)
    _insert(temp_db, [
        ("pinned-only", "s0", 500, 1),  # oldest, but sleep() can never take it
        ("a", "s1", 400),
        ("b", "s2", 300),
        ("c", "s3", 200),
    ])

    first = beam.sleep_all_sessions(max_sessions=2)
    assert [r["session_id"] for r in first["session_results"]] == ["s1", "s2"]
    state = _consolidated(temp_db)
    assert state["a"] and state["b"] and not state["c"]
    assert not state["pinned-only"]

    second = beam.sleep_all_sessions(max_sessions=2)
    assert [r["session_id"] for r in second["session_results"]] == ["s3"]
    assert _consolidated(temp_db)["c"]


def test_sweep_time_budget_counts_attempted_sessions_including_errors(temp_db, monkeypatch):
    beam = BeamMemory(session_id="current", db_path=temp_db)
    _insert(temp_db, [("a", "s1", 400), ("b", "s2", 300)])
    calls = []

    def _boom(self, **kwargs):
        calls.append(self.session_id)
        raise RuntimeError("boom")

    monkeypatch.setattr(BeamMemory, "sleep", _boom)

    result = beam.sleep_all_sessions(time_budget_seconds=0)

    assert calls == ["s1"], "an errored session still uses up the budget"
    assert result["errors"] == 1
    assert result["stopped_reason"] == "time_budget"


def test_sweep_without_host_llm_skips_instead_of_aaak(temp_db, monkeypatch):
    # A remote URL / local model being "available" must not be used either.
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)
    monkeypatch.setattr(local_llm, "_host_backend_will_handle_call", lambda: False)
    beam = BeamMemory(session_id="current", db_path=temp_db)
    _insert(temp_db, [("a", "s1", 400)])

    result = beam.sleep_all_sessions(require_host_llm=True)

    assert result["stopped_reason"] == "no_host_llm"
    assert result["items_consolidated"] == 0
    assert not _consolidated(temp_db)["a"]
    assert _count(temp_db, "episodic_memory") == 0


def test_sleep_allow_aaak_false_unclaims_group_without_llm_summary(temp_db, monkeypatch):
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)
    monkeypatch.setattr(local_llm, "_summarize_memories", lambda *a, **k: None)
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    _insert(temp_db, [("a", "s1", 400), ("b", "s1", 300)])

    result = beam.sleep(allow_aaak=False)

    assert result["status"] == "no_op"
    conn = sqlite3.connect(temp_db)
    rows = conn.execute("SELECT consolidated_at, consolidation_claimed_at FROM working_memory").fetchall()
    conn.close()
    # un-claimed, with claimed_at kept as the retry-backoff marker
    assert [r[0] for r in rows] == [None, None]
    assert all(r[1] is not None for r in rows)
    assert _count(temp_db, "episodic_memory") == 0
    assert _count(temp_db, "consolidation_log") == 0
    # reclaim_orphans must not touch the marker rows (consolidated_at is NULL)
    assert beam.reclaim_orphans(stale_after_seconds=0)["candidates"] == 0


def test_failed_session_backs_off_then_is_retried(temp_db, monkeypatch):
    """A session whose LLM always fails must not hog the head of the queue."""
    monkeypatch.setattr(local_llm, "_host_backend_will_handle_call", lambda: True)
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)
    # sleep() calls the private _summarize_memories since upstream 4.0.
    monkeypatch.setattr(local_llm, "_summarize_memories",
                        lambda lines, **k: None if "bad" in lines[0] else "LLM SUMMARY")
    beam = BeamMemory(session_id="current", db_path=temp_db)
    _insert(temp_db, [("bad", "s1", 500), ("good", "s2", 400)])
    cutoff = (datetime.now() - timedelta(hours=1)).isoformat()

    first = beam.sleep_all_sessions(max_sessions=1, require_host_llm=True)
    assert [r["session_id"] for r in first["session_results"]] == ["s1"]
    assert not _consolidated(temp_db)["bad"]
    assert beam._count_unconsolidated_before(cutoff, respect_backoff=True) == 1  # only "good"

    second = beam.sleep_all_sessions(max_sessions=1, require_host_llm=True)
    assert [r["session_id"] for r in second["session_results"]] == ["s2"]
    assert _consolidated(temp_db)["good"]
    assert beam._count_unconsolidated_before(cutoff, respect_backoff=True) == 0
    # session-local / tool path ignores the backoff (unchanged behaviour)
    assert beam._count_unconsolidated_before(cutoff) == 1

    # backoff expires -> picked again, and sleep()'s claim overwrites the marker
    expired = (datetime.now() - timedelta(hours=7)).isoformat()
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE working_memory SET consolidation_claimed_at = ? WHERE id = 'bad'", (expired,))
    conn.commit()
    conn.close()
    assert beam._count_unconsolidated_before(cutoff, respect_backoff=True) == 1
    monkeypatch.setattr(local_llm, "_summarize_memories", lambda lines, **k: "LLM SUMMARY")
    third = beam.sleep_all_sessions(max_sessions=1, require_host_llm=True)
    assert [r["session_id"] for r in third["session_results"]] == ["s1"]
    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT consolidated_at, consolidation_claimed_at FROM working_memory WHERE id='bad'").fetchone()
    conn.close()
    assert row[0] is not None and row[1] is None


def test_host_sweep_skips_maintenance_passes(temp_db, fake_host_llm, monkeypatch):
    beam = BeamMemory(session_id="current", db_path=temp_db)
    _insert(temp_db, [("a", "s1", 400)])
    called = []
    monkeypatch.setattr(BeamMemory, "degrade_episodic", lambda self, **k: called.append("degrade"))
    monkeypatch.setattr(BeamMemory, "_deduplicate_memoria_cross_session",
                        lambda self: called.append("dedup"))

    result = beam.sleep_all_sessions(require_host_llm=True)

    assert result["items_consolidated"] == 1
    assert called == []


def test_sweep_select_and_census_run_under_session_lock(temp_db, fake_host_llm, monkeypatch):
    """The session select and the fleet census touch the shared DB too (#498)."""
    import mnemosyne.core.beam as beam_mod

    lock = threading.RLock()
    seen = {"select": [], "census": []}
    real_cutoff = beam_mod._retry_backoff_cutoff
    monkeypatch.setattr(beam_mod, "_retry_backoff_cutoff",
                        lambda: seen["select"].append(lock._is_owned()) or real_cutoff())
    real_census = BeamMemory._attach_fleet_conflict_census
    monkeypatch.setattr(BeamMemory, "_attach_fleet_conflict_census",
                        lambda self, result, enabled=True: seen["census"].append(
                            (enabled, lock._is_owned())) or real_census(self, result, enabled))
    beam = BeamMemory(session_id="current", db_path=temp_db)

    beam.sleep_all_sessions(require_host_llm=True, session_lock=lock)  # nothing to do
    _insert(temp_db, [("a", "s1", 400)])
    beam.sleep_all_sessions(require_host_llm=True, session_lock=lock)

    # both selects, plus sleep()'s own backoff lookup inside the per-session lock
    assert len(seen["select"]) >= 2 and all(seen["select"])
    # the sweep's own census calls (enabled=True), not the per-session ones
    assert [owned for enabled, owned in seen["census"] if enabled] == [True, True]


# --- provider._maybe_auto_sleep ---------------------------------------------

def test_auto_sleep_sweep_holds_beam_lock_per_session_only(temp_db, fake_host_llm, monkeypatch):
    """#498 lock is held during each session's sleep but free between sessions,
    so prefetch/tool calls are not blocked for the whole sweep."""
    provider = Provider()
    provider._beam = BeamMemory(session_id="current", db_path=temp_db)
    _insert(temp_db, [("a", "s1", 400), ("b", "s2", 300)])
    provider._auto_sleep_threshold = 0
    lock = provider._ensure_beam_access_lock()

    def _free_for_other_thread():
        got = []
        t = threading.Thread(target=lambda: got.append(lock.acquire(blocking=False)) or (got[0] and lock.release()))
        t.start()
        t.join(5)
        return got[0]

    during, between = [], []
    real_sleep = BeamMemory.sleep
    monkeypatch.setattr(BeamMemory, "sleep",
                        lambda self, **kw: during.append(_free_for_other_thread()) or real_sleep(self, **kw))
    real_check = local_llm._host_backend_will_handle_call

    def _check():
        # the sweep loop's per-session host-LLM check runs between sessions
        if sys._getframe(1).f_code.co_name == "sleep_all_sessions":
            between.append(_free_for_other_thread())
        return real_check()

    monkeypatch.setattr(local_llm, "_host_backend_will_handle_call", _check)

    provider._maybe_auto_sleep()
    _wait_for_sweep(provider)

    assert during == [False, False], "lock must be held during each session's sleep"
    assert between == [True, True], "lock must be free between sessions"
    assert all(_consolidated(temp_db).values())


def test_auto_sleep_consolidates_other_sessions_with_host_llm(temp_db, fake_host_llm):
    """Repro for #252: old rows in OTHER sessions stayed unconsolidated forever."""
    current = BeamMemory(session_id="current", db_path=temp_db)
    _insert(temp_db, [
        ("other1-a", "other1", 300),
        ("other1-b", "other1", 299),
        ("other2-a", "other2", 200),
    ])
    provider = Provider()
    provider._beam = current
    provider._auto_sleep_threshold = 0

    provider._maybe_auto_sleep()
    _wait_for_sweep(provider)

    assert all(_consolidated(temp_db).values()), _consolidated(temp_db)
    conn = sqlite3.connect(temp_db)
    logs = conn.execute("SELECT session_id, summary_preview FROM consolidation_log ORDER BY session_id").fetchall()
    conn.close()
    assert [s for s, _ in logs] == ["other1", "other2"]
    assert all("(llm)" in preview for _, preview in logs)
    assert fake_host_llm, "host LLM was never called"


def _mock_provider(monkeypatch, host, sweep_beam=None):
    monkeypatch.setattr(local_llm, "_host_backend_will_handle_call", lambda: host)
    beam = MagicMock(session_id="current", db_path="/tmp/x.db", author_id=None,
                     author_type=None, channel_id=None)
    beam.get_working_stats.return_value = {"total": 99}
    beam._count_unconsolidated_before.return_value = 3
    sweep_beam = sweep_beam or MagicMock()
    monkeypatch.setattr("hermes_memory_provider._get_beam_class", lambda: lambda **kw: sweep_beam)
    provider = Provider()
    provider._beam = beam
    provider._auto_sleep_threshold = 1
    provider._sweep_max_sessions = 4
    provider._sweep_time_budget = 7.0
    return provider, beam, sweep_beam


@pytest.mark.parametrize("host,max_sessions", [(False, 4), (True, 0)])
def test_auto_sleep_without_host_llm_or_disabled_stays_session_local(monkeypatch, host, max_sessions):
    provider, beam, sweep_beam = _mock_provider(monkeypatch, host)
    provider._sweep_max_sessions = max_sessions

    provider._maybe_auto_sleep()
    provider._sweep_thread.join(5)

    # eligibility counted for the current session only, matching sleep()
    assert beam._count_unconsolidated_before.call_args.kwargs["session_id"] == "current"
    sweep_beam.sleep.assert_called_once()
    sweep_beam.sleep_all_sessions.assert_not_called()


def test_auto_sleep_sweep_is_bounded_unjoined_and_not_overlapping(monkeypatch):
    release = threading.Event()
    sweep_beam = MagicMock()
    sweep_beam.sleep_all_sessions.side_effect = lambda **kw: release.wait(10) and {}
    provider, _, _ = _mock_provider(monkeypatch, True, sweep_beam)
    other, _, _ = _mock_provider(monkeypatch, True, sweep_beam)

    start = time.monotonic()
    provider._maybe_auto_sleep()
    assert time.monotonic() - start < 2, "sweep must not be joined on the sync path"

    other._maybe_auto_sleep()  # second provider instance, sweep still in flight
    assert other._sweep_thread is None
    assert other._reflect_calls_this_session == 0, "skipped overlap must not burn budget"

    release.set()
    _wait_for_sweep(provider)
    assert sweep_beam.sleep_all_sessions.call_count == 1
    assert sweep_beam.sleep_all_sessions.call_args.kwargs == {
        "max_sessions": 4, "time_budget_seconds": 7.0, "require_host_llm": True,
        "session_lock": provider._ensure_beam_access_lock()}
    sweep_beam.reclaim_orphans.assert_called_once_with(stale_after_seconds=6 * 3600)

    other._maybe_auto_sleep()  # lock released -> next sweep may run
    _wait_for_sweep(other)
    assert sweep_beam.sleep_all_sessions.call_count == 2


def test_auto_sleep_thread_start_failure_frees_lock(monkeypatch):
    provider, _, sweep_beam = _mock_provider(monkeypatch, True)

    def _fail(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _fail)
    provider._maybe_auto_sleep()  # errors are swallowed by _maybe_auto_sleep

    assert not Provider._SWEEP_LOCK.locked()
    assert Provider._SWEEP_HELD_SINCE is None
    sweep_beam.sleep_all_sessions.assert_not_called()


def test_auto_sleep_warns_when_sweep_lock_held_too_long(monkeypatch, caplog):
    provider, _, sweep_beam = _mock_provider(monkeypatch, True)
    assert Provider._SWEEP_LOCK.acquire(blocking=False)
    try:
        Provider._SWEEP_HELD_SINCE = time.monotonic() - 3 * provider._sweep_time_budget - 1
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider._maybe_auto_sleep()
    finally:
        Provider._SWEEP_LOCK.release()
    assert "previous sweep may be hung" in caplog.text
    sweep_beam.sleep_all_sessions.assert_not_called()


def test_sweep_yields_lock_so_waiters_acquire_between_sessions(temp_db, monkeypatch):
    """CPython locks are unfair: releasing and immediately re-taking the Beam lock
    starves a waiting thread for the whole sweep. Each waiter that arrives
    mid-sweep must get the lock within one session, not after the sweep.
    Three sequential waiters: an unfair re-acquire can win the race once by luck,
    not three times."""
    def _slow(prompt, **kwargs):
        time.sleep(0.3)
        return "LLM SUMMARY"

    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
    set_host_llm_backend(CallableLLMBackend("slow-host", _slow))
    try:
        beam = BeamMemory(session_id="current", db_path=temp_db)
        _insert(temp_db, [(f"r{i}", f"s{i}", 400 + i) for i in range(6)])
        lock = threading.RLock()
        started = threading.Event()
        done = []
        real_sleep = BeamMemory.sleep

        def _counting_sleep(self, **kw):
            started.set()
            out = real_sleep(self, **kw)
            done.append(1)
            return out

        monkeypatch.setattr(BeamMemory, "sleep", _counting_sleep)
        sweep = threading.Thread(target=lambda: beam.sleep_all_sessions(
            max_sessions=6, require_host_llm=True, session_lock=lock))
        sweep.start()
        assert started.wait(10)
        for _ in range(3):
            at_acquire = []

            def _waiter():
                n0 = len(done)
                with lock:
                    at_acquire.append(len(done) - n0)

            waiter = threading.Thread(target=_waiter)
            waiter.start()
            waiter.join(60)
            assert at_acquire, "waiter never acquired the lock"
            assert at_acquire[0] <= 1, "waiter starved across multiple sessions"
        sweep.join(60)
    finally:
        set_host_llm_backend(None)


def test_sweep_yields_lock_before_census_after_last_session(temp_db, monkeypatch):
    """A waiter blocked during the LAST session must get the lock before the
    fleet census re-acquires it."""
    def _slow(prompt, **kwargs):
        time.sleep(0.3)
        return "LLM SUMMARY"

    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
    set_host_llm_backend(CallableLLMBackend("slow-host", _slow))
    try:
        beam = BeamMemory(session_id="current", db_path=temp_db)
        _insert(temp_db, [(f"r{i}", f"s{i}", 400 + i) for i in range(2)])
        lock = threading.RLock()
        events, started = [], threading.Event()
        real_sleep = BeamMemory.sleep
        real_census = BeamMemory._attach_fleet_conflict_census

        def _sleep(self, **kw):
            started.set()
            return real_sleep(self, **kw)

        def _census(self, result):
            events.append("census")
            return real_census(self, result)

        monkeypatch.setattr(BeamMemory, "sleep", _sleep)
        monkeypatch.setattr(BeamMemory, "_attach_fleet_conflict_census", _census)
        sweep = threading.Thread(target=lambda: beam.sleep_all_sessions(
            max_sessions=2, require_host_llm=True, session_lock=lock))
        sweep.start()
        assert started.wait(10)
        # Let the first session finish so the waiter arrives during the last one.
        time.sleep(0.5)

        def _waiter():
            with lock:
                events.append("waiter")

        waiter = threading.Thread(target=_waiter)
        waiter.start()
        waiter.join(60)
        sweep.join(60)
        assert events[:2] == ["waiter", "census"], events
    finally:
        set_host_llm_backend(None)
