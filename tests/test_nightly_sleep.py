"""Nightly cross-session consolidation (one sweep per local day, gateway-owned)."""

import sqlite3
import threading
from datetime import datetime, timedelta

import pytest

import hermes_memory_provider as hmp
from hermes_memory_provider import MnemosyneMemoryProvider as Provider
from mnemosyne.core import beam as beam_mod
from mnemosyne.core import local_llm
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.llm_backends import CallableLLMBackend, set_host_llm_backend


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(hmp, "_NIGHTLY_POLL_SECONDS", 3600.0)  # tests drive ticks directly
    # Never register the real Hermes auxiliary LLM from initialize().
    from hermes_memory_provider import hermes_llm_adapter
    monkeypatch.setattr(hermes_llm_adapter, "register_hermes_host_llm", lambda: False)
    monkeypatch.setattr(hermes_llm_adapter, "unregister_hermes_host_llm", lambda: None)
    assert not Provider._SWEEP_LOCK.locked()
    yield
    hmp._stop_nightly_timers(5)
    assert Provider._SWEEP_LOCK.acquire(timeout=10), "sweep lock leaked"
    Provider._SWEEP_LOCK.release()


@pytest.fixture
def host_llm(monkeypatch):
    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
    set_host_llm_backend(CallableLLMBackend("fake-host", lambda prompt, **kw: "LLM SUMMARY"))
    yield
    set_host_llm_backend(None)


def _provider(tmp_path, **kw):
    p = Provider()
    kw.setdefault("nightly_sleep_enabled", True)
    p.initialize(session_id="s-live", hermes_home=str(tmp_path / "home"),
                 agent_identity="hermes-switch", platform="telegram", **kw)
    assert p._beam is None or str(tmp_path) in str(p._beam.db_path)
    return p


def _job(p):
    return hmp._NIGHTLY_JOBS[str(p._beam.db_path)]


def _at(monkeypatch, *args):
    monkeypatch.setattr(hmp, "_nightly_now", lambda: datetime(*args))


def _insert(db, rows):
    """rows: (id, session_id, hours_old)"""
    conn = sqlite3.connect(db)
    for rid, sid, hours in rows:
        ts = (datetime.now() - timedelta(hours=hours)).isoformat()
        conn.execute("INSERT INTO working_memory (id, content, source, timestamp, session_id) "
                     "VALUES (?, ?, 'conversation', ?, ?)", (rid, f"content of {rid}", ts, sid))
    conn.commit()
    conn.close()


def _consolidated(db):
    conn = sqlite3.connect(db)
    out = {r[0]: r[1] is not None for r in conn.execute("SELECT id, consolidated_at FROM working_memory")}
    conn.close()
    return out


def test_disabled_by_default(tmp_path):
    _provider(tmp_path, nightly_sleep_enabled=None)
    assert hmp._NIGHTLY_JOBS == {}


@pytest.mark.parametrize("skip", ["cron,flush", "flush"])  # cron fully skipped / passive (beam open)
def test_cron_context_never_owns_the_timer(tmp_path, skip):
    p = _provider(tmp_path, agent_context="cron", skip_contexts=skip)
    assert (p._beam is None) == (skip == "cron,flush")
    assert hmp._NIGHTLY_JOBS == {}


def test_two_providers_on_one_db_share_one_timer(tmp_path):
    a = _provider(tmp_path)
    b = _provider(tmp_path)
    assert a._beam.db_path == b._beam.db_path
    assert len(hmp._NIGHTLY_JOBS) == 1
    alive = [t for t in threading.enumerate() if t.name == "mnemosyne-nightly-sleep"]
    assert alive == [_job(a)["thread"]]


def test_last_primary_shutdown_stops_the_timer(tmp_path, monkeypatch):
    p = _provider(tmp_path)
    thread = _job(p)["thread"]
    monkeypatch.setattr(hmp, "_active_provider_count", 1)  # p is the last active primary
    p.shutdown()
    assert hmp._NIGHTLY_JOBS == {}
    assert not thread.is_alive()


def test_runs_once_after_the_hour_not_before_and_not_twice_across_reinit(tmp_path, monkeypatch, host_llm):
    p = _provider(tmp_path)
    db = p._beam.db_path
    _insert(db, [("a", "old-chat", 30)])

    _at(monkeypatch, 2026, 10, 7, 2, 59)
    assert hmp._nightly_tick(_job(p)) is None
    assert not _consolidated(db)["a"]

    _at(monkeypatch, 2026, 10, 7, 3, 1)
    assert hmp._nightly_tick(_job(p))["sessions_consolidated"] == 1
    assert _consolidated(db)["a"]

    # Same night, new work: neither a second tick nor a re-initialized provider runs again.
    _insert(db, [("b", "other-chat", 30)])
    _at(monkeypatch, 2026, 10, 7, 23, 0)
    assert hmp._nightly_tick(_job(p)) is None
    hmp._stop_nightly_timers(5)  # gateway restart
    p2 = _provider(tmp_path)
    assert hmp._nightly_tick(_job(p2)) is None
    assert not _consolidated(db)["b"]

    # Gateway down at 03:00 the next day: first check after the hour runs.
    _at(monkeypatch, 2026, 10, 8, 14, 0)
    assert hmp._nightly_tick(_job(p2))["sessions_consolidated"] == 1
    assert _consolidated(db)["b"]


def test_age_cutoff_is_per_call_and_global_untouched(tmp_path, monkeypatch, host_llm):
    before = beam_mod.SLEEP_AGE_HOURS
    assert before > 30  # the default cutoff alone would leave the 30h row alone
    p = _provider(tmp_path, nightly_sleep_age_hours=24)
    _insert(p._beam.db_path, [("old", "c1", 30), ("young", "c2", 10)])
    _at(monkeypatch, 2026, 10, 7, 4, 0)
    hmp._nightly_tick(_job(p))
    assert _consolidated(p._beam.db_path) == {"old": True, "young": False}
    assert beam_mod.SLEEP_AGE_HOURS == before


def test_no_session_cap_when_unset(tmp_path, monkeypatch, host_llm):
    p = _provider(tmp_path)
    assert _job(p)["max_sessions"] is None
    _insert(p._beam.db_path, [(f"r{i}", f"chat{i}", 30 + i) for i in range(15)])  # > the 10-turn sweep cap
    _at(monkeypatch, 2026, 10, 7, 3, 0)
    assert hmp._nightly_tick(_job(p))["sessions_consolidated"] == 15
    assert all(_consolidated(p._beam.db_path).values())


def test_sweep_beams_carry_profile_owner(tmp_path, monkeypatch, host_llm):
    p = _provider(tmp_path)
    _insert(p._beam.db_path, [("a", "c1", 30), ("b", "c2", 30)])
    seen = []
    real_sleep = BeamMemory.sleep
    monkeypatch.setattr(BeamMemory, "sleep", lambda self, **kw: seen.append(
        (self.canonical_owner_id, self.agent_context)) or real_sleep(self, **kw))
    _at(monkeypatch, 2026, 10, 7, 3, 0)
    hmp._nightly_tick(_job(p))
    assert seen == [("hermes-switch", "primary")] * 2


def test_skips_without_host_llm_and_keeps_the_night(tmp_path, monkeypatch):
    p = _provider(tmp_path)
    _insert(p._beam.db_path, [("a", "c1", 30)])
    _at(monkeypatch, 2026, 10, 7, 3, 0)
    assert hmp._nightly_tick(_job(p)) is None
    assert hmp._nightly_day_open(_job(p)["db_path"], "2026-10-07", claim=False)


def test_auto_sleep_false_stops_turn_trigger_but_not_session_end(tmp_path, monkeypatch):
    p = _provider(tmp_path, auto_sleep=False)
    turn_triggers, sleeps = [], []
    monkeypatch.setattr(Provider, "_maybe_auto_sleep", lambda self: turn_triggers.append(1))
    monkeypatch.setattr(BeamMemory, "sleep", lambda self, **kw: sleeps.append(1) or {})
    for i in range(20):
        p.sync_turn(f"user message number {i} with content", "")
    assert turn_triggers == []
    p.on_session_end([])
    assert sleeps == [1]

    on = _provider(tmp_path, auto_sleep=True)
    for i in range(10):
        on.sync_turn(f"user message number {i} with content", "")
    assert turn_triggers == [1]
