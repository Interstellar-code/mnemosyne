"""Config-driven memory hygiene: write-time skips, sleep-time skips, retention."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import hermes_memory_provider as hmp
from hermes_memory_provider import MnemosyneMemoryProvider as Provider, _strip_reply_quotes
from mnemosyne.core import local_llm
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.llm_backends import CallableLLMBackend, set_host_llm_backend


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(hmp, "_NIGHTLY_POLL_SECONDS", 3600.0)
    from hermes_memory_provider import hermes_llm_adapter
    monkeypatch.setattr(hermes_llm_adapter, "register_hermes_host_llm", lambda: False)
    monkeypatch.setattr(hermes_llm_adapter, "unregister_hermes_host_llm", lambda: None)
    yield
    hmp._stop_nightly_timers(5)


@pytest.fixture
def llm_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
    set_host_llm_backend(CallableLLMBackend("fake-host", lambda prompt, **kw: calls.append(prompt) or "LLM SUMMARY"))
    yield calls
    set_host_llm_backend(None)


def _provider(tmp_path, gateway_session_key="", **kw):
    p = Provider()
    p.initialize(session_id="s-live", hermes_home=str(tmp_path / "home"), agent_identity="hermes-switch",
                 platform="telegram", gateway_session_key=gateway_session_key, **kw)
    assert str(tmp_path) in str(p._beam.db_path)
    return p


def _stored(p):
    conn = sqlite3.connect(p._beam.db_path)
    rows = [r[0] for r in conn.execute("SELECT content FROM working_memory WHERE source = 'conversation'")]
    conn.close()
    return rows


TG_QUOTE = ('[Replying to: "Done. Two options:\n- A: keep x["a"] as is\n- B: "rewrite" it ]\n\nPick one."]'
            "\n\nlet's go with option B but keep the tests")
SWITCHUI_QUOTE = ("> [Quote: #150]\n> Recommended design — agent REVIEW node\n>\n> Runs only on failure\n\n"
                  "> [Re: #151] Dispatching both now (B first; A queued…\n\nship B first, then A tomorrow")


# --- reply-quote stripping --------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (TG_QUOTE, "let's go with option B but keep the tests"),
    ('[Replying to your previous message: "hi"]\n\nthanks', "thanks"),
    (SWITCHUI_QUOTE, "ship B first, then A tomorrow"),
    ('[Replying to: "only a quote"]\n\n', ""),
    ("plain [Replying to: \"x\"]\n\nnot a prefix", "plain [Replying to: \"x\"]\n\nnot a prefix"),
    ("  untouched  ", "  untouched  "),
    # Truncated quote (no closing `"]` + blank line): left as is, never eats user words.
    ('[Replying to: "cut off mid quo', '[Replying to: "cut off mid quo'),
])
def test_strip_reply_quotes(raw, expected):
    assert _strip_reply_quotes(raw) == expected


# --- write-time hygiene -----------------------------------------------------

def test_defaults_store_everything_unchanged(tmp_path):
    p = _provider(tmp_path, gateway_session_key="agent:main:a2a_fleet:dm:x")
    p.sync_turn(TG_QUOTE, "")
    p.sync_turn("ok thx", "")
    assert _stored(p) == [f"[USER] {TG_QUOTE}", "[USER] ok thx"]
    assert p._sync_turn_diagnostics()["hygiene_skipped"] == 0


def test_skip_session_patterns(tmp_path):
    cfg = {"skip_session_patterns": ["a2a_fleet", "^hermes_cron_"]}
    skipped = _provider(tmp_path, gateway_session_key="agent:main:a2a_fleet:dm:x", **cfg)
    skipped.sync_turn("a fleet peer asking for a status report", "assistant reply text here")
    assert _stored(skipped) == []
    assert skipped._sync_turn_diagnostics()["hygiene_skipped"] == 1
    kept = _provider(tmp_path, gateway_session_key="agent:main:telegram:dm:1", **cfg)
    kept.sync_turn("a real user message worth keeping", "")
    assert _stored(kept) == ["[USER] a real user message worth keeping"]


def test_skip_session_patterns_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SKIP_SESSION_PATTERNS", "a2a_fleet,cron")
    p = _provider(tmp_path, gateway_session_key="agent:main:a2a_fleet:dm:x")
    p.sync_turn("a fleet peer asking for a status report", "")
    assert _stored(p) == []


def test_min_turn_chars(tmp_path):
    p = _provider(tmp_path, min_turn_chars=40)
    p.sync_turn("   ok go ahead with that please        ", "")
    p.sync_turn("please also migrate the nightly timer to the new config keys", "")
    assert _stored(p) == ["[USER] please also migrate the nightly timer to the new config keys"]
    assert p._sync_turn_diagnostics()["hygiene_skipped"] == 1


def test_skip_turn_patterns(tmp_path):
    p = _provider(tmp_path, skip_turn_patterns=[r"^canary[-_ ]?\w+$", "work kanban task t_"])
    p.sync_turn("canary-7f3a9", "")
    p.sync_turn("work kanban task t_3bf00a12", "")
    p.sync_turn("the canary in the coal mine is a good metaphor here", "")
    assert _stored(p) == ["[USER] the canary in the coal mine is a good metaphor here"]


def test_strip_reply_quotes_stores_only_user_words(tmp_path):
    p = _provider(tmp_path, strip_reply_quotes=True, min_turn_chars=10)
    p.sync_turn(TG_QUOTE, "")
    p.sync_turn(SWITCHUI_QUOTE, "")
    p.sync_turn('[Replying to: "a long assistant answer that should not come back"]\n\nok', "")
    assert _stored(p) == ["[USER] let's go with option B but keep the tests",
                          "[USER] ship B first, then A tomorrow"]


# --- sleep-time hygiene -----------------------------------------------------

def _insert(db, rows, *, hours=30):
    """rows: (id, session_id, content)"""
    conn = sqlite3.connect(db)
    ts = (datetime.now() - timedelta(hours=hours)).isoformat()
    for rid, sid, content in rows:
        conn.execute("INSERT INTO working_memory (id, content, source, timestamp, session_id) "
                     "VALUES (?, ?, 'conversation', ?, ?)", (rid, content, ts, sid))
    conn.commit()
    conn.close()


def _state(db):
    conn = sqlite3.connect(db)
    wm = {r[0]: (r[1] is not None, r[2]) for r in conn.execute(
        "SELECT id, consolidated_at, consolidation_claimed_at FROM working_memory")}
    episodic = [r[0] for r in conn.execute("SELECT summary_of FROM episodic_memory")]
    conn.close()
    return wm, episodic


def test_sleep_skip_session_pattern_no_llm_no_episodic(tmp_path, llm_calls):
    db = tmp_path / "beam.db"
    beam = BeamMemory(session_id="live", db_path=db)
    _insert(db, [("f1", "hermes_agent:main:a2a_fleet:dm:x", "[USER] a fleet message " * 5),
                 ("f2", "hermes_agent:main:a2a_fleet:dm:x", "[USER] another fleet message " * 5)])
    result = beam.sleep_all_sessions(require_host_llm=True, min_age_hours=24, skip_session_patterns=["a2a_fleet"])
    assert llm_calls == []
    wm, episodic = _state(db)
    assert wm == {"f1": (True, None), "f2": (True, None)}  # consolidated, no claim left behind
    assert episodic == []
    assert (result["sessions_skipped"], result["items_skipped"], result["summaries_created"]) == (1, 2, 0)
    # Never comes back: neither orphan reclaim nor a later sweep re-queues them.
    beam.reclaim_orphans(stale_after_seconds=0)
    assert beam.sleep_all_sessions(require_host_llm=True, min_age_hours=24)["sessions_scanned"] == 0


def test_sleep_min_session_chars(tmp_path, llm_calls):
    db = tmp_path / "beam.db"
    beam = BeamMemory(session_id="live", db_path=db)
    _insert(db, [("s1", "short-chat", "[USER] ok"), ("s2", "short-chat", "[USER] yes"),
                 ("l1", "long-chat", "[USER] " + "a long and substantive message " * 20)])
    result = beam.sleep_all_sessions(require_host_llm=True, min_age_hours=24, min_session_chars=300)
    wm, episodic = _state(db)
    assert all(done for done, _ in wm.values())
    assert episodic == ["l1"]  # only the long session got a summary
    assert result["sessions_skipped"] == 1 and result["sessions_consolidated"] == 1
    assert llm_calls  # the long session did go through the LLM


def test_sleep_defaults_unchanged(tmp_path, llm_calls):
    db = tmp_path / "beam.db"
    beam = BeamMemory(session_id="live", db_path=db)
    _insert(db, [("f1", "hermes_agent:main:a2a_fleet:dm:x", "[USER] ok")])
    result = beam.sleep_all_sessions(require_host_llm=True, min_age_hours=24)
    assert result["sessions_skipped"] == 0 and result["summaries_created"] == 1
    assert _state(db)[1] == ["f1"]


def test_nightly_threads_sleep_hygiene_and_retention(tmp_path, monkeypatch, llm_calls):
    p = _provider(tmp_path, nightly_sleep_enabled=True, sleep_skip_session_patterns="a2a_fleet",
                  archive_consolidated_after_days=14)
    db = p._beam.db_path
    _insert(db, [("f1", "hermes_agent:main:a2a_fleet:dm:x", "[USER] fleet " * 30)])
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO working_memory (id, content, source, timestamp, session_id, consolidated_at) "
                 "VALUES ('old', '[USER] old', 'conversation', ?, 'c', ?)",
                 ((datetime.now() - timedelta(days=40)).isoformat(),
                  (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=20)).isoformat()))
    conn.commit()
    conn.close()
    monkeypatch.setattr(hmp, "_nightly_now", lambda: datetime(2026, 10, 7, 3, 30))
    result = hmp._nightly_tick(hmp._NIGHTLY_JOBS[str(db)])
    assert result["sessions_skipped"] == 1 and result["archived"] == 1
    assert llm_calls == []
    assert _state(db)[1] == []


# --- retention --------------------------------------------------------------

def test_archive_consolidated_only_old_unpinned_conversation_rows(tmp_path):
    db = tmp_path / "beam.db"
    beam = BeamMemory(session_id="live", db_path=db)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS audit_log (event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "timestamp REAL NOT NULL, action TEXT NOT NULL, memory_id TEXT, bank TEXT, scope TEXT, "
                 "profile TEXT, session_id TEXT, source_tool TEXT, tokens_used INTEGER, reason TEXT, "
                 "metadata_json TEXT)")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [  # id, source, consolidated days ago (None = not consolidated), pinned
        ("old1", "conversation", 30, 0), ("old2", "conversation", 20, 0), ("old3", "conversation", 15, 0),
        ("recent", "conversation", 3, 0), ("pinned", "conversation", 30, 1),
        ("note", "remembered", 30, 0), ("builtin", "builtin_memory", 30, 0), ("raw", "conversation", None, 0),
    ]
    for rid, source, days, pinned in rows:
        conn.execute("INSERT INTO working_memory (id, content, source, timestamp, session_id, consolidated_at, pinned) "
                     "VALUES (?, ?, ?, ?, 's', ?, ?)",
                     (rid, rid, source, (now - timedelta(days=40)).isoformat(),
                      None if days is None else (now - timedelta(days=days)).isoformat(), pinned))
    conn.commit()

    assert beam.archive_consolidated(14, max_rows=2) == 2  # cap: oldest first
    assert beam.archive_consolidated(14, max_rows=2000) == 1
    archived = {r[0]: r[1] for r in conn.execute(
        "SELECT id, superseded_by FROM working_memory WHERE valid_until IS NOT NULL")}
    assert archived == {"old1": "archived:consolidated", "old2": "archived:consolidated",
                        "old3": "archived:consolidated"}
    audit = conn.execute("SELECT action, source_tool, metadata_json FROM audit_log").fetchall()
    assert [a[:2] for a in audit] == [("archive_consolidated", "nightly_sleep")] * 2
    assert '"count": 2' in audit[0][2]
    conn.close()
