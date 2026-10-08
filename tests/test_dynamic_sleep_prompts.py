"""Dynamic nightly-consolidation prompts (PROMPTS.md §5).

Placeholders resolved once per sweep, prompt files read per sweep, render
errors falling back to the built-in prompt, the NOTHING_DURABLE sentinel, and
the model-refresh weak-slot guards. No real LLM: a fake host backend answers.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core import local_llm, model_refresh
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.canonical import CanonicalStore
from mnemosyne.core.llm_backends import CallableLLMBackend, set_host_llm_backend

OWNER = "hermes-switch"
IDENTITY = "Rohit Sharma. Lives in Waldkirch, Germany. Works as a consultant."
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    data = tmp_path / "mnemosyne"
    data.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data))
    for var in ("MNEMOSYNE_PRINCIPAL_NAME", "MNEMOSYNE_SLEEP_PROMPT", "MNEMOSYNE_SLEEP_PROMPT_FILE",
                "MNEMOSYNE_SLEEP_MODEL_REFRESH_PROMPT", "MNEMOSYNE_SLEEP_MODEL_REFRESH_PROMPT_FILE",
                "MNEMOSYNE_SLEEP_PROMPT_CARD_CHARS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(local_llm, "SLEEP_PROMPT", "")
    monkeypatch.setattr(local_llm, "_render_warned", False)
    return data


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def host(monkeypatch):
    """Fake host LLM. Set .summary / .refresh to control replies; .prompts records."""

    class Host:
        summary = "Rohit chose the blue theme."
        refresh = "[]"
        prompts = []

    def _complete(prompt, **kwargs):
        Host.prompts.append(prompt)
        if "canonical model slots" in prompt or prompt.startswith("REFRESH"):
            return Host.refresh
        return Host.summary(prompt) if callable(Host.summary) else Host.summary

    Host.prompts = []
    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
    set_host_llm_backend(CallableLLMBackend("fake-host", _complete))
    try:
        yield Host
    finally:
        set_host_llm_backend(None)


def _insert(db, rows):
    """rows: (id, session_id, hours_old, content)"""
    conn = sqlite3.connect(db)
    for rid, sid, hours, content in rows:
        ts = (datetime.now() - timedelta(hours=hours)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, 'conversation', ?, ?)", (rid, content, ts, sid))
    conn.commit()
    conn.close()


def _q(db, sql):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _beam(db, identity=True):
    beam = BeamMemory(session_id="current", db_path=db)
    beam.canonical_owner_id = OWNER
    if identity:
        store = CanonicalStore(db_path=db, conn=beam.conn)
        store.remember(OWNER, "model:user", "identity", IDENTITY)
        store.remember(OWNER, "model:workflow", "self-verification preference", "Verify before claiming done.")
    return beam


# 1 ---------------------------------------------------------------------------
def test_render_prompt_vars(caplog):
    values = {"principal": "Rohit", "memories": "- a"}
    assert local_llm._render_prompt("{principal}|{nope}|{{x}}|{memories}", values) == "Rohit||{x}|- a"
    with caplog.at_level(logging.WARNING, logger="mnemosyne.core.local_llm"):
        assert local_llm._render_prompt("broken { brace", values) is None
        assert local_llm._render_prompt("also {0[x]} broken", values) is None
    assert len([r for r in caplog.records if "does not render" in r.getMessage()]) == 1
    # A broken template yields the built-in prompt, never an exception.
    pv = {"_sleep_template": "bad {"}
    assert local_llm._build_host_prompt(["note"], source="s", prompt_vars=pv).startswith("Summarize")
    pv = {**local_llm.PROMPT_VAR_DEFAULTS, "_sleep_template": "{principal}/{session_kind}/{memory_count}:{memories}",
          "principal": "Rohit", "session_kind": "dm"}
    assert local_llm._build_host_prompt(["a", "b"], prompt_vars=pv) == "Rohit/dm/2:- a\n- b"


def test_broken_prompt_file_does_not_strand_rows(db, host, _isolated):
    (_isolated / "sleep_prompt.md").write_text("Summarise for {principal} {broken")
    beam = _beam(db)
    _insert(db, [("a", "s1", 400, "Rohit wants the blue theme everywhere.")])
    result = beam.sleep_all_sessions(require_host_llm=True)
    assert result["items_consolidated"] == 1
    assert host.prompts[0].startswith("Summarize the following memories")  # built-in
    assert _q(db, "SELECT consolidated_at IS NOT NULL, consolidation_claimed_at FROM working_memory "
                  "WHERE id='a'") == [(1, None)]
    assert _q(db, "SELECT COUNT(*) FROM episodic_memory") == [(1,)]


def test_prompt_file_precedence_and_live_edit(db, host, _isolated, tmp_path, monkeypatch):
    beam = _beam(db)
    default = _isolated / "sleep_prompt.md"
    default.write_text("DEFAULT for {principal} ({session_kind}, {date_range}, {profile}):\n{memories}")
    _insert(db, [("a", "hermes_agent:main:telegram:dm:42", 400, "note one is long enough")])
    beam.sleep_all_sessions(require_host_llm=True)
    today = (datetime.now() - timedelta(hours=400)).date().isoformat()
    assert host.prompts[0] == f"DEFAULT for Rohit Sharma (dm, {today}, {OWNER}):\n- note one is long enough"

    # Edited file applies on the next sweep without any reload; config path beats default.
    explicit = tmp_path / "explicit.md"
    explicit.write_text("EXPLICIT {principal}: {memories}")
    monkeypatch.setenv("MNEMOSYNE_SLEEP_PROMPT_FILE", str(explicit))
    _insert(db, [("b", "s2", 400, "note two")])
    beam.sleep_all_sessions(require_host_llm=True, prompt_settings={"principal_name": "R."})
    assert "EXPLICIT R.: - note two" in host.prompts

    # Missing files fall through to the env template, then to the built-in.
    monkeypatch.setenv("MNEMOSYNE_SLEEP_PROMPT_FILE", str(tmp_path / "missing.md"))
    default.unlink()
    monkeypatch.setenv("MNEMOSYNE_SLEEP_PROMPT", "ENV {memory_count}")
    assert local_llm._sleep_prompt_template("sleep") == "ENV {memory_count}"
    monkeypatch.delenv("MNEMOSYNE_SLEEP_PROMPT")
    assert local_llm._sleep_prompt_template("sleep") is None


# 2 ---------------------------------------------------------------------------
def test_resolve_prompt_vars(db, monkeypatch):
    beam = _beam(db)
    v = local_llm.resolve_prompt_vars(beam.conn, db, OWNER)
    assert v["principal"] == "Rohit Sharma"
    assert v["principal_card"] == IDENTITY
    assert v["profile"] == OWNER
    assert v["existing_slots"] == "- model:user/identity\n- model:workflow/self-verification preference"
    assert v["_slot_names"] == ("model:user/identity", "model:workflow/self-verification preference")

    monkeypatch.setenv("MNEMOSYNE_PRINCIPAL_NAME", "Env Name")
    assert local_llm.resolve_prompt_vars(beam.conn, db, OWNER)["principal"] == "Env Name"
    v = local_llm.resolve_prompt_vars(beam.conn, db, OWNER, settings={"principal_name": "Config Name"})
    assert v["principal"] == "Config Name"  # host config beats env and the slot
    monkeypatch.delenv("MNEMOSYNE_PRINCIPAL_NAME")

    assert local_llm.resolve_prompt_vars(beam.conn, db, OWNER, card_chars=45)["principal_card"] == \
        "Rohit Sharma. Lives in Waldkirch, Germany."
    v = local_llm.resolve_prompt_vars(beam.conn, db, OWNER, card_chars=0)
    assert v["principal_card"] == "(no description on file)" and v["principal"] == "Rohit Sharma"
    monkeypatch.setenv("MNEMOSYNE_SLEEP_PROMPT_CARD_CHARS", "13")
    assert local_llm.resolve_prompt_vars(beam.conn, db, OWNER)["principal_card"] == "Rohit Sharma."

    v = local_llm.resolve_prompt_vars(beam.conn, db, "other-profile")
    assert (v["principal"], v["principal_card"], v["existing_slots"]) == \
        ("the user", "(no description on file)", "(none)")

    v = local_llm.resolve_prompt_vars(beam.conn, db, OWNER, settings={"sleep_prompt_card_chars": 0})
    assert v["principal_card"] == "(no description on file)"  # int 0 from YAML omits too

    conn = sqlite3.connect(":memory:")
    conn.close()
    v = local_llm.resolve_prompt_vars(conn, db, OWNER)  # never raises
    assert v["principal"] == "the user" and v["_slot_names"] is None
    v = local_llm.resolve_prompt_vars(conn, db, OWNER, settings={"principal_name": "Rohit"})
    assert v["principal"] == "Rohit"  # configured name survives a DB failure


# 3 ---------------------------------------------------------------------------
def test_first_sentence_name():
    assert local_llm._first_sentence_name("Rohit Sharma. Lives in Waldkirch.") == "Rohit Sharma"
    assert local_llm._first_sentence_name("The user is Rohit, 45, from Waldkirch.") == "Rohit"
    assert local_llm._first_sentence_name("A consultant who lives in Waldkirch and works on Hermes.") is None


def test_session_kind_and_date_range():
    assert local_llm.session_kind("hermes_agent:main:telegram:dm:12345") == "dm"
    assert local_llm.session_kind("hermes_agent:main:telegram:group:-100:7") == "group"
    assert local_llm.session_kind("hermes_api_abc123") == "api"
    assert local_llm.session_kind("hermes_0b7e6c1d-1111-2222-3333-444455556666") == "cli"
    assert local_llm.session_kind("default") == "chat"
    assert local_llm.session_kind(None) == "chat"
    assert local_llm.date_range([{"timestamp": "2026-07-02T10:00:00"},
                                 {"timestamp": "2026-07-02T23:00:00"}]) == "2026-07-02"
    assert local_llm.date_range([{"timestamp": "2026-07-02T10:00:00"}, {"timestamp": "now"},
                                 {"timestamp": "2026-06-30T01:00:00"}]) == "2026-06-30 to 2026-07-02"
    assert local_llm.date_range([]) == "unknown date"


# 4 ---------------------------------------------------------------------------
def test_sleep_nothing_durable(db, host):
    host.summary = "NOTHING_DURABLE"
    host.refresh = json.dumps([{"category": "model:workflow", "name": "x", "body": "Always x.",
                                "confidence": 0.95, "evidence_ids": ["a", "b"], "action": "update"}])
    beam = _beam(db)
    _insert(db, [("a", "s1", 400, "work kanban task t_b72a2a02"), ("b", "s1", 399, "ok")])

    result = beam.sleep_all_sessions(require_host_llm=True)  # allow_aaak=False

    assert result["items_skipped"] == 2 and result["nothing_durable"] == 1
    assert result["items_consolidated"] == 0
    assert result["session_results"][0]["status"] == "skipped"
    assert _q(db, "SELECT COUNT(*) FROM working_memory WHERE id IN ('a','b') "
                  "AND consolidated_at IS NOT NULL AND consolidation_claimed_at IS NULL") == [(2,)]
    assert _q(db, "SELECT COUNT(*) FROM episodic_memory") == [(0,)]
    assert _q(db, "SELECT COUNT(*) FROM working_memory WHERE source='sleep_model_refresh_proposal'") == [(0,)]
    calls = len(host.prompts)
    assert calls == 1  # summary only; model refresh never ran
    again = beam.sleep_all_sessions(require_host_llm=True)
    assert again["status"] == "no_op" and len(host.prompts) == calls  # no nightly retry loop

    # allow_aaak=True must not AAAK-encode a NOTHING_DURABLE group either.
    _insert(db, [("c", "current", 400, "thanks!")])
    single = beam.sleep(allow_aaak=True)
    assert single["status"] == "skipped" and single["nothing_durable"] == 1
    assert _q(db, "SELECT COUNT(*) FROM episodic_memory") == [(0,)]
    assert local_llm._nothing_durable_or(" nothing_durable. ") is local_llm._NOTHING_DURABLE
    assert local_llm._nothing_durable_or("NOTHING_DURABLE here, but also a fact") != local_llm._NOTHING_DURABLE


# 5 ---------------------------------------------------------------------------
def test_summarize_mixed_chunks(host, monkeypatch):
    monkeypatch.setattr(local_llm, "chunk_memories_by_budget", lambda m, source="", **_: [[x] for x in m])
    host.summary = lambda p: "NOTHING_DURABLE" if "- junk" in p else "Real fact from B."
    assert local_llm._summarize_memories(["junk", "fact"]) == "Real fact from B."
    assert local_llm._summarize_memories(["junk", "junk"]) is local_llm._NOTHING_DURABLE
    assert local_llm.summarize_memories(["junk"]) is None


# 6 ---------------------------------------------------------------------------
def test_chunk_budget_uses_rendered_header(monkeypatch):
    monkeypatch.setattr(local_llm, "_prompt_token_budget", lambda: 1000)
    memories = ["x" * 40] * 200  # 10 + 1 tokens each
    builtin = local_llm.chunk_memories_by_budget(memories, prompt_vars={"_sleep_template": None})
    custom = local_llm.chunk_memories_by_budget(memories, prompt_vars={"_sleep_template": "y" * 3000 + "{memories}"})
    shrink_tokens = (len(builtin[0]) - len(custom[0])) * 11
    assert 700 <= shrink_tokens <= 800


# 7 ---------------------------------------------------------------------------
def test_model_refresh_prompt_override_and_guard(db, host, _isolated):
    beam = _beam(db)
    items = [{"id": "a", "content": "[USER] always run tests"}, {"id": "b", "content": "[USER] yes always"}]
    builtin = model_refresh.build_model_refresh_prompt(
        items, prompt_vars=local_llm.resolve_prompt_vars(beam.conn, db, OWNER))
    assert "Hermes, the personal agent of Rohit Sharma" in builtin
    assert "- model:workflow/self-verification preference" in builtin
    assert "- id=a: [USER] always run tests" in builtin

    (_isolated / "model_refresh_prompt.md").write_text("REFRESH {principal}\n{existing_slots}\n{memories}")
    pv = local_llm.resolve_prompt_vars(beam.conn, db, OWNER)
    assert model_refresh.build_model_refresh_prompt(items, prompt_vars=pv).startswith(
        "REFRESH Rohit Sharma\n- model:user/identity\n- model:workflow/self-verification preference\n- id=a:")

    def p(name, body="Always run the tests.", category="model:workflow", action="update"):
        return {"category": category, "name": name, "body": body, "confidence": 0.95,
                "evidence_ids": ["a", "b"], "action": action}

    host.refresh = json.dumps([
        p("Rohit", category="model:user"),
        p("rohit sharma", category="model:user"),
        p("users/rohits"),
        p("home", body="The user is Rohit (home directory /Users/rohits).", category="model:user"),
        p("identity", body="Rohit Sharma is Rohit, the owner.", category="model:user"),
        p("Coding Style"),
        p("Self-Verification Preference", body="Verify, then report."),
        p("devices", body="Rohit uses a Mac mini.", category="model:user", action="keep"),
    ])
    out = model_refresh.infer_model_update_proposals(items, prompt_vars=pv)
    assert [(x["name"], x["action"]) for x in out] == [
        ("coding-style", "update"), ("self-verification preference", "update"), ("devices", "keep")]
    assert host.prompts[-1].startswith("REFRESH Rohit Sharma")
    # Without principal / slot knowledge (legacy callers) names pass through untouched.
    raw = json.dumps([p("Rohit", category="model:user"), p("Coding Style"), p("x", action="ignore")])
    assert [x["name"] for x in model_refresh.parse_model_update_proposals(raw)] == ["Rohit", "Coding Style", "x"]


def test_sleep_stores_only_update_proposals(db, host):
    host.refresh = json.dumps([
        {"category": "model:workflow", "name": "test-policy", "body": "Always run the tests first.",
         "confidence": 0.5, "evidence_ids": ["a", "b"], "action": "update"},
        {"category": "model:workflow", "name": "self-verification preference", "body": "Verify before claiming done.",
         "confidence": 0.95, "evidence_ids": ["a", "b"], "action": "keep"},
    ])
    beam = _beam(db)
    _insert(db, [("a", "s1", 400, "always run tests first"), ("b", "s1", 399, "yes, always")])
    result = beam.sleep_all_sessions(require_host_llm=True)
    assert result["model_refresh"]["proposals"] == 1
    rows = _q(db, "SELECT metadata_json FROM working_memory WHERE source='sleep_model_refresh_proposal'")
    assert [json.loads(r[0])["action"] for r in rows] == ["update"]


# 8 ---------------------------------------------------------------------------
def test_sweep_resolves_once(db, host, monkeypatch):
    beam = _beam(db)
    _insert(db, [("a", "s1", 400, "fact one"), ("b", "s2", 300, "fact two"), ("c", "s3", 200, "fact three")])
    real = local_llm.resolve_prompt_vars
    calls, marker, seen = [], object(), []

    def counting(*a, **k):
        calls.append(1)
        return {**real(*a, **k), "_marker": marker}

    real_summarize = local_llm._summarize_memories

    def spy(memories, source="", **k):
        seen.append(k.get("prompt_vars"))
        return real_summarize(memories, source, **k)

    monkeypatch.setattr(local_llm, "resolve_prompt_vars", counting)
    monkeypatch.setattr(local_llm, "_summarize_memories", spy)
    result = beam.sleep_all_sessions(require_host_llm=True)
    assert result["items_consolidated"] == 3
    assert len(calls) == 1
    assert len(seen) == 3 and all(v["_marker"] is marker and v["principal"] == "Rohit Sharma" for v in seen)
    assert beam._sleep_prompt_vars is None  # restored after the sweep


def test_example_prompt_files_match_builtin():
    assert (REPO / "examples" / "model_refresh_prompt.md").read_text().strip() == model_refresh.MODEL_REFRESH_PROMPT
    text = (REPO / "examples" / "sleep_prompt.md").read_text()
    rendered = local_llm._render_prompt(text, {**local_llm.PROMPT_VAR_DEFAULTS, "memories": "- x"})
    assert rendered and "NOTHING_DURABLE" in rendered and "{" not in rendered


def test_provider_passes_hermes_prompt_settings(monkeypatch):
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider.__new__(MnemosyneMemoryProvider)
    provider._sleep_skip_session_patterns = ()
    provider._sleep_min_session_chars = 0
    cfg = {"principal_name": "Rohit Sharma", "sleep_prompt_card_chars": 0, "sleep_prompt_file": ""}
    monkeypatch.setattr(provider, "_read_config_key", cfg.get)
    assert provider._sleep_hygiene_kwargs() == {
        "prompt_settings": {"principal_name": "Rohit Sharma", "sleep_prompt_card_chars": 0}}
    monkeypatch.setattr(provider, "_read_config_key", lambda key: None)
    assert provider._sleep_hygiene_kwargs() == {}
