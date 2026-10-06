"""Pre-go-live fixes: bounded prefetch lock wait; dry_run honoured with the safety gate off."""
import json
import threading
import time

import pytest

import hermes_memory_provider as hmp
from hermes_memory_provider import MnemosyneMemoryProvider


def _call(p, name, args):
    return json.loads(p.handle_tool_call(name, args))


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.delenv("MNEMOSYNE_MATRIX_SAFETY", raising=False)
    home = tmp_path / "hermes"
    (home / "memories").mkdir(parents=True)
    p = MnemosyneMemoryProvider()
    p.initialize(session_id="prelive", hermes_home=str(home))
    yield p
    p.shutdown()


def _hold(p, seconds):
    lock, held = p._ensure_beam_access_lock(), threading.Event()

    def run():
        with lock:
            held.set()
            time.sleep(seconds)

    t = threading.Thread(target=run)
    t.start()
    assert held.wait(5)
    return t


def test_prefetch_returns_empty_when_lock_stuck(provider, monkeypatch):
    monkeypatch.setattr(hmp, "_PREFETCH_LOCK_TIMEOUT_S", 0.3)
    t = _hold(provider, 1.5)
    t0 = time.monotonic()
    assert provider.prefetch("anything") == ""
    assert time.monotonic() - t0 < 1.0
    t.join()


def test_prefetch_waits_for_short_hold(provider, monkeypatch):
    monkeypatch.setattr(hmp, "_PREFETCH_LOCK_TIMEOUT_S", 3.0)
    called = []
    monkeypatch.setattr(provider, "_prefetch_locked", lambda q, session_id="": called.append(q) or "ok")
    t = _hold(provider, 0.2)
    assert provider.prefetch("q") == "ok"
    t.join()
    assert called == ["q"]


def test_prefetch_reentrant_same_thread(provider, monkeypatch):
    monkeypatch.setattr(hmp, "_PREFETCH_LOCK_TIMEOUT_S", 0.3)
    monkeypatch.setattr(provider, "_prefetch_locked", lambda q, session_id="": "ok")
    with provider._ensure_beam_access_lock():
        assert provider.prefetch("q") == "ok"
    # lock fully released afterwards
    assert provider._ensure_beam_access_lock().acquire(blocking=False)
    provider._ensure_beam_access_lock().release()


def test_forget_dry_run_gate_off_keeps_row(provider):
    mid = _call(provider, "mnemosyne_remember", {"content": "keep me around please"})["memory_id"]
    out = _call(provider, "mnemosyne_forget", {"memory_id": mid, "dry_run": True})
    assert out["dry_run"] is True and out["action"] == "mnemosyne_forget"
    assert provider._beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id=?", (mid,)).fetchone()[0] == 1
    # absent / false still deletes
    assert _call(provider, "mnemosyne_forget", {"memory_id": mid, "dry_run": False})["status"] == "deleted"


def test_update_dry_run_gate_off_keeps_row(provider):
    mid = _call(provider, "mnemosyne_remember", {"content": "original text here"})["memory_id"]
    out = _call(provider, "mnemosyne_update", {"memory_id": mid, "content": "changed", "dry_run": True})
    assert out["dry_run"] is True
    row = provider._beam.conn.execute(
        "SELECT content FROM working_memory WHERE id=?", (mid,)).fetchone()
    assert row[0] == "original text here"
    out = _call(provider, "mnemosyne_update", {"memory_id": mid, "content": "changed"})
    assert "dry_run" not in out
    row = provider._beam.conn.execute(
        "SELECT content FROM working_memory WHERE id=?", (mid,)).fetchone()
    assert row[0] == "changed"


def test_remember_gate_off_still_writes(provider):
    assert _call(provider, "mnemosyne_remember", {"content": "plain write"}).get("status") == "stored"
