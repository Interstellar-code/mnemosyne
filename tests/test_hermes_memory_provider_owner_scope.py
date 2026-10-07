from __future__ import annotations

from hermes_memory_provider import MnemosyneMemoryProvider


def _provider(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    home = tmp_path / "home"
    home.mkdir()
    p = MnemosyneMemoryProvider()
    p.initialize(session_id="s1", hermes_home=str(home), agent_identity="hermes-switch")
    assert p._beam is not None
    return p


def test_beam_canonical_owner_follows_agent_identity(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    assert p._beam.canonical_owner_id == p._canonical_owner() == "hermes-switch"


def test_sync_turn_stays_session_scoped_under_global_default(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    p._default_scope = "global"
    p._sync_roles = {"user", "assistant"}
    calls = []
    real = p._beam.remember

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(p._beam, "remember", spy)
    p.sync_turn("user says something long enough", "assistant replies with enough text")
    conv = [c for c in calls if c.get("source") == "conversation"]
    assert len(conv) == 2
    assert {c["scope"] for c in conv} == {"session"}
