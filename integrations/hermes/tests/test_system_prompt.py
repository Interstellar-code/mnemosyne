from __future__ import annotations

from mnemosyne_hermes import MnemosyneMemoryProvider


def test_system_prompt_routes_to_structured_mnemosyne_surfaces():
    provider = MnemosyneMemoryProvider()
    provider._beam = object()

    block = provider.system_prompt_block()

    assert "Mnemosyne is primary" in block
    assert "legacy memory tool is deprecated" in block
    assert "mnemosyne_recall" in block
    assert "mnemosyne_remember for ordinary facts/preferences/insights" in block
    assert "mnemosyne_remember_canonical" in block
    assert "mnemosyne_triple_add" in block
    assert "mnemosyne_graph_link/query" in block
    assert "mnemosyne_validate/invalidate/update/forget" in block
    assert "mnemosyne_scratchpad_*" in block
    assert "mnemosyne_shared_*" in block
    assert "Do not save one-off task progress" in block


def test_system_prompt_tells_agent_to_read_mnemosyne_context_first():
    """Pin the rule added in #321.

    The Mnemosyne context block is injected into the turn and the agent
    should answer from it before reaching for session_search. Without
    this standing rule, the agent defaults to retrieval tools out of
    habit and wastes a turn.
    """
    provider = MnemosyneMemoryProvider()
    provider._beam = object()

    block = provider.system_prompt_block()

    assert "## Mnemosyne Context" in block
    assert "read it before calling retrieval tools" in block
    assert "answer directly" in block
    assert "session_search" in block
    assert "missing, stale, or insufficient" in block
