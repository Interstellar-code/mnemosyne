"""
Test dynamic HERMES_HOME path resolution in Mnemosyne (_default_data_dir) (#160).
"""
import os
from pathlib import Path

from mnemosyne.core.memory import _default_data_dir as memory_default_dir
from mnemosyne.core.beam import _default_data_dir as beam_default_dir
from mnemosyne.core.banks import _default_data_dir as banks_default_dir


def test_mnemosyne_default_data_dir_resolves_dynamically(tmp_path, monkeypatch):
    p1 = tmp_path / "profile_a"
    p2 = tmp_path / "profile_b"
    p1.mkdir()
    p2.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(p1))
    assert memory_default_dir() == p1 / "mnemosyne" / "data"
    assert beam_default_dir() == p1 / "mnemosyne" / "data"
    assert banks_default_dir() == p1 / "mnemosyne" / "data"

    monkeypatch.setenv("HERMES_HOME", str(p2))
    assert memory_default_dir() == p2 / "mnemosyne" / "data"
    assert beam_default_dir() == p2 / "mnemosyne" / "data"
    assert banks_default_dir() == p2 / "mnemosyne" / "data"
