"""Connections stranded by exited threads must not accumulate (hermes-agent#196).

Every ``sqlite3.Connection`` is cyclic garbage — CPython builds the statement
cache as ``lru_cache(...)(connection)``, so the connection references itself
and refcounting can never free it. The memory layer caches one connection per
thread, so a daemon serving short-lived threads leaked handles until ``open()``
failed with EMFILE. ``conn_sweep`` periodically runs a cyclic collection.
"""

import gc
import os
import subprocess
import sys
import threading

import pytest


def mnemosyne_db_handles() -> int:
    out = subprocess.run(["lsof", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if "mnemosyne.db" in line)


@pytest.fixture
def memory(tmp_path, monkeypatch):
    """Point mnemosyne at a throwaway data dir, never the real profile.

    Deliberately does NOT purge ``sys.modules``: _default_data_dir() re-reads
    the environment on every call, so setenv alone is enough. An earlier
    version of this fixture dropped every mnemosyne module to force a
    re-import, which re-executed them mid-run and broke 58 unrelated tests
    later in the suite — the same sys.modules pollution class of bug as
    fix/gateway-test-pollution.
    """
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    from mnemosyne.core import memory as m
    m.init_db()
    return m


def serve_on_threads(memory, n):
    """Open a connection on n short-lived threads, holding no references."""
    for _ in range(n):
        t = threading.Thread(
            target=lambda: memory._get_connection().execute("SELECT 1").fetchone()
        )
        t.start()
        t.join()


def test_sqlite_connections_are_self_referential():
    """The premise: this is why refcounting alone never closes them.

    If a future CPython drops the lru_cache statement cache, the sweep is
    unnecessary and this test should start failing — that is the signal.
    """
    import sqlite3
    conn = sqlite3.connect(":memory:")
    try:
        wrappers = [r for r in gc.get_referrers(conn)
                    if type(r).__name__ == "_lru_cache_wrapper"]
        assert wrappers, "expected sqlite3 to hold a statement cache over the connection"
        assert any(w.__wrapped__ is conn for w in wrappers), \
            "expected the statement cache to wrap the connection itself"
    finally:
        conn.close()


@pytest.mark.skipif(sys.platform == "win32", reason="lsof is POSIX-only")
def test_handles_stay_bounded_across_many_threads(memory):
    """Handle count must not scale with the number of threads served."""
    from mnemosyne.core import conn_sweep

    serve_on_threads(memory, conn_sweep._SWEEP_EVERY * 2)
    after_first = mnemosyne_db_handles()

    serve_on_threads(memory, conn_sweep._SWEEP_EVERY * 6)
    after_second = mnemosyne_db_handles()

    growth = after_second - after_first
    assert growth <= 4, (
        f"handles grew by {growth} over 6 more sweep intervals "
        f"({after_first} -> {after_second}); connections are accumulating"
    )


def test_sweep_only_fires_once_per_interval():
    from mnemosyne.core import conn_sweep

    conn_sweep._since_sweep = 0
    fired = [conn_sweep.note_connection_opened()
             for _ in range(conn_sweep._SWEEP_EVERY)]
    assert fired.count(True) == 1, "expected exactly one sweep per interval"
    assert fired[-1] is True, "expected the sweep on the interval boundary"


def test_sweep_does_not_close_a_referenced_connection(memory, tmp_path):
    """A connection someone still holds must survive a sweep.

    This is the constraint that sank two earlier attempts at this fix: these
    connections are opened check_same_thread=False and legitimately outlive
    the thread that created them.
    """
    from mnemosyne.core import conn_sweep

    held = memory.Mnemosyne(session_id="holder")
    other = memory.Mnemosyne(session_id="other", db_path=tmp_path / "other.db")

    conn_sweep._since_sweep = 0
    for _ in range(conn_sweep._SWEEP_EVERY):
        conn_sweep.note_connection_opened()

    assert held.conn.execute("SELECT 1").fetchone()[0] == 1
    assert other.conn.execute("SELECT 1").fetchone()[0] == 1
