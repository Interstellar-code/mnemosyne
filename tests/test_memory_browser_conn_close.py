"""The browser must not leak a connection when a query raises (hermes-agent#196).

Each of these functions used to call ``conn.close()`` as the last statement
inside its ``try``, so any exception on the way there skipped the close and
stranded the handle. In a long-lived daemon that accumulates until ``open()``
starts failing with EMFILE.
"""

import sqlite3

import pytest

from mnemosyne.integrations import memory_browser


@pytest.fixture
def corrupt_db(tmp_path):
    """A file that is not a database, so every query against it raises."""
    db = tmp_path / "corrupt.sqlite"
    db.write_bytes(b"this is definitely not a sqlite database")
    return db


@pytest.fixture
def track_connections(monkeypatch):
    """Record every connection the browser opens."""
    opened = []
    real = memory_browser._get_connection

    def tracking(db_path):
        conn = real(db_path)
        opened.append(conn)
        return conn

    monkeypatch.setattr(memory_browser, "_get_connection", tracking)
    return opened


def _assert_all_closed(opened):
    """Every recorded connection must be *closed*, not merely unusable.

    The check has to be specific: a corrupt database raises
    ``DatabaseError("file is not a database")`` on execute whether or not the
    connection was closed, so a bare ``pytest.raises(Exception)`` passes even
    against the unfixed code. Only a closed connection raises
    ``ProgrammingError`` mentioning "closed".
    """
    assert opened, "expected the browser to have opened at least one connection"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


def test_get_memory_stats_closes_on_error(corrupt_db, track_connections):
    result = memory_browser.get_memory_stats(str(corrupt_db))
    assert "error" in result, "expected the corrupt database to surface an error"
    _assert_all_closed(track_connections)


def test_search_memories_closes_on_error(corrupt_db, track_connections):
    assert memory_browser.search_memories(str(corrupt_db), query="anything") == []
    _assert_all_closed(track_connections)


def test_get_memory_detail_closes_on_error(corrupt_db, track_connections):
    assert memory_browser.get_memory_detail(str(corrupt_db), "some-id") is None
    _assert_all_closed(track_connections)


def test_get_memory_detail_closes_on_the_early_return_path(tmp_path, track_connections):
    """The early ``return result`` branch must close too, not just the fall-through."""
    db = tmp_path / "real.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO memories VALUES ('abc', 'hello')")
    conn.commit()
    conn.close()

    found = memory_browser.get_memory_detail(str(db), "abc")
    assert found is not None and found["tier"] == "memories"
    _assert_all_closed(track_connections)
