"""Reclaim SQLite connections stranded by threads that have exited.

Every ``sqlite3.Connection`` is unavoidably cyclic garbage: CPython builds its
statement cache as ``functools.lru_cache(maxsize=...)(connection)``, so the
connection is its own ``__wrapped__`` and refcounting can never free it. Only
the cyclic collector can. Verified on CPython 3.13:

    >>> [type(r).__name__ for r in gc.get_referrers(sqlite3.connect(path))]
    ['_lru_cache_wrapper', 'dict', 'dict']

This matters because the memory layer caches a connection per thread in a
``threading.local``. When a short-lived thread exits, its connection becomes
unreachable but stays *open* until a cyclic collection happens to run. A
long-lived daemon serving work on pooled or short-lived threads accumulates
those handles until every ``open()`` fails with ``EMFILE`` — hermes-agent#196
reported 163 handles to one database in a single gateway process.

The fix is deliberately *not* to close connections ourselves. They are opened
``check_same_thread=False`` and legitimately outlive the thread that created
them (``Mnemosyne.conn`` holds one for the object's whole lifetime), so
"the owning thread exited" does not imply "this connection is garbage".
Closing on that signal breaks live callers — it was tried, and it broke 131
tests. Reachability is the only correct discriminator, and ``gc.collect()``
is exactly that: it reclaims connections nothing references and leaves every
connection someone still holds alone.
"""

from __future__ import annotations

import gc
import threading

# ponytail: a plain counter, swept on the slow path. Connections are created
# once per thread, not per operation, so this runs rarely; a time-based or
# fd-watermark trigger would be more precise and is not worth it until the
# fixed interval demonstrably fails.
_SWEEP_EVERY = 50

_lock = threading.Lock()
_since_sweep = 0


def note_connection_opened() -> bool:
    """Record that a new connection was opened; sweep every _SWEEP_EVERY.

    Call this on the connection *creation* path only. Returns True when a
    collection actually ran, which the tests assert on.
    """
    global _since_sweep
    with _lock:
        _since_sweep += 1
        if _since_sweep < _SWEEP_EVERY:
            return False
        _since_sweep = 0
    # Outside the lock: gc.collect() can run arbitrary __del__ methods, and
    # holding a lock across those risks deadlocking against anything that
    # opens a connection while being finalized.
    gc.collect()
    return True
