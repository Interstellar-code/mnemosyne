"""
Tests for Mnemosyne CanonicalStore (issue #256).

CanonicalStore is the owner-scoped single-source-of-truth layer: each
(owner_id, category, name) slot holds exactly one current value, restating a
value is a no-op, and a new value supersedes the old one (kept as history).

These tests pin the contract:
- Upsert-in-place: identical body is a no-op; new body bumps version + supersedes.
- Exactly one current row per slot (partial unique index enforcement).
- Owner isolation: one owner never sees/touches another's slots.
- History, list, and substring search read paths.
- Export/import round-trip parity with the sibling stores.
- BeamMemory wires `self.canonical` on the shared connection.
- The Hermes provider tools round-trip and derive owner from the profile.
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ImportError until CanonicalStore lands — red against main, green after.
from mnemosyne.core.canonical import (
    CanonicalStore,
    init_canonical,
    remember_canonical,
    recall_canonical,
)


class _TempDBTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        self.store = CanonicalStore(db_path=self.db_path)

    def tearDown(self):
        try:
            self.store.conn.close()
        except Exception:
            pass
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass


class TestUpsertSemantics(_TempDBTest):
    def test_create_then_recall(self):
        row = self.store.remember("jessi", "identity", "name", "My name is Jessi.")
        self.assertEqual(row["status"], "created")
        self.assertEqual(row["version"], 1)
        got = self.store.recall("jessi", "identity", "name")
        self.assertEqual(got["body"], "My name is Jessi.")
        self.assertIsNone(got["valid_until"])

    def test_identical_body_is_noop(self):
        self.store.remember("jessi", "identity", "name", "My name is Jessi.")
        row = self.store.remember("jessi", "identity", "name", "My name is Jessi.")
        self.assertEqual(row["status"], "unchanged")
        self.assertEqual(row["version"], 1)
        # Exactly one row total — no duplicate accumulated.
        self.assertEqual(len(self.store.history("jessi", "identity", "name")), 1)

    def test_new_body_supersedes_and_versions(self):
        self.store.remember("jessi", "identity", "name", "My name is Jessi.")
        row = self.store.remember("jessi", "identity", "name", "I go by Jess now.")
        self.assertEqual(row["status"], "updated")
        self.assertEqual(row["version"], 2)
        # Current value is the new one.
        self.assertEqual(self.store.recall("jessi", "identity", "name")["body"],
                         "I go by Jess now.")
        hist = self.store.history("jessi", "identity", "name")
        self.assertEqual([h["body"] for h in hist],
                         ["I go by Jess now.", "My name is Jessi."])
        # The superseded row is closed; the current one is open.
        self.assertIsNone(hist[0]["valid_until"])
        self.assertIsNotNone(hist[1]["valid_until"])

    def test_exactly_one_current_row_invariant(self):
        """The partial unique index permits only one live row per slot."""
        for body in ("a body value", "b body value", "c body value"):
            self.store.remember("jessi", "identity", "name", body)
        cur = self.store.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts "
            "WHERE owner_id=? AND category=? AND name=? AND valid_until IS NULL",
            ("jessi", "identity", "name"),
        ).fetchone()[0]
        self.assertEqual(cur, 1)
        # History retains every version.
        self.assertEqual(len(self.store.history("jessi", "identity", "name")), 3)

    def test_direct_duplicate_live_row_rejected(self):
        """Defense-in-depth: a hand-inserted second live row violates the index."""
        self.store.remember("jessi", "identity", "name", "first body value")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.conn.execute(
                "INSERT INTO canonical_facts "
                "(owner_id, category, name, body, version, valid_from, valid_until) "
                "VALUES (?,?,?,?,?,?,NULL)",
                ("jessi", "identity", "name", "sneaky dup", 99, "2026-01-01"),
            )
            self.store.conn.commit()
        self.store.conn.rollback()

    def test_blank_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.store.remember("jessi", "identity", "name", "   ")
        with self.assertRaises(ValueError):
            self.store.remember("jessi", "", "name", "body value here")


class TestForget(_TempDBTest):
    def test_forget_retires_current_keeps_history(self):
        self.store.remember("jessi", "identity", "name", "My name is Jessi.")
        self.assertTrue(self.store.forget("jessi", "identity", "name"))
        self.assertIsNone(self.store.recall("jessi", "identity", "name"))
        # Row preserved as history (auditable), not deleted.
        self.assertEqual(len(self.store.history("jessi", "identity", "name")), 1)

    def test_forget_empty_slot_returns_false(self):
        self.assertFalse(self.store.forget("jessi", "identity", "ghost"))

    def test_remember_after_forget_starts_fresh_current(self):
        self.store.remember("jessi", "identity", "name", "My name is Jessi.")
        self.store.forget("jessi", "identity", "name")
        row = self.store.remember("jessi", "identity", "name", "Reborn as Jess.")
        self.assertEqual(self.store.recall("jessi", "identity", "name")["body"],
                         "Reborn as Jess.")
        # version keeps climbing across the gap.
        self.assertEqual(row["version"], 2)


class TestOwnerIsolation(_TempDBTest):
    def test_same_slot_different_owners_coexist(self):
        self.store.remember("jessi", "identity", "name", "I am Jessi.")
        self.store.remember("atlas", "identity", "name", "I am Atlas.")
        self.assertEqual(self.store.recall("jessi", "identity", "name")["body"], "I am Jessi.")
        self.assertEqual(self.store.recall("atlas", "identity", "name")["body"], "I am Atlas.")

    def test_list_is_owner_scoped(self):
        self.store.remember("jessi", "identity", "name", "I am Jessi.")
        self.store.remember("atlas", "identity", "name", "I am Atlas.")
        jessi = self.store.list("jessi")
        self.assertEqual(len(jessi), 1)
        self.assertEqual(jessi[0]["owner_id"], "jessi")

    def test_search_is_owner_scoped(self):
        self.store.remember("jessi", "identity", "name", "I am Jessi the cartographer.")
        self.store.remember("atlas", "identity", "name", "I am Atlas the cartographer.")
        hits = self.store.search("jessi", "cartographer")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["owner_id"], "jessi")


class TestListAndSearch(_TempDBTest):
    def test_list_all_and_by_category(self):
        self.store.remember("jessi", "identity", "name", "I am Jessi.")
        self.store.remember("jessi", "identity", "pronouns", "she/her")
        self.store.remember("jessi", "voice", "register", "warm and direct")
        self.assertEqual(len(self.store.list("jessi")), 3)
        ident = self.store.list("jessi", category="identity")
        self.assertEqual({r["name"] for r in ident}, {"name", "pronouns"})

    def test_search_matches_body_name_category(self):
        self.store.remember("jessi", "voice", "register", "warm and direct")
        self.assertTrue(self.store.search("jessi", "warm"))       # body
        self.assertTrue(self.store.search("jessi", "register"))   # name
        self.assertTrue(self.store.search("jessi", "voice"))      # category
        self.assertEqual(self.store.search("jessi", "nonexistent"), [])

    def test_search_excludes_superseded(self):
        self.store.remember("jessi", "voice", "register", "icy and terse")
        self.store.remember("jessi", "voice", "register", "warm and direct")
        # Old value must not surface in search.
        self.assertEqual(self.store.search("jessi", "icy"), [])
        self.assertTrue(self.store.search("jessi", "warm"))

    def test_search_blank_query_returns_empty(self):
        self.store.remember("jessi", "voice", "register", "warm")
        self.assertEqual(self.store.search("jessi", "   "), [])


class TestExportImport(_TempDBTest):
    def test_round_trip(self):
        self.store.remember("jessi", "identity", "name", "I am Jessi.")
        self.store.remember("jessi", "identity", "name", "I am Jess.")  # creates history
        self.store.remember("jessi", "voice", "register", "warm")
        exported = self.store.export_all()
        self.assertEqual(len(exported), 3)

        # Fresh DB, import.
        other = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        other.close()
        try:
            dst = CanonicalStore(db_path=Path(other.name))
            stats = dst.import_all(exported)
            self.assertEqual(stats["inserted"], 3)
            self.assertEqual(sum(stats.values()), 3)
            self.assertEqual(dst.recall("jessi", "identity", "name")["body"], "I am Jess.")
            self.assertEqual(len(dst.history("jessi", "identity", "name")), 2)
            dst.conn.close()
        finally:
            os.unlink(other.name)

    def test_idempotent_reimport_skips(self):
        self.store.remember("jessi", "identity", "name", "I am Jessi.")
        exported = self.store.export_all()
        stats = self.store.import_all(exported)
        # Re-importing the same export into the same DB is a no-op.
        self.assertEqual(stats["skipped"], len(exported))


class TestModuleConvenience(_TempDBTest):
    def test_module_level_functions(self):
        remember_canonical("jessi", "identity", "name", "I am Jessi.", db_path=self.db_path)
        got = recall_canonical("jessi", "identity", "name", db_path=self.db_path)
        self.assertEqual(got["body"], "I am Jessi.")

    def test_init_is_idempotent(self):
        init_canonical(self.db_path)
        init_canonical(self.db_path)  # no error on second call


class TestBeamWiring(unittest.TestCase):
    """CanonicalStore is reachable as beam.canonical on the shared connection."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_beam_exposes_canonical_sharing_connection(self):
        from mnemosyne.core.beam import BeamMemory
        beam = BeamMemory(session_id="t", db_path=self.db_path)
        try:
            self.assertTrue(hasattr(beam, "canonical"))
            # Same connection object — no extra file descriptor.
            self.assertIs(beam.canonical.conn, beam.conn)
            row = beam.canonical.remember("jessi", "identity", "name", "I am Jessi.")
            self.assertEqual(row["status"], "created")
            self.assertEqual(
                beam.canonical.recall("jessi", "identity", "name")["body"],
                "I am Jessi.",
            )
        finally:
            try:
                beam.conn.close()
            except Exception:
                pass


class TestProviderTools(unittest.TestCase):
    """The two Hermes provider tools round-trip and isolate by profile."""

    def setUp(self):
        from hermes_memory_provider import MnemosyneMemoryProvider
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        from mnemosyne.core.beam import BeamMemory
        self.provider = MnemosyneMemoryProvider()
        self.provider._beam = BeamMemory(session_id="t", db_path=self.db_path)
        self.provider._agent_identity = "jessi"

    def tearDown(self):
        try:
            self.provider._beam.conn.close()
        except Exception:
            pass
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_remember_and_recall_roundtrip(self):
        out = json.loads(self.provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {"category": "identity", "name": "name", "body": "My name is Jessi."},
        ))
        self.assertEqual(out["status"], "created")
        self.assertEqual(out["owner_id"], "jessi")

        got = json.loads(self.provider.handle_tool_call(
            "mnemosyne_recall_canonical",
            {"category": "identity", "name": "name"},
        ))
        self.assertEqual(got["mode"], "recall")
        self.assertTrue(got["found"])
        self.assertEqual(got["result"]["body"], "My name is Jessi.")

    def test_recall_search_mode(self):
        self.provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {"category": "voice", "name": "register", "body": "warm and direct"},
        )
        got = json.loads(self.provider.handle_tool_call(
            "mnemosyne_recall_canonical", {"query": "warm"},
        ))
        self.assertEqual(got["mode"], "search")
        self.assertEqual(got["count"], 1)

    def test_recall_history_mode(self):
        for body in ("icy and terse", "warm and direct"):
            self.provider.handle_tool_call(
                "mnemosyne_remember_canonical",
                {"category": "voice", "name": "register", "body": body},
            )
        got = json.loads(self.provider.handle_tool_call(
            "mnemosyne_recall_canonical",
            {"category": "voice", "name": "register", "include_history": True},
        ))
        self.assertEqual(got["mode"], "history")
        self.assertEqual(got["count"], 2)

    def test_owner_defaults_and_isolates_by_profile(self):
        self.provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {"category": "identity", "name": "name", "body": "I am Jessi."},
        )
        # Switch profile → different owner bank, nothing visible.
        self.provider._agent_identity = "atlas"
        got = json.loads(self.provider.handle_tool_call(
            "mnemosyne_recall_canonical", {"category": "identity", "name": "name"},
        ))
        self.assertEqual(got["owner_id"], "atlas")
        self.assertFalse(got["found"])

    def test_missing_required_fields(self):
        out = json.loads(self.provider.handle_tool_call(
            "mnemosyne_remember_canonical", {"category": "identity", "name": "name"},
        ))
        self.assertIn("error", out)

    def test_schemas_exposed(self):
        names = {s["name"] for s in self.provider.get_tool_schemas()}
        self.assertIn("mnemosyne_remember_canonical", names)
        self.assertIn("mnemosyne_recall_canonical", names)


if __name__ == "__main__":
    unittest.main()
