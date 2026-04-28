# Planforge: Temporal Query Capabilities for Mnemosyne
# Plan ID: mnemosyne-temporal-queries-v1.0
# Version: 1.0.0
# Status: DRAFT — Pending Abdias Review
# Created: 2026-04-28
# Target Release: Mnemosyne v1.13.0

---

## 1. EXECUTIVE SUMMARY

### 1.1 Problem Statement

Mnemosyne v1.12.0 has a critical temporal memory retrieval gap. The `recall()` function uses hybrid search (FTS5 + vector similarity) but **timestamps are metadata, not searchable content**. This means memories from specific dates (e.g., "what did I do last Monday") are stored in the database but **unfindable via normal recall queries**.

**Impact:**
- 3,177 working memories and 8,890 legacy memories exist with timestamp metadata that cannot be queried by date range
- Users cannot ask time-bounded questions like "show me memories from April 21, 2026"
- The BEAM architecture (working → episodic consolidation) loses temporal context during consolidation because summaries don't preserve date-range metadata
- TripleStore exists as a separate temporal knowledge graph but is **not integrated** with BEAM memory tiers

### 1.2 Goals

1. **Date-range recall**: Enable `recall(from="2026-04-21", to="2026-04-21")` to retrieve memories within a date window
2. **Temporal triples**: Auto-generate `(session, occurred_on, YYYY-MM-DD)` triples for each memory on write
3. **Periodic memory tagging**: Cron-originated memories auto-tagged with `source="cron"` and `topic="CareerOps"`
4. **Backward compatibility**: Existing `recall(query, top_k)` API must work unchanged
5. **BEAM integration**: Temporal queries must span both working_memory and episodic_memory tiers
6. **TripleStore bridge**: Connect temporal triples to BEAM memory lifecycle

---

## 2. TECHNICAL SOLUTIONS

### 2.1 Solution A: Date-Range Filtering in `recall()`

**Implementation:**

Add optional `from_date` and `to_date` parameters to `BeamMemory.recall()` and `Mnemosyne.recall()`:

```python
def recall(self, query: str = None, top_k: int = 5,
           from_date: str = None, to_date: str = None,
           source: str = None, tier: str = None) -> List[Dict]:
```

**Behavior:**
- If only `from_date`/`to_date` provided (no `query`): return all memories in date range, sorted by timestamp DESC
- If `query` + date range provided: apply hybrid scoring first, then filter by timestamp
- Date format: ISO 8601 (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`). Parsed via `datetime.fromisoformat()`
- SQL filter uses indexed `timestamp` column with `BETWEEN` or `>= / <=` operators

**Schema changes:**
- None — `timestamp` column already exists and is indexed (`idx_wm_timestamp`, `idx_em_timestamp`)
- Add composite index for efficient date+source filtering:
  ```sql
  CREATE INDEX IF NOT EXISTS idx_wm_timestamp_source ON working_memory(timestamp, source);
  CREATE INDEX IF NOT EXISTS idx_em_timestamp_source ON episodic_memory(timestamp, source);
  ```

**Backward compatibility:**
- `recall(query, top_k)` signature unchanged — new params are keyword-only with defaults `None`
- Existing code paths unaffected when temporal params not provided

### 2.2 Solution B: Temporal Triples Auto-Generation

**Implementation:**

On every `remember()` call, automatically write a temporal triple to TripleStore:

```python
# In Mnemosyne.remember() and BeamMemory.remember()
from mnemosyne.core.triples import add_triple

# Extract date portion from ISO timestamp
date_str = timestamp[:10]  # "2026-04-21"
add_triple(
    subject=memory_id,           # or session_id
    predicate="occurred_on",
    object=date_str,
    valid_from=timestamp,
    source=source,
    confidence=1.0,
    db_path=self.db_path         # Align with BEAM DB
)
```

**Schema changes:**
- TripleStore currently uses a **separate database** (`triples.db`). For integration, support passing `db_path` to align with BEAM's `mnemosyne.db`
- Add index on `(subject, predicate)` for fast temporal lookups:
  ```sql
  CREATE INDEX IF NOT EXISTS idx_triples_subject_predicate ON triples(subject, predicate);
  ```

**Migration:**
- Backfill script to generate temporal triples for all existing memories (legacy + BEAM)
- Batch processing: 1,000 memories at a time to avoid memory pressure

### 2.3 Solution C: Periodic Memory Tagging

**Implementation:**

Extend `remember()` to accept and preserve `topic` metadata. Cron jobs set explicit tags:

```python
# Cron caller:
mnemosyne.remember(
    content="New job posting: Senior Rust Engineer at StarkWare",
    source="cron",
    metadata={"topic": "CareerOps", "frequency": "daily", "url": "..."}
)
```

**Schema changes:**
- None — `metadata_json` column already stores arbitrary JSON
- Add `source` filtering to `recall()`:
  ```python
  recall(query="StarkWare", source="cron", from_date="2026-04-21")
  ```

**Indexing:**
- `source` is already indexed (`idx_wm_source`, `idx_em_source`)
- Add `topic` extraction helper for querying metadata JSON:
  ```python
  def _topic_filter(topic: str) -> str:
      # SQLite JSON extraction for metadata_json.topic
      return "json_extract(metadata_json, '$.topic') = ?"
  ```

### 2.4 Solution D: BEAM Tier-Aware Temporal Queries

**Implementation:**

When querying with date range, search both tiers and merge results:

```python
def _recall_temporal(self, from_date: str, to_date: str,
                     query: str = None, top_k: int = 5,
                     tier: str = None) -> List[Dict]:
    """
    tier=None → both working + episodic
    tier="working" → working_memory only
    tier="episodic" → episodic_memory only
    """
    results = []
    
    # Build WHERE clause for temporal filter
    temporal_clause = "timestamp >= ? AND timestamp <= ?"
    params = [from_date, to_date]
    
    if tier in (None, "working"):
        # Query working_memory with optional FTS5 + temporal filter
        ...
    
    if tier in (None, "episodic"):
        # Query episodic_memory with optional vec/FTS5 + temporal filter
        ...
    
    # Merge, score, deduplicate, sort
    return results[:top_k]
```

**Consolidation awareness:**
- When working memories are consolidated into episodic summaries, the summary's timestamp should reflect the **latest** timestamp in the group (not the consolidation time)
- This preserves the ability to query "what happened on April 21" even after consolidation

### 2.5 Solution E: TripleStore ↔ BEAM Bridge

**Implementation:**

Create a bridge module `mnemosyne/core/temporal_bridge.py`:

```python
def query_memories_by_date(session_id: str, date: str,
                         db_path: Path = None) -> List[Dict]:
    """
    Use TripleStore temporal triples to find memory IDs,
    then fetch full memory records from BEAM.
    """
    from mnemosyne.core.triples import query_triples
    
    triples = query_triples(
        predicate="occurred_on",
        object=date,
        db_path=db_path
    )
    memory_ids = [t["subject"] for t in triples]
    
    # Fetch from BEAM working + episodic
    return _fetch_memories_by_ids(memory_ids, db_path)
```

**Use case:**
- Alternative query path when exact date matching is needed (vs. range filtering)
- Enables "what happened exactly on 2026-04-21" queries with triple-backed precision

---

## 3. PHASES, TASKS & ACCEPTANCE CRITERIA

### PHASE 1: Design & Specification (Days 1–2)

| Task | Owner | Deliverable | Acceptance Criteria |
|------|-------|-------------|-------------------|
| 1.1 Finalize API signatures | Abdias | Updated docstrings + type hints | All new params have defaults; no positional signature changes |
| 1.2 Index strategy review | Dev | EXPLAIN QUERY PLAN for date-range queries | Temporal queries use index scan (not full table scan) on 10k+ rows |
| 1.3 TripleStore alignment decision | Dev | ADR: shared DB vs separate DB | Decision recorded with trade-offs; migration path defined |
| 1.4 Backfill strategy | Dev | Migration script design | Script can process 12,000 memories without OOM; idempotent |

**Phase 1 Exit Criteria:**
- [ ] Abdias approves API design
- [ ] Index strategy validated on production-sized dataset (12k memories)
- [ ] ADR merged to `.planning/adr/`

---

### PHASE 2: Core Implementation (Days 3–7)

| Task | Owner | File(s) | Acceptance Criteria |
|------|-------|---------|-------------------|
| 2.1 Add temporal params to `BeamMemory.recall()` | Dev | `beam.py` | `recall(from_date="2026-04-21", to_date="2026-04-21")` returns ≥1 result when memories exist on that date |
| 2.2 Add temporal params to `Mnemosyne.recall()` | Dev | `memory.py` | Passes through to `beam.recall()`; legacy table fallback works |
| 2.3 Composite index migration | Dev | `beam.py` | `EXPLAIN QUERY PLAN` shows `USING INDEX idx_wm_timestamp_source` |
| 2.4 Auto-generate temporal triples on `remember()` | Dev | `memory.py`, `beam.py`, `triples.py` | Every `remember()` call creates exactly 1 `(memory_id, occurred_on, YYYY-MM-DD)` triple |
| 2.5 TripleStore DB alignment | Dev | `triples.py` | `TripleStore(db_path=mnemosyne.db_path)` stores triples in same SQLite file as BEAM |
| 2.6 Topic metadata extraction helper | Dev | `beam.py` | `json_extract(metadata_json, '$.topic')` queries work on SQLite ≥3.38 |
| 2.7 Temporal bridge module | Dev | `temporal_bridge.py` | `query_memories_by_date()` returns correct memories within 50ms on 10k dataset |

**Phase 2 Exit Criteria:**
- [ ] All new code paths have unit tests (≥90% coverage for new lines)
- [ ] `pytest tests/test_temporal.py` passes
- [ ] No regression in `pytest tests/test_beam.py`
- [ ] Benchmark: temporal query on 12k memories completes in <100ms (p95)

---

### PHASE 3: Integration & Cron Tagging (Days 8–10)

| Task | Owner | File(s) | Acceptance Criteria |
|------|-------|---------|-------------------|
| 3.1 Cron memory tagging convention | Dev | `memory.py` docs | `source="cron"` + `metadata={"topic": "CareerOps"}` is documented and tested |
| 3.2 `recall(source="cron", from_date=...)` filtering | Dev | `beam.py` | Returns only cron-sourced memories in date range |
| 3.3 Consolidation timestamp preservation | Dev | `beam.py` | Episodic summary timestamp = max(timestamp) of source working memories, not `datetime.now()` |
| 3.4 Backfill script for existing memories | Dev | `scripts/backfill_temporal_triples.py` | Processes 12,000 memories; generates ~12,000 triples; idempotent (re-run safe) |
| 3.5 Legacy memory temporal fallback | Dev | `memory.py` | `recall()` on legacy table supports date filtering when BEAM has no matches |

**Phase 3 Exit Criteria:**
- [ ] Backfill script tested on copy of production DB
- [ ] Cron-tagged memories queryable by date + source + topic
- [ ] Consolidation preserves temporal bounds

---

### PHASE 4: Testing & Validation (Days 11–14)

| Task | Owner | Deliverable | Acceptance Criteria |
|------|-------|-------------|-------------------|
| 4.1 Unit tests: date-range parsing | Dev | `tests/test_temporal.py` | Invalid dates raise `ValueError` with helpful message; edge cases (DST, leap years) covered |
| 4.2 Unit tests: tier-aware queries | Dev | `tests/test_temporal.py` | `tier="working"` returns only working; `tier="episodic"` returns only episodic; `tier=None` merges |
| 4.3 Unit tests: TripleStore bridge | Dev | `tests/test_temporal_bridge.py` | Bridge queries return same results as direct BEAM queries |
| 4.4 Performance benchmark | Dev | `tests/benchmark_temporal.py` | Date-range query p95 <100ms on 12k memories; backfill completes in <5 minutes |
| 4.5 Backward compatibility test | Dev | `tests/test_backward_compat.py` | All existing `recall(query, top_k)` calls return identical results to v1.12.0 |
| 4.6 Integration test: end-to-end | Dev | `tests/test_temporal_e2e.py` | Full flow: remember → temporal triple → recall by date → verify content |

**Phase 4 Exit Criteria:**
- [ ] `pytest tests/` passes with 0 failures
- [ ] Benchmark results recorded in `.planning/benchmarks/temporal-v1.13.md`
- [ ] No API breakage detected by backward compatibility tests

---

### PHASE 5: Documentation (Days 15–16)

| Task | Owner | Deliverable | Acceptance Criteria |
|------|-------|-------------|-------------------|
| 5.1 API documentation update | Dev | `README.md` | `recall()` signature documented with examples for date-range, source, topic |
| 5.2 Temporal query guide | Dev | `docs/temporal-queries.md` | Step-by-step guide with 3+ real-world examples ("last Monday", "career ops this week") |
| 5.3 CHANGELOG entry | Dev | `CHANGELOG.md` | v1.13.0 section lists all temporal features with migration notes |
| 5.4 TripleStore integration doc | Dev | `docs/triples-integration.md` | Explains how temporal triples connect to BEAM; when to use triples vs direct date filter |
| 5.5 Cron tagging convention doc | Dev | `docs/cron-tagging.md` | Standard metadata schema for cron-originated memories |

**Phase 5 Exit Criteria:**
- [ ] All docs reviewed for accuracy against implementation
- [ ] Examples in docs are copy-paste runnable
- [ ] CHANGELOG follows Keep a Changelog format

---

## 4. BACKWARD COMPATIBILITY PLAN

### 4.1 API Compatibility

| Current API | New API | Behavior |
|-------------|---------|----------|
| `recall(query, top_k=5)` | `recall(query, top_k=5)` | **Unchanged** — identical results |
| `recall(query, top_k=5)` | `recall(query, top_k=5, from_date=None)` | `None` = no filter; same as before |
| `remember(content, source="conversation")` | `remember(content, source="conversation", metadata=None)` | Metadata already accepted; no change |

### 4.2 Database Compatibility

- **No breaking schema changes** — only additive (new indexes, new table `triples` in shared DB)
- Existing `mnemosyne.db` files work without migration (new indexes created on init)
- TripleStore backfill is **optional** — new triples generated going forward; old memories queryable via direct date filter on `timestamp` column

### 4.3 Deprecation Policy

- No deprecations in this release
- Legacy `memories` table continues to be dual-written; no removal timeline

---

## 5. INTEGRATION WITH EXISTING ARCHITECTURE

### 5.1 BEAM Tiers

```
┌─────────────────────────────────────────────────────────┐
│                    TEMPORAL QUERY                       │
│         recall(from="2026-04-21", to="2026-04-21")       │
├─────────────────────────────────────────────────────────┤
│  Working Memory (hot)    │  Episodic Memory (long-term) │
│  - idx_wm_timestamp      │  - idx_em_timestamp          │
│  - FTS5 fts_working     │  - FTS5 fts_episodes         │
│  - sqlite-vec (N/A)     │  - sqlite-vec vec_episodes   │
├─────────────────────────────────────────────────────────┤
│              MERGE → DEDUPLICATE → SORT                  │
│              Return top_k by composite score              │
└─────────────────────────────────────────────────────────┘
```

### 5.2 TripleStore Integration

```
┌─────────────────────────────────────────────────────────┐
│  BEAM Memory Tier          │  TripleStore (temporal)    │
│  ──────────────────────────┼──────────────────────────  │
│  working_memory.id         │→ triples.subject          │
│  working_memory.timestamp  │→ triples.valid_from        │
│  "occurred_on"             │→ triples.predicate         │
│  "2026-04-21"              │→ triples.object             │
└─────────────────────────────────────────────────────────┘
```

**Query paths:**
1. **Fast path**: Direct SQL `BETWEEN` on `timestamp` column (no triples needed)
2. **Precise path**: TripleStore `query_triples(predicate="occurred_on", object=date)` for exact date matching
3. **Hybrid path**: TripleStore finds memory IDs → BEAM fetches full records with embeddings

### 5.3 Consolidation (Sleep) Impact

- **Before**: Consolidation summary gets `timestamp = datetime.now()` (loss of temporal context)
- **After**: Consolidation summary gets `timestamp = max(item["timestamp"] for item in group)` (preserves latest date from source memories)
- **Triple preservation**: Individual working memory triples remain; new summary triple added with same date logic

---

## 6. TIMELINE ESTIMATES

| Phase | Duration | Calendar Days | Key Milestone |
|-------|----------|---------------|---------------|
| 1. Design & Spec | 2 days | Mon–Tue | Abdias approval |
| 2. Core Implementation | 5 days | Wed–Sun (1 weekend) | All tests green |
| 3. Integration & Cron | 3 days | Mon–Wed | Backfill complete |
| 4. Testing & Validation | 4 days | Thu–Sun | Benchmarks pass |
| 5. Documentation | 2 days | Mon–Tue | Docs merged |
| **Buffer** | **2 days** | **Wed–Thu** | Contingency |
| **TOTAL** | **18 days** | **~3 weeks** | **Release v1.13.0** |

**Risk-adjusted timeline:**
- If TripleStore DB alignment is complex (shared vs separate): +2 days
- If index performance is poor on 12k rows: +1 day for query optimization
- If backfill reveals data quality issues: +2 days for cleanup

---

## 7. RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| TripleStore DB migration breaks existing triples.db users | Medium | High | Keep `triples.db` as default; shared DB is opt-in via `db_path` param |
| Date-range queries slow on large datasets | Low | Medium | Composite indexes + query plan validation; fallback to pagination |
| Backfill script OOM on 12k memories | Low | Medium | Batch processing (1,000 rows/transaction); streaming JSON parse |
| Cron tagging convention conflicts with existing metadata | Low | Low | Use nested key `metadata["topic"]"; no top-level schema change |
| Consolidation timestamp change breaks existing queries | Medium | Medium | Gate behind env var `MNEMOSYNE_PRESERVE_TEMPORAL=1`; default off for one release |

---

## 8. ACCEPTANCE CRITERIA (Overall)

The plan is approved for implementation when:

- [ ] Abdias has reviewed and commented on this document
- [ ] GitHub issue #<TBD> created with "temporal queries" label
- [ ] Phase 1 tasks completed and design decisions recorded
- [ ] No blocking risks remain in Risk Register

The implementation is complete when:

- [ ] All 5 phases finished with exit criteria met
- [ ] `pytest tests/` passes (0 failures, ≥90% new code coverage)
- [ ] Benchmark: temporal query p95 <100ms on 12k memories
- [ ] Backward compatibility test: v1.12.0 `recall()` results identical
- [ ] Documentation merged and examples verified runnable
- [ ] PR reviewed and approved by Abdias
- [ ] CHANGELOG.md updated for v1.13.0
- [ ] Version bump in `__init__.py` and `pyproject.toml`

---

## 9. APPENDIX

### A.1 Proposed `recall()` Signature (v1.13.0)

```python
def recall(
    self,
    query: str = None,
    top_k: int = 5,
    *,  # keyword-only from here
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    source: Optional[str] = None,
    topic: Optional[str] = None,
    tier: Optional[str] = None,  # "working" | "episodic" | None
    min_importance: Optional[float] = None,
) -> List[Dict]:
    """
    Hybrid memory retrieval with optional temporal filtering.
    
    Args:
        query: Semantic search query. If None, returns all memories filtered by other criteria.
        top_k: Maximum results to return.
        from_date: ISO date string (YYYY-MM-DD) — inclusive lower bound.
        to_date: ISO date string (YYYY-MM-DD) — inclusive upper bound.
        source: Filter by memory source (e.g., "cron", "conversation").
        topic: Filter by metadata topic (e.g., "CareerOps").
        tier: "working", "episodic", or None (both).
        min_importance: Minimum importance score (0.0–1.0).
    
    Returns:
        List of memory dicts with keys: id, content, source, timestamp, tier, score, ...
    
    Examples:
        >>> mnemosyne.recall("StarkWare", from_date="2026-04-21", to_date="2026-04-21")
        >>> mnemosyne.recall(from_date="2026-04-01", to_date="2026-04-30", source="cron")
        >>> mnemosyne.recall("career", topic="CareerOps", top_k=10)
    """
```

### A.2 Database Index Additions

```sql
-- Phase 2.3: Composite indexes for temporal + source queries
CREATE INDEX IF NOT EXISTS idx_wm_timestamp_source ON working_memory(timestamp, source);
CREATE INDEX IF NOT EXISTS idx_em_timestamp_source ON episodic_memory(timestamp, source);

-- Phase 2.5: TripleStore alignment
CREATE INDEX IF NOT EXISTS idx_triples_subject_predicate ON triples(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_triples_predicate_object ON triples(predicate, object);
```

### A.3 Backfill Script Outline

```python
# scripts/backfill_temporal_triples.py
"""One-time backfill of temporal triples for existing memories."""

from mnemosyne.core.beam import _get_connection
from mnemosyne.core.triples import add_triple
from pathlib import Path

BATCH_SIZE = 1000

def backfill(db_path: Path = None):
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    for table in ("working_memory", "episodic_memory", "memories"):
        cursor.execute(f"SELECT id, timestamp, source FROM {table}")
        batch = []
        for row in cursor:
            date_str = row["timestamp"][:10] if row["timestamp"] else None
            if date_str:
                batch.append((row["id"], "occurred_on", date_str, row["timestamp"], row["source"]))
            if len(batch) >= BATCH_SIZE:
                _insert_batch(batch, db_path)
                batch = []
        if batch:
            _insert_batch(batch, db_path)
    
    print("Backfill complete.")

def _insert_batch(batch, db_path):
    for subject, predicate, obj, valid_from, source in batch:
        add_triple(subject, predicate, obj, valid_from=valid_from, source=source, db_path=db_path)

if __name__ == "__main__":
    backfill()
```

### A.4 Test Cases (Phase 4)

```python
# tests/test_temporal.py — key test cases

def test_recall_date_range_working_memory():
    """Store 3 memories on different dates; query 1 date; expect 1 result."""
    
def test_recall_date_range_episodic_memory():
    """Consolidate memories; query by date; expect summary in results."""
    
def test_recall_date_range_no_query():
    """recall(from_date=X, to_date=Y) with no query returns all in range."""
    
def test_recall_source_filter():
    """recall(source="cron") returns only cron memories."""
    
def test_recall_topic_filter():
    """recall(topic="CareerOps") filters by metadata topic."""
    
def test_recall_backward_compat():
    """recall(query, top_k) returns identical results to v1.12.0."""
    
def test_temporal_triple_auto_generated():
    """remember() creates exactly 1 occurred_on triple."""
    
def test_backfill_idempotent():
    """Running backfill twice doesn't create duplicate triples."""
    
def test_consolidation_preserves_temporal_bounds():
    """Sleep consolidation sets summary timestamp = max(source timestamps)."""
```

---

## 10. SIGN-OFF

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Product Owner | Abdias J | _______________ | _______ |
| Technical Lead | <TBD> | _______________ | _______ |
| Implementer | <TBD> | _______________ | _______ |

---

*This plan was generated by Planforge (Hermes Agent) on 2026-04-28. Version 1.0.0 — DRAFT.*
