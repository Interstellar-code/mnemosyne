# Project State

**Updated:** 2026-05-05
**Current Phase:** 1 — Core Degradation Engine
**Phase Status:** ✅ Complete (shipped to main)

## Progress

| Phase | Status | Started | Ship Date |
|-------|--------|---------|-----------|
| 1 | ✅ Complete | 2026-05-05 | 2026-05-05 |
| 2 | Planned | - | - |
| 3 | Planned | - | - |

## Implementation Summary

### Completed Waves
- ✅ Wave 1: Schema migration (tier, degraded_at columns + backfill)
- ✅ Wave 2: Config constants (TIER2_DAYS, TIER3_DAYS, TIER*_WEIGHT, DEGRADE_BATCH_SIZE)
- ✅ Wave 3: degrade_episodic() function (LLM compression tier 1→2, text extraction tier 2→3)
- ✅ Wave 4: Tier multiplier in recall scoring (post-processing before sort)
- ✅ Wave 5: Sleep integration (degrade called in both sleep() and sleep_all_sessions())
- ✅ Wave 6: Tests (10 tests: schema, transitions, dry run, batch limit, weighting, sleep integration, end-to-end recall)

### Bug Fixes
- 🐛 Fixed `local_llm.summarize()` → `local_llm.summarize_memories()` (wrong function name, would crash on LLM path)
- 🐛 Fixed SQLite connection conflicts in batch test

### Files Changed
- `mnemosyne/core/beam.py` (+123 lines: config, schema, degrade_episodic, recall weighting, sleep integration)
- `tests/test_beam.py` (+262 lines: 10 new tests in TestTieredDegradation class)

### Commits
- `8ca39cd` — feat: tiered episodic degradation (Waves 1-5)
- *(pending)* — fix: summarize_memories call + Wave 6 tests

### Blockers
None.
