## [2.6] — 2026-05-11
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Simple Versioning](https://github.com/AxDSan/mnemosyne) (MAJOR.MINOR).

## [Unreleased]

### Changed

**E1 — BEAM benchmark adapter uses the real ingest pipeline (additive)**
- `tools/evaluate_beam_end_to_end.py::ingest_conversation` no longer runs a destructive `Batch N: first_3_msg[:100chars]` template summary + DELETE per batch. Per-batch the adapter now backdates the just-inserted `working_memory` rows past sleep's TTL/2 cutoff, then calls `beam.sleep()` — which (post-E3) produces real LLM-generated (or AAAK-fallback) summaries on top of preserved originals.
- Net effect: ~99% of message content was previously discarded at ingest time, leaving every experiment arm running on a corpus of ~500 episodic rows of mostly-empty template strings instead of the actual benchmark messages. Post-E1 the corpus is preserved in `working_memory` AND consolidated summaries land in `episodic_memory` — recall reaches actual content. Unblocks every BEAM-recovery experiment arm.
- Behavior change for benchmark stats output: `stats["wm_count"]` now grows monotonically with input message count (pre-E1 it was always 0 after each batch because the destructive consolidation deleted everything). This is the contract the experiment actually wants to measure.
- **Stacks on E3** (PR #73): depends on additive sleep being merged first; without E3, `beam.sleep()` would still DELETE source rows and the fix would be moot.

**E3 — Additive sleep (kill summarize-and-delete)**
- `BeamMemory.sleep()` no longer DELETEs source `working_memory` rows after writing the consolidated summary to `episodic_memory`. Originals are marked with a new `consolidated_at` timestamp and remain queryable through recall.
- Maintainer decision (2026-05-10): "Originals stay. Summaries become enrichment on top. Storage cost is fine — it's the lowest-cost tradeoff." Unblocks experiment Arm B (ADD-only ingest) of the BEAM-recovery experiment.
- No feature flag — additive is the only mode going forward.
- `BeamMemory._trim_working_memory` exempts rows with `consolidated_at IS NOT NULL`. The TTL/MAX_ITEMS sliding window only bounds NOT-YET-consolidated content; consolidated originals live until explicit `forget()`. Strict reading of the "originals stay" contract.
- `BeamMemory.remember`'s dedup-update path clears `consolidated_at = NULL` when re-asserting an already-consolidated row, so the refreshed occurrence becomes eligible for sleep again.
- `sleep()` now uses an atomic claim: marks `consolidated_at` BEFORE writing the episodic summary, gated on `consolidated_at IS STILL NULL`. Concurrent sleep callers see `rowcount=0` and bail; crash-after-claim leaves an orphan marker but no phantom summary.
- The `wm_au` FTS trigger now fires `AFTER UPDATE OF content`, not on every UPDATE. Sleep's marker writes don't churn the FTS index — important now that sleep is high-volume.

### Added
- **E5 PolyphonicRecallEngine** — Multi-voice recall fusion (vector, semantic, temporal, veracity) weighted by a tunable cross-voice scoring matrix
  - Feature-gated: set `MNEMOSYNE_POLYPHONIC_RECALL=1` to enable; disabled by default
  - Weight matrix baked into `polyphonic_recall.py`; no extra deps, no ML model behind a flag
  - Falls back to standard recall path when disabled; feature-flag gates the import, not the call site

- **C4 recall path provenance diagnostics** — Callers can inspect which recall paths returned each result
  - `recall(source_breakdown=True)` returns per-voice counts and vector coverage stats
  - New `recall_diagnostics.py` module with `compute_recall_path_coverage()` helper
  - Backward compatible: existing `recall()` calls are unaffected (default: no breakdown)

- **C13.b fact-extraction failure diagnostics** — Per-extraction failure counters and sentiment-bias detection
  - Each extraction attempt now records `attempts`, `failures`, and `last_error` counters
  - New `mnemosyne/extraction/diagnostics.py` with `analyze_extraction_health()` and `detect_sentiment_bias()`
  - Extraction pipeline no longer silently swallows repeated failures

### Fixed
- **C25 + docs: DeltaSync table/column allowlist guards schema drift**
  - Previously DeltaSync blindly mirrored all tables/columns; any upstream schema drift would replicate into beam
  - Now uses a hardcoded allowlist of `table → [columns]` for working_memory, episodic_memory, scratchpad, knowledge_graph, and memory_embeddings
  - Snapshot DBs or upstream changes adding columns (e.g., `wm.x_new_field`) are silently dropped on the receiving side
  - Requires no server-side changes (DeltaSync wire protocol unchanged); only the receiving end enforces the allowlist
  - See `docs/api-reference.md` (expanded DeltaSync section) and migration notes in `UPDATING.md`

- **E3 additive sleep**
  - `sleep()` no longer **deletes** consolidated working memory rows — it marks them with `consolidated_at`
  - Originals remain queryable and searchable alongside the new episodic summary
  - Backward compatible: pre-E3 DBs gain the column automatically at startup; old rows are backfilled to skip re-summarization
  - Working memory trim now exempts already-consolidated rows so the "originals stay" contract survives TTL window expiry
  - Re-remembering a consolidated row clears `consolidated_at` so it re-enters the sleep queue (avoids permanent skip)

- **PII-safe diagnose —fix flag**
  - `mnemosyne diagnose --fix` auto-installs missing dependencies (fastembed, sqlite-vec, numpy, huggingface_hub)
  - `mnemosyne doctor` alias for quick access
  - `--dry-run` preview mode shows what would be installed without downloading anything

- **deploy_hermes_provider.sh fix for curl | bash**
  - No more `BASH_SOURCE[0]: unbound variable` error
  - Piped install now auto-clones the repo to `~/.hermes/mnemosyne-repo/`
  - Backward compatible: local clones still work as before

### Upgraded
- **E4 veracity threading** — working-memory recall multiplier now integrates per-row veracity scores
  - `remember_batch()` propagates veracity into each working memory insertion
  - The `recall()` path applies a veracity multiplier (`wm_vm`) to working-memory tier results, so trusted rows rank higher

- **E1 benchmark adapter** — ICLR 2026 BEAM benchmark now runs the real ingest pipeline instead of a toy adapter
  - No more skipped transforms: benchmarks exercise the same code path as production
  - Stacks on PR #73 (additive sleep) for backfill compatibility
  - `tools/evaluate_beam_end_to_end.py` upgraded with full-pipeline harness

- **/review hardening**
  - Post-filter counters: `kept-not-scanned`, `post_filter_removed`, rate clamp diagnostics
  - `kept_not_scanned` tracks vector-only hits pulled from pool but never score-tested (valuable coverage metric)

## [2.5] — 2026-05-10

### Added
- Working memory TTL: `mnemosyne update <id> --ttl 7d` expires rows after the given interval
- Dedicated WHL builds in the public release asset pipeline (CI matrix + install smoke test)
- `heal_quality` pipeline: LLM-judge based quality assessment for episodic summaries (factual density, format compliance, grounding)
- `mnemosyne_invalidate` memory tool + forget cascade (deletes source reference + down-weighted/fuzzy matches)
- ICLR 2026 BEAM SOTA benchmark harness (compare Mnemosyne against Honcho, Hindsight, LIGHT, Mem0, LangMem, Zep, Memobase)
- SOTA results page on docs site with BEAM tier badges

### Changed
- Episodic consolidation quality significantly improved (LLM judge enforces factual density ≥100B, strict grounding)
- Dimensionality-aware vector comparison (binary uses hamming, float uses cosine)
- Episodic memory recall correctly handles `valid_until` and `superseded_by` filters
- Memory embeddings: `memory_id` uniqueness constraint prevents double-registration
- BEAM benchmark now runs the full ingest pipeline instead of a simple adapter

### Fixed
- Export `--from` semantics: BEAM export correctly uses `export_from` path with detached-db copy
- RRF (Reciprocal Rank Fusion) hybrid scoring with k=60 fuses 4 retrieval voices
- Ghost session bug: `sleep_all_sessions()` now correctly discovers sessions containing NULL `session_id` rows
- `--version` flag now shows the correct package version string
