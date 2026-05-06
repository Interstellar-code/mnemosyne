# Mnemosyne BEAM Benchmark

**Evaluated against ICLR 2026 BEAM dataset (Tavakoli et al.)**
**Date:** 2026-05-06 | **Version:** Mnemosyne 0.3.2

---

## Status: Preliminary

These results are proof-of-concept only. We ran the official end-to-end BEAM protocol (ingest, retrieve, LLM answer, LLM-as-judge score) on a tiny sample: 1 conversation per scale, 2 questions per ability.

Full-scale evaluation (100 conversations, 2,000+ questions) is pending.

---

## End-to-End Results (LLM-as-Judge)

| Scale | Mnemosyne E2E | Published at 10M |
|-------|--------------|------------------|
| 100K | 26.9% | Hindsight: 64.1% |
| 1M | 19.0% | Honcho: 40.6% |
| | | LIGHT: 26.6% |
| | | RAG: 24.9% |

Published baselines are from Tavakoli et al., ICLR 2026, measured at 10M scale with the identical BEAM protocol. Direct comparison across scales is approximate.

**What this means:**
- At 100K, Mnemosyne scores near RAG-tier (26.9% vs RAG's 24.9% at 10M). Promising but not competitive.
- At 1M, performance drops to 19.0%, below RAG. This indicates the episodic consolidation pipeline is losing information at scale.
- We are far from Hindsight (64.1%) and Honcho (40.6%).

---

## Per-Ability Breakdown (100K, 16 questions)

| Ability | Score | Notes |
|---------|-------|-------|
| ABS (Abstention) | 50% | Can identify some unanswerable questions |
| IE (Info Extraction) | 50% | Extracts specific facts when retrieved correctly |
| TR (Temporal) | 52% | Time-difference questions work reasonably |
| CR (Contradiction) | 37% | Some contradiction detection |
| SUM (Summarization) | 25% | Weak across conversation windows |
| EO (Event Ordering) | 0% | Cannot order events chronologically |
| MR (Multi-hop) | 0% | Cannot connect facts across messages |
| KU (Knowledge Update) | 0% | Cannot track changing values over time |

---

## Known Issues

1. **Episodic consolidation is not producing entries.** The `consolidate_to_episodic()` call in the benchmark ingestion code runs but creates zero episodic entries. This means the retrieval path is missing its primary speed/quality tier.

2. **Sample size is too small.** 16 questions cannot produce statistically meaningful results. Full benchmark requires all 2,000+ questions across all 100 conversations.

3. **No comparison against same-scale baselines.** Published numbers are at 10M only. We need to run Honcho, Hindsight, and RAG at 100K/500K/1M for valid comparison.

4. **LLM-as-judge variance.** With only 2 questions per ability, individual judge calls swing results significantly.

---

## Next Steps

1. Fix episodic consolidation to actually produce entries during benchmark ingestion
2. Run full end-to-end benchmark on all 100 conversations
3. Set up baseline systems (RAG, Honcho) at matching scales for valid comparison
4. Report with confidence intervals (not point estimates)
