"""
Mnemosyne Structured Fact Extraction
====================================
LLM-driven fact extraction as a derived layer.
Extracts 2-5 concise factual statements from raw text.
Facts are stored as TripleStore triples, not replacements for raw text.

Uses the same LLM fallback chain as local_llm.py:

0. Host-provided LLM backend (when MNEMOSYNE_HOST_LLM_ENABLED=true and a
   backend is registered). On host attempt with no usable output, skips
   the remote URL and goes straight to local GGUF.
1. Remote OpenAI-compatible API (if MNEMOSYNE_LLM_BASE_URL set
   AND MNEMOSYNE_LLM_ENABLED is not false).
2. Local ctransformers GGUF model.
3. Skip extraction (graceful degradation).

Extraction uses temperature=0.0 (deterministic) so re-ingesting the same
content does not create near-duplicate facts in the facts table.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Reuse local_llm infrastructure
from mnemosyne.core import local_llm
from mnemosyne.core.local_llm import (
    llm_available,
    _call_remote_llm,
    _load_llm,
    _try_host_llm,
    LLM_BASE_URL,
    LLM_ENABLED,
    LLM_MAX_TOKENS,
    _clean_output,
)

# --- Config ------------------------------------------------------------------
EXTRACTION_PROMPT = os.environ.get(
    "MNEMOSYNE_EXTRACTION_PROMPT",
    "Extract 2-5 concise factual statements from the following text. "
    "Each fact should be a complete sentence describing something true about the subject. "
    "Focus on preferences, opinions, experiences, and factual claims. "
    "Return one fact per line. Do not number them. "
    "If no facts can be extracted, return 'NO_FACTS'.\n\nText: {text}\n\nFacts:"
)


def _build_extraction_prompt(text: str) -> str:
    """Build the extraction prompt with the user text inserted."""
    return EXTRACTION_PROMPT.format(text=text)


def _parse_facts(raw_output: str) -> List[str]:
    """Parse LLM output into individual facts."""
    if not raw_output or raw_output.strip().upper() == "NO_FACTS":
        return []
    
    # Split on newlines, filter empty lines
    lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
    
    # Clean up any numbering or bullet prefixes
    cleaned = []
    for line in lines:
        # Remove leading numbers/bullets: "1. fact" or "- fact" or "* fact"
        line = line.lstrip("0123456789.-* ").strip()
        if line and len(line) > 10:  # Minimum fact length
            cleaned.append(line)
    
    return cleaned[:5]  # Cap at 5 facts


def extract_facts(text: str) -> List[str]:
    """
    Extract structured facts from raw text using LLM.

    Args:
        text: Raw memory content to extract facts from

    Returns:
        List of extracted fact strings (0-5 items). Empty list if LLM unavailable.

    Notes:
        - The host backend (Hermes auxiliary client) is consulted first when
          enabled. Temperature is fixed at 0.0 so re-ingesting the same content
          produces deterministic facts (avoids near-duplicate writes to the
          facts table).
        - When the host attempt produces no usable text, the remote URL is
          **skipped** — falls through to local GGUF, then []. This honors the
          plan's host-vs-remote precedence rule.
        - [C13.b] All tier transitions and failures are recorded to the
          process-global `ExtractionDiagnostics`. Operators query via
          `mnemosyne.extraction.get_extraction_stats()` to see why
          extraction is producing empty results.
    """
    # Lazy import to avoid pulling the diagnostics chain at module load
    # (extraction.py is imported very early in some test contexts).
    from mnemosyne.extraction.diagnostics import get_diagnostics
    diag = get_diagnostics()

    if not text or not text.strip():
        # Caller passed nothing — this isn't a failure, just no work.
        # Don't record_call: this isn't really an extraction attempt.
        return []

    if not local_llm.llm_available():
        diag.record_failure(
            "local", reason="llm_unavailable_at_call_site",
        )
        diag.record_call(succeeded=False, all_empty=False)
        return []

    prompt = _build_extraction_prompt(text)

    # 0. Host backend (deterministic; temperature=0.0).
    # Reference live module values so monkeypatch on local_llm reaches us.
    diag.record_attempt("host")
    attempted, host_text = local_llm._try_host_llm(
        prompt, max_tokens=local_llm.LLM_MAX_TOKENS, temperature=0.0
    )
    if attempted:
        if host_text:
            facts = _parse_facts(host_text)
            if facts:
                diag.record_success("host", fact_count=len(facts))
                diag.record_call(succeeded=True)
                return facts
            diag.record_no_output("host")
        else:
            diag.record_no_output("host")
        # Host attempted but produced no facts. Skip remote per A3; try local.
        diag.record_attempt("local")
        llm = local_llm._load_llm()
        if llm is not None:
            try:
                raw_output = llm(
                    prompt,
                    max_new_tokens=local_llm.LLM_MAX_TOKENS,
                    stop=["</s>", "<|user|>"],
                )
                facts = _parse_facts(local_llm._clean_output(raw_output))
                if facts:
                    diag.record_success("local", fact_count=len(facts))
                    diag.record_call(succeeded=True)
                else:
                    diag.record_no_output("local")
                    diag.record_call(succeeded=False, all_empty=True)
                return facts
            except Exception as e:
                diag.record_failure("local", exc=e, reason="ctransformers_raised")
                logger.warning(
                    "extract_facts: local LLM raised on host-fallback path: %r", e
                )
                diag.record_call(succeeded=False)
                return []
        diag.record_failure("local", reason="model_not_loaded")
        diag.record_call(succeeded=False, all_empty=True)
        return []

    # 1. Remote LLM. Pass temperature=0.0 so the C2 determinism contract
    # holds even on the standalone remote path (where extract_facts shares
    # _call_remote_llm with summarize_memories' default of 0.3).
    if local_llm.LLM_ENABLED and local_llm.LLM_BASE_URL:
        diag.record_attempt("remote")
        raw_output = local_llm._call_remote_llm(prompt, temperature=0.0)
        if raw_output:
            facts = _parse_facts(local_llm._clean_output(raw_output))
            if facts:
                diag.record_success("remote", fact_count=len(facts))
                diag.record_call(succeeded=True)
                return facts
            diag.record_no_output("remote")
        else:
            diag.record_no_output("remote")

    # 2. Local LLM.
    diag.record_attempt("local")
    llm = local_llm._load_llm()
    if llm is not None:
        try:
            raw_output = llm(
                prompt,
                max_new_tokens=local_llm.LLM_MAX_TOKENS,
                stop=["</s>", "<|user|>"],
            )
            facts = _parse_facts(local_llm._clean_output(raw_output))
            if facts:
                diag.record_success("local", fact_count=len(facts))
                diag.record_call(succeeded=True)
            else:
                diag.record_no_output("local")
                diag.record_call(succeeded=False, all_empty=True)
            return facts
        except Exception as e:
            diag.record_failure("local", exc=e, reason="ctransformers_raised")
            logger.warning(
                "extract_facts: local LLM raised: %r", e
            )
            diag.record_call(succeeded=False)
            return []

    diag.record_failure("local", reason="model_not_loaded")
    diag.record_call(succeeded=False, all_empty=True)
    return []


def extract_facts_safe(text: str) -> List[str]:
    """
    Best-effort fact extraction that never raises.
    Wrapper for extract_facts with exception handling.

    [C13.b] Outer-wrapper failures (anything `extract_facts` lets
    escape) are recorded as `local` tier failures with reason
    `outer_wrapper_caught` so operators can see what's slipping
    past the inner instrumentation.
    """
    try:
        return extract_facts(text)
    except Exception as e:
        from mnemosyne.extraction.diagnostics import get_diagnostics
        diag = get_diagnostics()
        diag.record_failure(
            "local", exc=e, reason="outer_wrapper_caught"
        )
        diag.record_call(succeeded=False)
        logger.warning(
            "extract_facts_safe: extract_facts() raised: %r", e
        )
        return []
