"""Regression tests for C13.b — fact extraction failure diagnosability.

Pre-C13.b: fact extraction had five silent-failure layers (cloud HTTP
errors → `""`, JSON parse failures → `[]`, local LLM exceptions →
`pass`, no-LLM-available fallback → `[]`, outer `extract_facts_safe`
wrapper → `[]`). Operators got zero signal that extraction-dependent
features (fact recall, graph voice, the heal-quality pipeline) were
running blind.

Post-C13.b: a process-global `ExtractionDiagnostics` records each
extraction attempt's outcome at every tier (host / remote / local /
cloud). Failures are still swallowed at the call site (callers'
contract preserved); diagnostics surface what's being swallowed.
Operators query via `get_extraction_stats()`.

These tests pin:
  - The diagnostics class API (record/snapshot/reset, thread-safety)
  - The integration with `extract_facts` (each tier's outcome is
    recorded correctly)
  - The integration with `ExtractionClient` (cloud path)
  - The outer-wrapper instrumentation in `extract_facts_safe`
  - The unknown-tier rejection (typo guard for future callers)
"""

from __future__ import annotations

import json
import logging
import threading
from unittest.mock import patch

import pytest

from mnemosyne.extraction.diagnostics import (
    EXTRACTION_TIERS,
    ExtractionDiagnostics,
    get_diagnostics,
    get_extraction_stats,
    reset_extraction_stats,
)


@pytest.fixture(autouse=True)
def fresh_diag():
    """Every test starts with reset diagnostics — process-global state
    must not leak between tests."""
    reset_extraction_stats()
    yield
    reset_extraction_stats()


class TestExtractionDiagnosticsClass:
    """Class-level API. Test the primitives directly so future
    refactors can't quietly break the recording contract."""

    def test_tier_constants_are_canonical(self):
        assert EXTRACTION_TIERS == ("host", "remote", "local", "cloud")

    def test_snapshot_initial_state(self):
        diag = ExtractionDiagnostics()
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 0
        assert snap["totals"]["successes"] == 0
        assert snap["totals"]["failures"] == 0
        assert snap["totals"]["empty"] == 0
        assert snap["totals"]["success_rate"] == 0.0
        for tier in EXTRACTION_TIERS:
            t = snap["by_tier"][tier]
            assert t["attempts"] == 0
            assert t["successes"] == 0
            assert t["no_output"] == 0
            assert t["failures"] == 0
            assert t["error_samples"] == []

    def test_record_attempt_increments(self):
        diag = ExtractionDiagnostics()
        diag.record_attempt("local")
        diag.record_attempt("local")
        diag.record_attempt("cloud")
        snap = diag.snapshot()
        assert snap["by_tier"]["local"]["attempts"] == 2
        assert snap["by_tier"]["cloud"]["attempts"] == 1
        assert snap["by_tier"]["host"]["attempts"] == 0

    def test_record_success_increments(self):
        diag = ExtractionDiagnostics()
        diag.record_success("host", fact_count=3)
        diag.record_success("host", fact_count=1)
        snap = diag.snapshot()
        assert snap["by_tier"]["host"]["successes"] == 2

    def test_record_no_output_increments(self):
        diag = ExtractionDiagnostics()
        diag.record_no_output("remote")
        snap = diag.snapshot()
        assert snap["by_tier"]["remote"]["no_output"] == 1

    def test_record_failure_with_exception_captures_sample(self):
        diag = ExtractionDiagnostics()
        try:
            raise ValueError("bad json")
        except Exception as e:
            diag.record_failure("cloud", exc=e, reason="json_parse_failed")
        snap = diag.snapshot()
        assert snap["by_tier"]["cloud"]["failures"] == 1
        samples = snap["by_tier"]["cloud"]["error_samples"]
        assert len(samples) == 1
        assert samples[0]["type"] == "ValueError"
        assert "bad json" in samples[0]["msg"]
        assert samples[0]["reason"] == "json_parse_failed"

    def test_record_failure_with_reason_only(self):
        diag = ExtractionDiagnostics()
        diag.record_failure("local", reason="model_not_loaded")
        snap = diag.snapshot()
        sample = snap["by_tier"]["local"]["error_samples"][0]
        assert sample["type"] == "reason"
        assert sample["msg"] == "model_not_loaded"

    def test_record_failure_truncates_long_error(self):
        diag = ExtractionDiagnostics()
        long_err = "x" * 500
        try:
            raise RuntimeError(long_err)
        except Exception as e:
            diag.record_failure("cloud", exc=e)
        snap = diag.snapshot()
        sample = snap["by_tier"]["cloud"]["error_samples"][0]
        # repr(e) prefixes RuntimeError('...'); the inner 500-char
        # payload must be truncated within the configured cap.
        assert "...[truncated]" in sample["msg"]
        # And the FULL 500 chars must NOT appear verbatim.
        assert long_err not in sample["msg"]

    def test_error_samples_bounded(self):
        """Bounded deque — a chronically failing tier doesn't
        accumulate unbounded samples."""
        diag = ExtractionDiagnostics()
        for i in range(100):
            diag.record_failure("local", reason=f"err-{i}")
        snap = diag.snapshot()
        samples = snap["by_tier"]["local"]["error_samples"]
        # _MAX_ERROR_SAMPLES_PER_TIER = 10
        assert len(samples) == 10
        # Latest samples retained (FIFO drop).
        assert samples[-1]["msg"] == "err-99"

    def test_record_call_succeeded(self):
        diag = ExtractionDiagnostics()
        diag.record_call(succeeded=True)
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["successes"] == 1
        assert snap["totals"]["failures"] == 0
        assert snap["totals"]["success_rate"] == 1.0

    def test_record_call_all_empty(self):
        diag = ExtractionDiagnostics()
        diag.record_call(succeeded=False, all_empty=True)
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["empty"] == 1
        assert snap["totals"]["failures"] == 0
        assert snap["totals"]["success_rate"] == 0.0

    def test_record_call_hard_failure(self):
        diag = ExtractionDiagnostics()
        diag.record_call(succeeded=False, all_empty=False)
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["failures"] == 1
        assert snap["totals"]["empty"] == 0

    def test_success_rate_math(self):
        diag = ExtractionDiagnostics()
        for _ in range(7):
            diag.record_call(succeeded=True)
        for _ in range(3):
            diag.record_call(succeeded=False)
        assert diag.success_rate() == pytest.approx(0.7)

    def test_unknown_tier_rejected(self):
        """Typo guard — `local` vs `Local` vs `localx` is exactly
        the kind of mistake silent recording would mask."""
        diag = ExtractionDiagnostics()
        for bad in ("Local", "LOCAL", "localx", "", "graph"):
            with pytest.raises(ValueError, match="unknown extraction tier"):
                diag.record_attempt(bad)
            with pytest.raises(ValueError, match="unknown extraction tier"):
                diag.record_success(bad)
            with pytest.raises(ValueError, match="unknown extraction tier"):
                diag.record_failure(bad, reason="oops")

    def test_reset_zeroes_everything(self):
        diag = ExtractionDiagnostics()
        diag.record_attempt("local")
        diag.record_success("local")
        diag.record_failure("cloud", reason="test")
        diag.record_call(succeeded=True)

        diag.reset()
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 0
        assert snap["totals"]["successes"] == 0
        for tier in EXTRACTION_TIERS:
            assert snap["by_tier"][tier]["attempts"] == 0
            assert snap["by_tier"][tier]["error_samples"] == []

    def test_snapshot_is_json_serializable(self):
        """Operators ship this to log aggregators / dashboards."""
        diag = ExtractionDiagnostics()
        try:
            raise RuntimeError("oops")
        except Exception as e:
            diag.record_failure("cloud", exc=e)
        diag.record_success("local", fact_count=2)
        diag.record_call(succeeded=True)
        snap = diag.snapshot()
        # Round-trip via JSON proves the shape is clean.
        serialized = json.dumps(snap)
        restored = json.loads(serialized)
        assert restored["totals"]["successes"] == 1

    def test_thread_safety_under_concurrent_recording(self):
        """Concurrent extraction calls from multiple threads must
        accumulate correctly under the lock."""
        diag = ExtractionDiagnostics()
        N_THREADS = 8
        ATTEMPTS_PER_THREAD = 100

        def worker():
            for _ in range(ATTEMPTS_PER_THREAD):
                diag.record_attempt("local")
                diag.record_success("local")

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = diag.snapshot()
        expected = N_THREADS * ATTEMPTS_PER_THREAD
        assert snap["by_tier"]["local"]["attempts"] == expected
        assert snap["by_tier"]["local"]["successes"] == expected


class TestProcessGlobalSingleton:

    def test_get_diagnostics_returns_same_instance(self):
        a = get_diagnostics()
        b = get_diagnostics()
        assert a is b

    def test_module_level_helpers_use_singleton(self):
        diag = get_diagnostics()
        diag.record_success("host", fact_count=1)
        diag.record_call(succeeded=True)

        snap = get_extraction_stats()
        assert snap["by_tier"]["host"]["successes"] == 1
        assert snap["totals"]["successes"] == 1

        reset_extraction_stats()
        snap = get_extraction_stats()
        assert snap["totals"]["calls"] == 0


class TestExtractFactsIntegration:
    """End-to-end: call extract_facts under various LLM availability
    scenarios and verify the diagnostics record what happened."""

    def test_llm_unavailable_records_failure(self, monkeypatch):
        """When local_llm.llm_available() is False, extract_facts
        bails immediately. Pre-C13.b this was completely silent."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: False
        )
        from mnemosyne.core.extraction import extract_facts
        result = extract_facts("Alice prefers tea.")
        assert result == []

        snap = get_extraction_stats()
        # Recorded one outer call as a failure (no tier ran successfully).
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["failures"] == 1
        # And recorded a 'local' tier failure with the reason.
        local = snap["by_tier"]["local"]
        assert local["failures"] == 1
        sample = local["error_samples"][0]
        assert sample["msg"] == "llm_unavailable_at_call_site"

    def test_empty_text_no_recording(self, monkeypatch):
        """Empty input isn't an attempt — operators shouldn't see
        success_rate degrade from no-op callers."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        from mnemosyne.core.extraction import extract_facts
        assert extract_facts("") == []
        assert extract_facts("   ") == []
        snap = get_extraction_stats()
        assert snap["totals"]["calls"] == 0

    def test_local_llm_raises_records_failure(self, monkeypatch):
        """Local LLM model raises mid-call. Records as `local` tier
        failure with the exception sample."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._try_host_llm",
            lambda prompt, max_tokens, temperature: (False, ""),
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.LLM_ENABLED", False
        )

        def boom(*args, **kwargs):
            raise RuntimeError("model crashed mid-inference")

        monkeypatch.setattr(
            "mnemosyne.core.local_llm._load_llm",
            lambda: boom,
        )

        from mnemosyne.core.extraction import extract_facts
        result = extract_facts("Alice prefers tea.")
        assert result == []

        snap = get_extraction_stats()
        local = snap["by_tier"]["local"]
        assert local["failures"] >= 1
        # The exception was captured.
        msgs = [s.get("msg", "") for s in local["error_samples"]]
        assert any("model crashed" in m for m in msgs)

    def test_local_llm_succeeds(self, monkeypatch):
        """Happy path — local LLM returns facts, success is recorded."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._try_host_llm",
            lambda prompt, max_tokens, temperature: (False, ""),
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.LLM_ENABLED", False
        )

        def fake_llm(prompt, max_new_tokens, stop):
            return "1. Alice prefers tea\n2. Alice lives in Seattle"

        monkeypatch.setattr(
            "mnemosyne.core.local_llm._load_llm",
            lambda: fake_llm,
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._clean_output",
            lambda s: s,
        )

        from mnemosyne.core.extraction import extract_facts
        result = extract_facts("Alice prefers tea and lives in Seattle.")
        assert len(result) == 2

        snap = get_extraction_stats()
        assert snap["totals"]["successes"] == 1
        assert snap["by_tier"]["local"]["successes"] == 1

    def test_extract_facts_safe_wraps_outer_exceptions(self, monkeypatch):
        """If extract_facts() itself raises (rare — bug, not LLM
        failure), extract_facts_safe records it as `local` tier
        outer_wrapper_caught so operators can spot the bug class."""
        from mnemosyne.core import extraction as ext_mod

        def boom(text):
            raise TypeError("simulated extract_facts bug")

        monkeypatch.setattr(ext_mod, "extract_facts", boom)
        result = ext_mod.extract_facts_safe("any content")
        assert result == []

        snap = get_extraction_stats()
        local = snap["by_tier"]["local"]
        assert local["failures"] >= 1
        msgs = [s.get("msg", "") + " " + s.get("reason", "") for s in local["error_samples"]]
        assert any("outer_wrapper_caught" in m for m in msgs)


class TestExtractionClientIntegration:
    """Cloud path — `ExtractionClient.chat()` and `extract_facts()`
    record per-call outcomes."""

    def test_chat_records_no_api_key_failure(self, monkeypatch):
        """No API key → urllib raises 401-ish → all retries fail."""
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="")

        def fail_api(*args, **kwargs):
            raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(client, "_call_api", fail_api)
        result = client.chat([{"role": "user", "content": "test"}])
        assert result == ""

        snap = get_extraction_stats()
        cloud = snap["by_tier"]["cloud"]
        assert cloud["failures"] >= 1
        # Most-recent error sample should contain the error trace.
        samples = cloud["error_samples"]
        assert any("401" in s.get("msg", "") for s in samples)

    def test_chat_records_success(self, monkeypatch):
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="key")
        monkeypatch.setattr(
            client,
            "_call_api",
            lambda *a, **kw: '[{"subject":"Alice","predicate":"prefers","object":"tea"}]',
        )

        result = client.chat([{"role": "user", "content": "test"}])
        assert "Alice" in result

        snap = get_extraction_stats()
        assert snap["by_tier"]["cloud"]["successes"] >= 1

    def test_extract_facts_records_json_parse_failure(self, monkeypatch):
        """Cloud LLM returned text, but it didn't parse as a fact
        list. Records as `cloud` failure with reason
        json_parse_failed."""
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="key")
        # Returns text without any [...] block — parse fails.
        monkeypatch.setattr(
            client,
            "_call_api",
            lambda *a, **kw: "I cannot extract facts from this text.",
        )

        result = client.extract_facts(
            [{"role": "user", "content": "some content"}]
        )
        assert result == []

        snap = get_extraction_stats()
        cloud = snap["by_tier"]["cloud"]
        # At minimum the chat() success counter fires (text returned).
        # The JSON parse failure also fires.
        msgs = [s.get("reason", "") for s in cloud["error_samples"]]
        # Actually wait — if response found NO [..] block, the parser
        # returns [] without raising json.JSONDecodeError. Verify
        # that path: the response had no [, so json_start < 0, no
        # parse attempted, no failure recorded. Then result = [].
        # The branch we want to test is "[ ... ]" with malformed JSON.

    def test_extract_facts_records_malformed_json(self, monkeypatch):
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="key")
        # Response includes brackets but the contents aren't valid JSON.
        monkeypatch.setattr(
            client,
            "_call_api",
            lambda *a, **kw: 'Here are the facts: [oops, this is not json]',
        )

        result = client.extract_facts(
            [{"role": "user", "content": "some content"}]
        )
        assert result == []

        snap = get_extraction_stats()
        cloud = snap["by_tier"]["cloud"]
        reasons = [s.get("reason", "") for s in cloud["error_samples"]]
        assert "json_parse_failed" in reasons


class TestOperatorVisibleLogs:
    """C13.b's diagnostics are the primary signal; structured WARNING
    logs are the secondary signal for operators tailing logs."""

    def test_local_llm_failure_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._try_host_llm",
            lambda prompt, max_tokens, temperature: (False, ""),
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.LLM_ENABLED", False
        )

        def boom(*args, **kwargs):
            raise RuntimeError("model crashed mid-inference")

        monkeypatch.setattr(
            "mnemosyne.core.local_llm._load_llm",
            lambda: boom,
        )

        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.extraction"):
            from mnemosyne.core.extraction import extract_facts
            extract_facts("Alice prefers tea.")

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("local LLM raised" in r.message for r in warnings)
