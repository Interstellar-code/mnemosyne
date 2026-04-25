import os
import pytest
from unittest.mock import patch, MagicMock

from mnemosyne.core import local_llm


class TestRemoteLLM:
    def test_llm_available_returns_true_when_base_url_set(self, monkeypatch):
        """BUG-2: llm_available() must report True when remote is configured."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://localhost:8080/v1")
        # Reset module-level cache
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setattr(local_llm, "_llm_available", None)
        monkeypatch.setattr(local_llm, "_llm_instance", None)

        assert local_llm.llm_available() is True

    def test_call_remote_llm_with_mock_response(self, monkeypatch):
        """BUG-2: _call_remote_llm parses OpenAI-compatible response correctly."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://test-server/v1")
        monkeypatch.setenv("MNEMOSYNE_LLM_API_KEY", "sk-test")
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://test-server/v1")
        monkeypatch.setattr(local_llm, "LLM_API_KEY", "sk-test")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "test-model")
        monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 128)

        mock_response = {
            "choices": [
                {"message": {"content": "This is a test summary."}}
            ]
        }

        # Mock httpx by patching the import inside _call_remote_llm
        mock_client = MagicMock()
        mock_client.post.return_value.raise_for_status = lambda: None
        mock_client.post.return_value.json.return_value = mock_response
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = lambda *args: None

        mock_httpx_module = MagicMock()
        mock_httpx_module.Client = MagicMock(return_value=mock_client)

        # Save original import to avoid recursion
        _orig_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "httpx":
                return mock_httpx_module
            return _orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", mock_import):
            result = local_llm._call_remote_llm("Test prompt")
            assert result == "This is a test summary."

            # Verify the call was made with correct payload
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://test-server/v1/chat/completions"
            payload = call_args[1]["json"]
            assert payload["model"] == "test-model"
            assert payload["messages"][0]["content"] == "Test prompt"
            assert "Authorization" in call_args[1]["headers"]

    def test_call_remote_llm_urllib_fallback(self, monkeypatch):
        """BUG-2: Falls back to urllib when httpx unavailable."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://test-server/v1")
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://test-server/v1")
        monkeypatch.setattr(local_llm, "LLM_API_KEY", "")
        monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 128)

        mock_response = {
            "choices": [
                {"message": {"content": "Fallback summary."}}
            ]
        }

        import json
        mock_data = json.dumps(mock_response).encode()

        class MockResponse:
            def read(self):
                return mock_data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        # Patch httpx import in local_llm module to simulate it not being installed
        with patch.dict("sys.modules", {"httpx": None}):
            with patch("urllib.request.urlopen", return_value=MockResponse()):
                result = local_llm._call_remote_llm("Test prompt")
                assert result == "Fallback summary."

    def test_summarize_memories_prefers_remote_over_local(self, monkeypatch):
        """BUG-2: summarize_memories() calls remote when BASE_URL is set."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://remote/v1")
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
        monkeypatch.setattr(local_llm, "_llm_available", False)
        monkeypatch.setattr(local_llm, "_llm_instance", None)

        with patch.object(local_llm, "_call_remote_llm", return_value="Remote summary.") as mock_remote:
            result = local_llm.summarize_memories(["Memory one", "Memory two"])
            assert result == "Remote summary."
            mock_remote.assert_called_once()

    def test_summarize_memories_falls_back_local_when_remote_fails(self, monkeypatch):
        """BUG-2: When remote fails and local is unavailable, return None (aaak fallback)."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://remote/v1")
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")

        # Remote returns None (failure), local _load_llm returns None (unavailable)
        with patch.object(local_llm, "_call_remote_llm", return_value=None) as mock_remote:
            with patch.object(local_llm, "_load_llm", return_value=None) as mock_load:
                result = local_llm.summarize_memories(["Memory one"])
                # Should return None since both remote and local fail
                assert result is None
                mock_remote.assert_called_once()
                mock_load.assert_called_once()
