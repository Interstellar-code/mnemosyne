"""
Mnemosyne Local LLM Consolidation
=================================
Lightweight on-device summarization for the sleep/consolidation cycle.
Uses ctransformers for GGUF model inference. Falls back to aaak encoding
if the model is unavailable or inference fails.

Model cache: ~/.hermes/mnemosyne/models/
Default model: TinyLlama-1.1B-Chat-v1.0-GGUF (Q4_K_M, ~600MB)
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Optional

# --- Config ------------------------------------------------------------------
DEFAULT_MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
DEFAULT_MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_CACHE_DIR = Path.home() / ".hermes" / "mnemosyne" / "models"

LLM_ENABLED = os.environ.get("MNEMOSYNE_LLM_ENABLED", "true").lower() in ("1", "true", "yes")
LLM_MAX_TOKENS = int(os.environ.get("MNEMOSYNE_LLM_MAX_TOKENS", "256"))
LLM_N_THREADS = int(os.environ.get("MNEMOSYNE_LLM_N_THREADS", "4"))
LLM_N_CTX = int(os.environ.get("MNEMOSYNE_LLM_N_CTX", "2048"))

# Override model via env
_env_repo = os.environ.get("MNEMOSYNE_LLM_REPO")
_env_file = os.environ.get("MNEMOSYNE_LLM_FILE")
if _env_repo and _env_file:
    DEFAULT_MODEL_REPO = _env_repo
    DEFAULT_MODEL_FILE = _env_file

# Remote API config
LLM_BASE_URL = os.environ.get("MNEMOSYNE_LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("MNEMOSYNE_LLM_API_KEY", "")
LLM_REMOTE_MODEL = os.environ.get("MNEMOSYNE_LLM_MODEL", "")

# --- Lazy singleton ----------------------------------------------------------
_llm_instance = None
_llm_available = None  # None = not checked yet


def _ensure_sys_path():
    """Ensure /usr/local/lib/python3.11/site-packages is in sys.path
    so ctransformers is discoverable when Hermes runs in a venv."""
    sp = "/usr/local/lib/python3.11/site-packages"
    if sp not in sys.path and os.path.isdir(sp):
        sys.path.append(sp)


def _model_path() -> Optional[Path]:
    """Return path to the local GGUF model file, or None if not downloaded."""
    candidate = MODEL_CACHE_DIR / DEFAULT_MODEL_FILE
    return candidate if candidate.exists() else None


def _download_model() -> Path:
    """Download the GGUF model from HuggingFace if not present."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_CACHE_DIR / DEFAULT_MODEL_FILE
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface-hub"
        )

    downloaded = hf_hub_download(
        repo_id=DEFAULT_MODEL_REPO,
        filename=DEFAULT_MODEL_FILE,
        local_dir=str(MODEL_CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def _load_llm():
    """Lazy-load the ctransformers model. Returns None on failure."""
    global _llm_instance, _llm_available
    if _llm_instance is not None:
        return _llm_instance
    if not LLM_ENABLED:
        _llm_available = False
        return None

    _ensure_sys_path()

    try:
        from ctransformers import AutoModelForCausalLM
    except ImportError:
        _llm_available = False
        return None

    model_file = _model_path()
    if model_file is None:
        try:
            model_file = _download_model()
        except Exception:
            _llm_available = False
            return None

    try:
        _llm_instance = AutoModelForCausalLM.from_pretrained(
            str(model_file),
            model_type="llama",
            max_new_tokens=LLM_MAX_TOKENS,
            threads=LLM_N_THREADS,
            context_length=LLM_N_CTX,
        )
        _llm_available = True
        return _llm_instance
    except Exception:
        _llm_available = False
        return None


def _build_prompt(memories: List[str], source: str = "") -> str:
    """Build a consolidation prompt from a list of memory strings."""
    # TinyLlama-1.1B-Chat uses the Zephyr format:
    # <|user|>\n...\n</s>\n<|assistant|>\n
    header = (
        "Summarize the following memories into 1-3 concise sentences. "
        "Preserve facts, names, preferences, and decisions. Discard fluff."
    )
    if source:
        header += f" Source: {source}."

    lines = "\n".join(f"- {m}" for m in memories if m)
    prompt = f"<|user|>\n{header}\n\n{lines}\n</s>\n<|assistant|>\n"
    return prompt


def _clean_output(text: str) -> str:
    """Strip assistant tokens and extra whitespace from model output."""
    # Remove any echoed prompt fragments
    text = text.replace("<|assistant|>", "").replace("<|user|>", "")
    text = text.replace("</s>", "").strip()
    # If the model echoed the instructions, cut up to the first real sentence
    text = re.sub(r"^(Summarize the following memories.*?[.!?:]\s*)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(Preserve facts.*?[.!?:]\s*)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^Source:.*?\n", "", text, flags=re.IGNORECASE)
    # Remove bullet echoes
    text = re.sub(r"^\s*[-*]\s.*\n", "", text, flags=re.MULTILINE)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English, with safety margin."""
    return max(1, len(text) // 4)


def _prompt_token_budget() -> int:
    """Return usable token budget for memory content (reserves overhead + output)."""
    overhead = 80  # system prompt, formatting tokens, stop sequences
    output_reserve = LLM_MAX_TOKENS
    safety_margin = int(LLM_N_CTX * 0.2)  # 20% safety buffer
    return max(64, LLM_N_CTX - overhead - output_reserve - safety_margin)


def chunk_memories_by_budget(memories: List[str], source: str = "") -> List[List[str]]:
    """
    Split memories into chunks that fit within the LLM context window.
    Returns list of memory sublists, each safe to pass to summarize_memories().
    """
    if not memories:
        return []

    budget = _prompt_token_budget()
    chunks = []
    current_chunk = []
    current_tokens = 0

    # Header overhead
    header = (
        "Summarize the following memories into 1-3 concise sentences. "
        "Preserve facts, names, preferences, and decisions. Discard fluff."
    )
    if source:
        header += f" Source: {source}."
    header_tokens = _estimate_tokens(header + "\n\n")

    # Format overhead per memory ("- " + "\n")
    format_overhead = _estimate_tokens("- \n")

    available = budget - header_tokens

    for memory in memories:
        mem_tokens = _estimate_tokens(memory) + format_overhead

        # If a single memory exceeds the entire budget, skip it (will fall back to aaak)
        if mem_tokens > budget:
            continue

        if current_tokens + mem_tokens > available and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(memory)
        current_tokens += mem_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def llm_available() -> bool:
    """Check whether the local LLM is loaded and ready."""
    global _llm_available
    # Remote LLM is always "available" if configured (we'll discover at call time)
    if LLM_BASE_URL:
        return True
    if _llm_available is not None:
        return _llm_available
    _load_llm()
    return bool(_llm_available)


def _call_remote_llm(prompt: str) -> Optional[str]:
    """Call an OpenAI-compatible remote endpoint for summarization."""
    if not LLM_BASE_URL:
        return None

    import json

    # Try httpx first, fall back to urllib
    try:
        import httpx
        has_httpx = True
    except ImportError:
        has_httpx = False

    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    model = LLM_REMOTE_MODEL or "local"  # llama.cpp server accepts any model name

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0.3,
        "stop": ["</s>", "<|user|>"]
    }

    try:
        if has_httpx:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        else:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                data = json.loads(resp.read().decode())

        choices = data.get("choices", [])
        if choices and choices[0].get("message", {}).get("content"):
            return choices[0]["message"]["content"]
        return None
    except Exception:
        return None


def summarize_memories(memories: List[str], source: str = "") -> Optional[str]:
    """
    Summarize a batch of working-memory items into a single episodic string.
    Returns None if the LLM is unavailable or inference fails (caller should
    fall back to aaak encoding).
    """
    if not memories:
        return None

    prompt = _build_prompt(memories, source=source)
    raw = None

    # --- Try remote LLM first if configured ---
    if LLM_BASE_URL:
        raw = _call_remote_llm(prompt)
        if raw:
            cleaned = _clean_output(raw)
            return cleaned if cleaned else None

    # --- Fall back to local ctransformers ---
    llm = _load_llm()
    if llm is not None:
        try:
            raw = llm(prompt, max_new_tokens=LLM_MAX_TOKENS, stop=["</s>", "<|user|>"])
            cleaned = _clean_output(raw)
            return cleaned if cleaned else None
        except Exception:
            pass

    return None
