"""
Mnemosyne Local LLM Consolidation
=================================
Lightweight on-device summarization for the sleep/consolidation cycle.
Uses llama-cpp-python (ARM64 + x86_64 native) with ctransformers fallback.
Falls back to aaak encoding if the model is unavailable or inference fails.

Model cache: ~/.hermes/mnemosyne/models/
Default model: openbmb/MiniCPM5-1B-GGUF (Q4_K_M, ~656MB)
"""

import logging
import os
import sys
import re
from pathlib import Path
from typing import List, Optional

from mnemosyne.core.user_agent import application_user_agent

# --- Config ------------------------------------------------------------------
DEFAULT_MODEL_REPO = "openbmb/MiniCPM5-1B-GGUF"
DEFAULT_MODEL_FILE = "MiniCPM5-1B-Q4_K_M.gguf"
MODEL_CACHE_DIR = Path.home() / ".hermes" / "mnemosyne" / "models"

logger = logging.getLogger(__name__)

LLM_ENABLED = os.environ.get("MNEMOSYNE_LLM_ENABLED", "true").lower() in ("1", "true", "yes")
LLM_MAX_TOKENS=int(os.environ.get("MNEMOSYNE_LLM_MAX_TOKENS", "2048") or "2048")
LLM_N_THREADS = int(os.environ.get("MNEMOSYNE_LLM_N_THREADS", "4"))
LLM_N_CTX = int(os.environ.get("MNEMOSYNE_LLM_N_CTX", "2048"))

# Override model via env
_env_repo = os.environ.get("MNEMOSYNE_LLM_REPO")
_env_file = os.environ.get("MNEMOSYNE_LLM_FILE")
if _env_repo and _env_file:
    DEFAULT_MODEL_REPO = _env_repo
    DEFAULT_MODEL_FILE = _env_file

# Override the cache location via env. Environment-only and read at import, like
# the repo/file overrides above. Unset or blank keeps the historical path.
#
# `strip()` decides only whether anything was named; the raw value is what
# becomes the path, because a POSIX directory name may legitimately begin or end
# with whitespace and stripping it would silently select a different one.
_env_cache_dir = os.environ.get("MNEMOSYNE_MODEL_CACHE_DIR", "")
# Provenance, not a derived fact: the user may set the variable to exactly the
# default path, so the value alone cannot say whether it was chosen explicitly.
# The error message below reads differently in each case.
MODEL_CACHE_DIR_FROM_ENV = bool(_env_cache_dir.strip())
if MODEL_CACHE_DIR_FROM_ENV:
    MODEL_CACHE_DIR = Path(_env_cache_dir).expanduser()

# Remote API config
LLM_BASE_URL = os.environ.get("MNEMOSYNE_LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("MNEMOSYNE_LLM_API_KEY", "")
LLM_REMOTE_MODEL = os.environ.get("MNEMOSYNE_LLM_MODEL", "")

# Optional named provider preset. When MNEMOSYNE_LLM_PROVIDER names a known
# preset (see mnemosyne.core.llm_providers), it fills in the OpenAI-compatible
# base URL and a default model for the selected MNEMOSYNE_LLM_REGION. Explicit
# MNEMOSYNE_LLM_BASE_URL / MNEMOSYNE_LLM_MODEL always win, so existing
# raw-env-var configurations behave exactly as before.
LLM_PROVIDER = os.environ.get("MNEMOSYNE_LLM_PROVIDER", "").strip()
LLM_REGION = os.environ.get("MNEMOSYNE_LLM_REGION", "").strip()
if LLM_PROVIDER:
    from mnemosyne.core.llm_providers import resolve_provider_defaults

    _preset_base_url, _preset_model = resolve_provider_defaults(LLM_PROVIDER, LLM_REGION)
    if not LLM_BASE_URL and _preset_base_url:
        LLM_BASE_URL = _preset_base_url.rstrip("/")
    if not LLM_REMOTE_MODEL and _preset_model:
        LLM_REMOTE_MODEL = _preset_model

LLM_TIMEOUT = float(os.environ.get("MNEMOSYNE_LLM_TIMEOUT", "60"))

# Retryable errors only: 404/400 (model-not-found), 5xx, connection. Not 401/403/429.
LLM_FALLBACK_MODELS = [
    m.strip() for m in os.environ.get("MNEMOSYNE_LLM_FALLBACK_MODELS", "").split(",")
    if m.strip()
]
LLM_FALLBACK_BASE_URL = os.environ.get("MNEMOSYNE_LLM_FALLBACK_BASE_URL", "").rstrip("/") or LLM_BASE_URL
LLM_FALLBACK_API_KEY = os.environ.get("MNEMOSYNE_LLM_FALLBACK_API_KEY", "") or LLM_API_KEY


# Payload keys the extra body may not set. The merge happens last, so without
# this a typo'd key would silently send different messages or a different model
# than every log line and config value says, and the bug report would be
# unreadable. Provider-specific keys, which is what the escape hatch is for,
# are unaffected.
_EXTRA_BODY_RESERVED = ("messages", "model", "stream")


def _parse_extra_body(var: str) -> dict:
    """Read a JSON object from ``var`` for merging into the request payload.

    Carries keys the OpenAI-compatible shape has no name for (a thinking-mode
    toggle, provider routing), one object per endpoint. Unset, blank, invalid
    JSON or a non-object value all mean nothing is merged; a reserved key
    (``messages``, ``model``, ``stream``) is dropped and the rest of the object
    is kept. The rejected cases say so on stderr, since this runs at import
    before logging is configured.
    """
    raw = os.environ.get(var, "").strip()
    if not raw:
        return {}
    import json
    try:
        data = json.loads(raw)
    except ValueError as exc:
        print(f"[mnemosyne] {var} is not valid JSON, ignored: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"[mnemosyne] {var} must be a JSON object, ignored", file=sys.stderr)
        return {}
    reserved = [k for k in _EXTRA_BODY_RESERVED if k in data]
    if reserved:
        print(
            f"[mnemosyne] {var} may not set {', '.join(reserved)}, "
            "dropped from the request body",
            file=sys.stderr,
        )
        data = {k: v for k, v in data.items() if k not in _EXTRA_BODY_RESERVED}
    return data


LLM_EXTRA_BODY = _parse_extra_body("MNEMOSYNE_LLM_EXTRA_BODY")
LLM_FALLBACK_EXTRA_BODY = _parse_extra_body("MNEMOSYNE_LLM_FALLBACK_EXTRA_BODY")

# Host LLM adapter (Hermes or another agent). Disabled by default to preserve
# existing standalone behavior. When MNEMOSYNE_HOST_LLM_ENABLED=true and a
# backend is registered via mnemosyne.core.llm_backends.set_host_llm_backend(),
# the host backend is consulted before the existing remote/local chain.
# See docs/hermes-llm-integration.md for the full behavior model.
HOST_LLM_ENABLED = os.environ.get("MNEMOSYNE_HOST_LLM_ENABLED", "false").lower() in ("1", "true", "yes")
HOST_LLM_PROVIDER = os.environ.get("MNEMOSYNE_HOST_LLM_PROVIDER", "").strip() or None
HOST_LLM_MODEL = os.environ.get("MNEMOSYNE_HOST_LLM_MODEL", "").strip() or None
# Per-attempt cap, passed explicitly to the host so it overrides Hermes'
# auxiliary.<task>.timeout. 15s sat at the median of a reasoning aux model on a
# consolidation prompt (7-18s measured), so calls timed out as a coin flip.
try:
    HOST_LLM_TIMEOUT = float(os.environ.get("MNEMOSYNE_HOST_LLM_TIMEOUT", "60") or "60")
except ValueError:
    HOST_LLM_TIMEOUT = 60.0
if HOST_LLM_TIMEOUT <= 0:  # 0/negative would fail every host call instantly
    HOST_LLM_TIMEOUT = 60.0
# Host context window: local-model-calibrated LLM_N_CTX (2048) is too small for
# Codex/GPT-class aux models; use this larger budget when the host is the path.
HOST_LLM_N_CTX = int(os.environ.get("MNEMOSYNE_HOST_LLM_N_CTX", "32000"))

# Optional consolidation prompt override. Supports {source}, {memories}, and
# {memory_count}; unset keeps the historical built-in prompt.
SLEEP_PROMPT = os.environ.get("MNEMOSYNE_SLEEP_PROMPT", "").strip()

# --- Lazy singleton ----------------------------------------------------------
_llm_instance = None
_llm_backend = None  # "llamacpp", "ctransformers", or None
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


def _ensure_model_cache_dir() -> Path:
    """Create the model cache directory, or fail naming it and why.

    An explicitly set ``MNEMOSYNE_MODEL_CACHE_DIR`` is authoritative. Falling
    back to the default on failure would reinstate the very location the user
    moved away from, quietly, which is the substitution this override exists to
    prevent.

    The error is logged as well as raised. ``_load_llm()`` catches every
    exception from the download path and degrades to AAAK, so a raised message
    alone would never reach the user; logging is what makes "fail clearly"
    actually clear.
    """
    if MODEL_CACHE_DIR_FROM_ENV:
        source = f"MNEMOSYNE_MODEL_CACHE_DIR is set to {MODEL_CACHE_DIR}"
        remedy = "Point it at a writable directory, or unset it to use the default."
    else:
        source = f"The model cache directory {MODEL_CACHE_DIR}"
        remedy = "Set MNEMOSYNE_MODEL_CACHE_DIR to relocate it."

    try:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        message = f"{source}, which could not be created ({exc}). {remedy}"
        logger.error("%s", message)
        raise RuntimeError(message) from exc

    # Write *and* search: a directory can be mode 0o200, which passes W_OK while
    # creating anything inside it still fails with PermissionError, because
    # traversing into a directory needs the execute bit.
    if not os.access(MODEL_CACHE_DIR, os.W_OK | os.X_OK):
        message = f"{source}, which is not writable. {remedy}"
        logger.error("%s", message)
        raise RuntimeError(message)

    return MODEL_CACHE_DIR


def _download_model() -> Path:
    """Download the GGUF model from HuggingFace if not present."""
    _ensure_model_cache_dir()
    local_path = MODEL_CACHE_DIR / DEFAULT_MODEL_FILE
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface-hub"
        )

    default_artifact = (
        DEFAULT_MODEL_REPO == "openbmb/MiniCPM5-1B-GGUF"
        and DEFAULT_MODEL_FILE == "MiniCPM5-1B-Q4_K_M.gguf"
    )
    size_notice = " (approximately 656 MB)" if default_artifact else ""
    logger.warning(
        "Downloading local LLM model %s from %s%s to %s. "
        "The current operation will block until the download completes. "
        "Set MNEMOSYNE_LLM_ENABLED=false for AAAK-only consolidation, "
        "or pre-cache the GGUF to avoid this download.",
        DEFAULT_MODEL_FILE,
        DEFAULT_MODEL_REPO,
        size_notice,
        MODEL_CACHE_DIR,
    )
    downloaded = hf_hub_download(
        repo_id=DEFAULT_MODEL_REPO,
        filename=DEFAULT_MODEL_FILE,
        local_dir=str(MODEL_CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def _load_llm_llamacpp(model_path: Path):
    """Load the GGUF model via llama-cpp-python. Returns Llama instance or None."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return None

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=LLM_N_CTX,
            n_threads=LLM_N_THREADS,
            verbose=False,
        )
        return llm
    except Exception:
        return None


def _load_llm_ctransformers(model_path: Path):
    """Load the GGUF model via ctransformers (x86_64 only). Returns model or None."""
    _ensure_sys_path()

    try:
        from ctransformers import AutoModelForCausalLM
    except ImportError:
        return None

    try:
        return AutoModelForCausalLM.from_pretrained(
            str(model_path),
            model_type="llama",
            max_new_tokens=LLM_MAX_TOKENS,
            threads=LLM_N_THREADS,
            context_length=LLM_N_CTX,
        )
    except Exception:
        return None


def _load_llm():
    """Lazy-load the best available local LLM backend.
    
    Priority: llama-cpp-python > ctransformers (x86_64 fallback).
    Returns the loaded model/LLM instance, or None if no backend works.
    """
    global _llm_instance, _llm_backend, _llm_available

    if _llm_instance is not None:
        return _llm_instance

    if not LLM_ENABLED:
        _llm_available = False
        return None

    # Get or download model file
    model_file = _model_path()
    if model_file is None:
        try:
            model_file = _download_model()
        except Exception:
            _llm_available = False
            return None

    # Try llama-cpp-python first (works on ARM64 + x86_64)
    llm = _load_llm_llamacpp(model_file)
    if llm is not None:
        _llm_instance = llm
        _llm_backend = "llamacpp"
        _llm_available = True
        return _llm_instance

    # Fall back to ctransformers (x86_64 only)
    llm = _load_llm_ctransformers(model_file)
    if llm is not None:
        _llm_instance = llm
        _llm_backend = "ctransformers"
        _llm_available = True
        return _llm_instance

    _llm_available = False
    return None


def _call_local_llm(prompt: str) -> Optional[str]:
    """Run inference on the local LLM using whichever backend is loaded."""
    llm = _load_llm()
    if llm is None:
        return None

    try:
        if _llm_backend == "llamacpp":
            # llama-cpp-python uses chat completion API
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                stop=["</s>", "<|user|>"],
                temperature=0.3,
            )
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return None
        else:
            # ctransformers uses direct callable
            return llm(prompt, max_new_tokens=LLM_MAX_TOKENS, stop=["</s>", "<|user|>"])
    except Exception:
        return None


def _memory_lines(memories: List[str]) -> str:
    """Format memories for insertion into consolidation prompts."""
    return "\n".join(f"- {m}" for m in memories if m)


# --- Dynamic sleep prompts ----------------------------------------------------
# Prompt templates are looked up when a sweep starts (resolve_prompt_vars), so
# an edited prompt file applies on the next night without a restart. Per kind:
# (config key naming a file, default file name in the profile's Mnemosyne
# config dir, config key holding the template text itself).
_PROMPT_SOURCES = {
    "sleep": ("sleep_prompt_file", "sleep_prompt.md", "sleep_prompt"),
    "model_refresh": ("sleep_model_refresh_prompt_file", "model_refresh_prompt.md",
                      "sleep_model_refresh_prompt"),
}
# Keys in prompt_vars carrying the templates resolved for this sweep.
_TEMPLATE_KEYS = {"sleep": "_sleep_template", "model_refresh": "_model_refresh_template"}
PROMPT_VAR_DEFAULTS = {
    "principal": "the user",
    "principal_card": "(no description on file)",
    "existing_slots": "(none)",
    "profile": "default",
    "session_kind": "chat",
    "date_range": "unknown date",
}
_render_warned = False


def _prompt_setting(key: str, settings: Optional[dict] = None):
    """settings (host config, e.g. Hermes memory.mnemosyne.*) > Mnemosyne
    config.yaml > env. Blank counts as unset at every level, so a seeded empty
    config.yaml value never shadows the env var."""
    val = (settings or {}).get(key)
    if val in (None, ""):
        try:
            from mnemosyne.core.config import ENV_VAR_MAP, get_config
            val = get_config().get(key)
            if val in (None, ""):
                env = ENV_VAR_MAP.get(key)
                val = os.environ.get(env) if env else None
        except Exception:
            val = None
    return None if val in (None, "") else val


def _sleep_prompt_template(kind: str, settings: Optional[dict] = None) -> Optional[str]:
    """Return the operator's template for ``kind`` ("sleep"/"model_refresh"), or None.

    Precedence: configured file path > ``<profile mnemosyne dir>/<kind file>``
    if it exists > template text from config/env (MNEMOSYNE_SLEEP_PROMPT /
    MNEMOSYNE_SLEEP_MODEL_REFRESH_PROMPT) > None (built-in prompt).
    """
    file_key, default_name, text_key = _PROMPT_SOURCES[kind]
    paths = []
    configured = _prompt_setting(file_key, settings)
    if configured:
        paths.append(Path(str(configured)).expanduser())
    try:
        from mnemosyne.core.config import _default_config_path
        paths.append(_default_config_path().parent / default_name)
    except Exception:
        pass
    for i, path in enumerate(paths):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            if configured and i == 0:
                logger.warning("sleep prompt: cannot read %s=%s (%s); trying the next source",
                               file_key, path, exc)
            continue
        if text:
            return text
    text = str(_prompt_setting(text_key, settings) or "").strip()
    if not text and kind == "sleep":
        text = SLEEP_PROMPT  # import-time env value, kept for compatibility
    return text or None


class _PromptVars(dict):
    def __missing__(self, key):  # unknown placeholder -> "" instead of KeyError
        return ""


def _render_prompt(template: str, values: dict) -> Optional[str]:
    """Render ``template`` with str.format_map; None (+ one WARNING) on error."""
    global _render_warned
    try:
        return template.format_map(_PromptVars(values))
    except (ValueError, IndexError, KeyError, AttributeError, TypeError) as exc:
        if not _render_warned:
            _render_warned = True
            logger.warning("sleep prompt: template does not render (%s: %s); using the "
                           "built-in prompt. Literal braces must be doubled: {{ }}",
                           type(exc).__name__, exc)
        return None


def _first_sentence_name(body: str) -> Optional[str]:
    """Name from an identity body such as "Rohit Sharma. Lives in ..."."""
    first = re.split(r"(?<=[.!?])\s", body.strip(), maxsplit=1)[0].strip().rstrip(".!?").strip()
    first = re.sub(r"^(?:the\s+)?user(?:'s\s+name)?\s+is\s+", "", first, flags=re.I)
    first = re.split(r"[,;(]", first, maxsplit=1)[0].strip()  # "Rohit, 45, ..." -> "Rohit"
    return first if 0 < len(first) <= 40 else None


def _cut_card(body: str, limit: int) -> str:
    body = " ".join(body.split())
    if len(body) <= limit:
        return body
    head = body[:limit]
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", head)]
    return head[:ends[-1]] if ends else head.rsplit(" ", 1)[0]


def resolve_prompt_vars(conn, db_path, owner_id: str, *, card_chars: Optional[int] = None,
                        settings: Optional[dict] = None) -> dict:
    """Resolve the per-sweep prompt variables and templates. Never raises.

    {principal}: setting principal_name > first sentence of the owner's
    model:user/identity slot > "the user". {principal_card}: that slot only,
    cut at a sentence end within sleep_prompt_card_chars (0 = omit).
    {existing_slots}: the owner's model:* slot names. Templates are read here,
    once per sweep, so every chunk of the sweep sees the same prompt.
    """
    global _render_warned
    _render_warned = False
    out = dict(PROMPT_VAR_DEFAULTS)
    out["profile"] = owner_id or "default"
    out["_slot_names"] = None  # unknown: parse keeps names verbatim
    for kind, key in _TEMPLATE_KEYS.items():
        try:
            out[key] = _sleep_prompt_template(kind, settings)
        except Exception as exc:
            logger.debug("sleep prompt: %s template lookup failed: %s", kind, exc)
            out[key] = None
    name = str(_prompt_setting("principal_name", settings) or "").strip()
    if name:
        out["principal"] = name
    if card_chars is None:
        try:
            raw = _prompt_setting("sleep_prompt_card_chars", settings)
            card_chars = 300 if raw is None else int(raw)
        except (TypeError, ValueError):
            card_chars = 300
    try:
        from mnemosyne.core.canonical import CanonicalStore
        store = CanonicalStore(db_path=db_path, conn=conn)
        row = store.recall(out["profile"], "model:user", "identity")
        body = str((row or {}).get("body") or "").strip()
        if body and card_chars > 0:
            out["principal_card"] = _cut_card(body, card_chars)
        out["principal"] = name or (_first_sentence_name(body) if body else None) or "the user"
        slots = [f"{r['category']}/{r['name']}" for r in store.list(out["profile"])
                 if str(r.get("category") or "").startswith("model:")]
        out["_slot_names"] = tuple(slots)
        if slots:
            out["existing_slots"] = "\n".join(f"- {s}" for s in slots)
    except Exception as exc:
        logger.debug("sleep prompt: variable resolution failed, using defaults: %s", exc)
    return out


def session_kind(session_id: Optional[str]) -> str:
    """dm / group / api / cli from the stored session id shape; else "chat"."""
    sid = str(session_id or "")
    if ":dm:" in sid:
        return "dm"
    if ":group:" in sid:
        return "group"
    if sid.startswith("hermes_api_"):
        return "api"
    if sid.startswith("hermes_") and ":" not in sid:
        return "cli"
    return "chat"


def date_range(items) -> str:
    """"YYYY-MM-DD" or "YYYY-MM-DD to YYYY-MM-DD" over the rows' timestamps."""
    days = sorted(
        str(item.get("timestamp") or "")[:10] for item in items or ()
        if re.match(r"\d{4}-\d{2}-\d{2}", str(item.get("timestamp") or ""))
    )
    if not days:
        return "unknown date"
    return days[0] if days[0] == days[-1] else f"{days[0]} to {days[-1]}"


def _format_sleep_prompt(memories: List[str], source: str = "",
                         prompt_vars: Optional[dict] = None) -> Optional[str]:
    """Render the operator's sleep prompt, or None for the built-in prompt.

    Placeholders: {source}, {memories} (bullet list), {memory_count},
    {principal}, {principal_card}, {profile}, {session_kind}, {date_range};
    unknown ones render empty. A render error returns None (built-in prompt),
    so a broken template never strands claimed rows.
    """
    if prompt_vars is not None and _TEMPLATE_KEYS["sleep"] in prompt_vars:
        template = prompt_vars[_TEMPLATE_KEYS["sleep"]]
    else:
        template = _sleep_prompt_template("sleep")
    if not template:
        return None
    values = {**PROMPT_VAR_DEFAULTS, **(prompt_vars or {})}
    values.update(source=source, memories=_memory_lines(memories),
                  memory_count=len([m for m in memories if m]))
    return _render_prompt(template, values)


def _build_prompt(memories: List[str], source: str = "", prompt_vars: Optional[dict] = None) -> str:
    """Build a consolidation prompt from a list of memory strings.

    Uses a plain-text instruction format (no special model tokens)
    suitable for both local GGUF models and any LLM. For host LLM
    calls, use :func:`_build_host_prompt` instead.
    """
    custom = _format_sleep_prompt(memories, source=source, prompt_vars=prompt_vars)
    if custom is not None:
        return custom

    header = (
        "Summarize the following memories into 1-3 concise sentences. "
        "Preserve facts, names, preferences, and decisions. Discard fluff."
    )
    if source:
        header += f" Source: {source}."

    prompt = f"/no_think\n{header}\n\n{_memory_lines(memories)}\n\nSummary:"
    return prompt


def _build_host_prompt(memories: List[str], source: str = "", prompt_vars: Optional[dict] = None) -> str:
    """Plain-text consolidation prompt for host LLMs (no local-model tokens).

    The host adapter wraps this string as the user-message content of a
    Chat Completions call; embedding local-model chat-template tokens here
    would degrade output quality on every modern aux provider.
    """
    custom = _format_sleep_prompt(memories, source=source, prompt_vars=prompt_vars)
    if custom is not None:
        return custom

    header = (
        "Summarize the following memories into 1-3 concise sentences. "
        "Preserve facts, names, preferences, and decisions. Discard fluff."
    )
    if source:
        header += f" Source: {source}."

    return f"{header}\n\n{_memory_lines(memories)}"


def _host_backend_will_handle_call() -> bool:
    """True iff the host backend will be the chosen path for an LLM call.

    Used to pick the right context budget at chunk time (HOST_LLM_N_CTX vs
    LLM_N_CTX) and to short-circuit llm_available() for Hermes-only users.
    """
    if not LLM_ENABLED or not HOST_LLM_ENABLED:
        return False
    try:
        from mnemosyne.core.llm_backends import get_host_llm_backend
        return get_host_llm_backend() is not None
    except Exception:
        return False


def _try_host_llm(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
):
    """Attempt the host LLM backend if enabled and registered.

    Returns ``(attempted, text)``:

    - ``(False, None)`` when host is disabled, MNEMOSYNE_LLM_ENABLED is false,
      or no backend is registered. Caller should proceed with the existing
      remote/local fallback chain.
    - ``(True, text-or-None)`` when the backend was called. The ``attempted``
      flag is the sentinel callers use to honor the precedence rule: when
      host is enabled and was attempted, the existing MNEMOSYNE_LLM_BASE_URL
      path MUST be skipped on failure; fall straight to local GGUF, then None.

    See ``docs/hermes-llm-integration.md`` for the full behavior model.
    """
    if not LLM_ENABLED or not HOST_LLM_ENABLED:
        return (False, None)
    try:
        from mnemosyne.core.llm_backends import call_host_llm, get_host_llm_backend
    except Exception:
        return (False, None)
    if get_host_llm_backend() is None:
        return (False, None)
    raw = call_host_llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=HOST_LLM_TIMEOUT,
        provider=HOST_LLM_PROVIDER,
        model=HOST_LLM_MODEL,
    )
    # NB: do NOT run host output through _clean_output(): that helper exists
    # to scrub local-model prompt-template echoes and bulleted prompt repeats
    # from local-model output. Host LLMs (Codex/GPT-class) don't echo our
    # prompt format, AND extract_facts() relies on `- bullet` lines surviving
    # so _parse_facts() can consume them. Just trim whitespace.
    text = raw.strip() if isinstance(raw, str) and raw.strip() else None
    if text:
        text = _sanitize_reasoning_output(text)
    return (True, text)


class _InvalidReasoningOutput:
    """Private marker for a model response that must never be persisted."""

    def __bool__(self) -> bool:
        return False


_INVALID_REASONING_OUTPUT = _InvalidReasoningOutput()


def _is_invalid_reasoning_output(value: object) -> bool:
    """Return whether *value* is an unsafe, malformed reasoning response."""
    return value is _INVALID_REASONING_OUTPUT


NOTHING_DURABLE = "NOTHING_DURABLE"


class _NothingDurable:
    """Private marker: the model judged the notes not worth remembering.

    Falsy like _INVALID_REASONING_OUTPUT, so callers that only test truthiness
    treat it as "no text"; sleep() checks it explicitly so the rows are closed
    instead of retried (a retried group would be a paid call every night)."""

    def __bool__(self) -> bool:
        return False


_NOTHING_DURABLE = _NothingDurable()


def _is_nothing_durable(value: object) -> bool:
    return value is _NOTHING_DURABLE


def _nothing_durable_or(text):
    """Map a bare NOTHING_DURABLE reply to the sentinel; pass anything else through."""
    if isinstance(text, str) and re.fullmatch(r"\W*NOTHING_DURABLE\W*", text, re.I):
        return _NOTHING_DURABLE
    return text


def _sanitize_reasoning_output(text: str):
    """Remove balanced think traces and reject malformed traces fail-closed."""
    if not isinstance(text, str):
        return _INVALID_REASONING_OUTPUT
    tags = list(re.finditer(r"<(/?)think\b[^>]*>", text, flags=re.IGNORECASE))
    depth = 0
    for tag in tags:
        if tag.group(1):
            depth -= 1
            if depth < 0:
                return _INVALID_REASONING_OUTPUT
        else:
            depth += 1
            if depth > 1:
                return _INVALID_REASONING_OUTPUT
    if depth:
        return _INVALID_REASONING_OUTPUT
    return re.sub(
        r"<think\b[^>]*>.*?</think\b[^>]*>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def _clean_output(text: str):
    """Strip assistant tokens and extra whitespace from model output."""
    text = _sanitize_reasoning_output(text)
    if _is_invalid_reasoning_output(text):
        return text
    text = text.replace("<|assistant|>", "").replace("<|user|>", "")
    text = text.replace("</s>", "").strip()
    text = re.sub(r"^(Summarize the following memories.*?[.!?:]\s*)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(Preserve facts.*?[.!?:]\s*)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^Source:.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[-*]\s.*\n", "", text, flags=re.MULTILINE)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English, with safety margin."""
    return max(1, len(text) // 4)


def _prompt_token_budget() -> int:
    """Return usable token budget for memory content (reserves overhead + output).

    Picks the larger HOST_LLM_N_CTX when the host backend will handle the
    call; otherwise the local-model-calibrated LLM_N_CTX. This avoids the
    multi-chunk-summary degradation on 128K-context aux providers.

    NOTE: output_reserve is capped at min(LLM_MAX_TOKENS, n_ctx // 4) so
    that LLM_MAX_TOKENS == n_ctx (the default for MiniCPM5-1B with 2048
    context) doesn't leave a negative budget for memory content. See
    BEAM-benchmark root-cause analysis (May 2026). Summarized output fits
    in 128-256 tokens for consolidation; reserving more than 1/4 of the
    context window for output starves the input side.
    """
    overhead = 80
    n_ctx = HOST_LLM_N_CTX if _host_backend_will_handle_call() else LLM_N_CTX
    output_reserve = min(LLM_MAX_TOKENS, max(128, n_ctx // 4))
    safety_margin = int(n_ctx * 0.2)
    return max(64, n_ctx - overhead - output_reserve - safety_margin)


def chunk_memories_by_budget(memories: List[str], source: str = "",
                             prompt_vars: Optional[dict] = None) -> List[List[str]]:
    """Split memories into chunks that fit within the LLM context window."""
    if not memories:
        return []

    budget = _prompt_token_budget()
    chunks = []
    current_chunk = []
    current_tokens = 0

    # A custom template's size counts against the budget, not the built-in's.
    header = _format_sleep_prompt([], source=source, prompt_vars=prompt_vars)
    if header is None:
        header = (
            "Summarize the following memories into 1-3 concise sentences. "
            "Preserve facts, names, preferences, and decisions. Discard fluff."
        )
        if source:
            header += f" Source: {source}."
    header_tokens = _estimate_tokens(header + "\n\n")

    format_overhead = _estimate_tokens("- \n")
    available = budget - header_tokens

    for memory in memories:
        mem_tokens = _estimate_tokens(memory) + format_overhead
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
    """Check whether any LLM backend (host, remote, or local) is available.

    Returns True for Hermes-only users (no MNEMOSYNE_LLM_BASE_URL, no local
    GGUF) as long as a host backend is registered and enabled — otherwise
    sleep would skip ``summarize_memories()`` before the host path could run.
    """
    global _llm_available
    # 0. Host backend (if a host is registered and the user opted in).
    if _host_backend_will_handle_call():
        return True
    # 1. Remote API: only consider it when LLM is globally enabled.
    if LLM_ENABLED and LLM_BASE_URL:
        return True
    if _llm_available is not None:
        return _llm_available
    _load_llm()
    return bool(_llm_available)


def _is_retryable_status(status_code: int) -> bool:
    """Return True if an HTTP status is worth retrying against a fallback model.

    404/400: the requested model is missing or unrecognized on this endpoint —
    another model name on the same host may exist. 5xx: transient server-side
    failure. 401/403 are NOT retryable: a bad key won't be fixed by swapping
    model names. 429 IS retryable: on OpenAI-compatible gateways and proxies
    the quota is usually attached to the model, not the endpoint, so a
    sibling model on the same host may still have quota remaining (#1000).
    """
    if status_code in (401, 403):
        return False
    if status_code in (404, 400, 429) or 500 <= status_code < 600:
        return True
    return False


_DIAG_ESCAPES = {"\r": "\\r", "\n": "\\n", "\t": "\\t"}
_DIAG_MAX_LEN = 60


def _diag(value) -> str:
    """Render one field from a remote body for a log line.

    The value is whatever the endpoint sent, so it is neither trusted nor
    bounded. Control characters are escaped, because this ends up in a
    WARNING and a reply that could inject line breaks could forge log lines,
    and the text is truncated, because a long field must not bury the rest
    of the diagnostic.
    """
    out = []
    for ch in str(value).replace("\\", "\\\\"):
        if ch in _DIAG_ESCAPES:
            out.append(_DIAG_ESCAPES[ch])
        elif ch.isprintable():
            out.append(ch)
        else:
            out.append(f"\\x{ord(ch):02x}")
    text = "".join(out)
    if len(text) > _DIAG_MAX_LEN:
        text = text[:_DIAG_MAX_LEN] + "..."
    return text


class EmptyAnswer(Exception):
    """A 2xx reply whose answer text is empty.

    Carries what the body said about why. A thinking model that spends the
    whole ``max_tokens`` budget on reasoning comes back ``finish_reason=length``
    with ``reasoning_content`` set and ``content`` empty.

    The attributes keep the raw values for callers; only the message text is
    escaped and bounded, since that is what reaches a log line.
    """

    def __init__(self, finish_reason=None, reasoning_tokens=None, has_reasoning=False):
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens
        self.has_reasoning = has_reasoning
        shown = _diag(finish_reason) if finish_reason else "n/a"
        parts = [f"finish_reason={shown}"]
        if reasoning_tokens is not None:
            parts.append(f"reasoning_tokens={_diag(reasoning_tokens)}")
        parts.append("reasoning_content present, content empty" if has_reasoning
                     else "content empty")
        super().__init__(", ".join(parts))


def _call_remote_llm_with_model(
    prompt: str,
    model: str,
    temperature: float = 0.3,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = LLM_TIMEOUT,
    extra_body: Optional[dict] = None,
) -> "tuple[Optional[str], Optional[int], Optional[Exception]]":
    """Call an OpenAI-compatible endpoint with a specific model name.

    Returns ``(text, status, exc)``:

    - ``(text, None, None)`` on success (text may be None for an empty body).
    - ``(None, status, exc)`` on HTTP/network failure. ``status`` is the HTTP
      status code when available, else None. ``exc`` is the underlying exception
      (HTTPStatusError, ConnectError, TimeoutException, etc.) or None.
    - ``(None, None, exc)`` for non-HTTP failures (JSON decode, malformed body).
    - ``(None, status, EmptyAnswer)`` for a 2xx whose answer text is empty.

    ``extra_body`` is merged into the payload last, so a provider-specific
    key such as a thinking toggle rides the same request.

    Callers (see ``_call_remote_llm``) use ``status`` to decide whether to
    retry against ``LLM_FALLBACK_MODELS``.
    """
    base_url = (base_url or LLM_BASE_URL).rstrip("/")
    if not base_url:
        return (None, None, RuntimeError("MNEMOSYNE_LLM_BASE_URL is empty"))
    api_key = api_key if api_key is not None else LLM_API_KEY

    import json

    try:
        import httpx
        has_httpx = True
    except ImportError:
        has_httpx = False

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": application_user_agent(),
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model or "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": temperature,
        "stop": ["</s>", "<|user|>"],
        "stream": False
    }
    if extra_body:
        payload.update(extra_body)

    try:
        if has_httpx:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                status = response.status_code
                if status >= 400:
                    try:
                        response.raise_for_status()
                    except Exception as exc:
                        return (None, status, exc)
                try:
                    data = response.json()
                except Exception as exc:
                    return (None, status, exc)
        else:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status = getattr(resp, "status", 200)
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                return (None, exc.code, exc)
            except Exception as exc:
                return (None, None, exc)

        choices = data.get("choices", []) if isinstance(data, dict) else []
        if choices and choices[0].get("message", {}).get("content"):
            return (choices[0]["message"]["content"], status, None)
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = first.get("message")
        usage = data.get("usage") if isinstance(data, dict) else None
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        return (None, status, EmptyAnswer(
            finish_reason=first.get("finish_reason"),
            reasoning_tokens=details.get("reasoning_tokens") if isinstance(details, dict) else None,
            has_reasoning=bool(isinstance(message, dict) and message.get("reasoning_content")),
        ))
    except Exception as exc:
        return (None, None, exc)


# Most recent remote-call failure, for the caller's fallback WARNING.
# _call_remote_llm_with_model returns the status and exception, but
# _call_remote_llm returns only text or None, so without this the caller
# could report that summarization failed, never why.
_last_llm_failure: Optional[str] = None


def last_llm_failure() -> Optional[str]:
    """Return the most recent remote-call failure as "model: reason", or None."""
    return _last_llm_failure


def _call_remote_llm(prompt: str, temperature: float = 0.3) -> Optional[str]:
    """Call an OpenAI-compatible remote endpoint for summarization.

    ``temperature`` defaults to 0.3 (paraphrase-safe for consolidation);
    callers that need deterministic output (e.g., fact extraction) can
    pass ``temperature=0.0``.

    Tries the primary ``LLM_REMOTE_MODEL`` first; on a retryable error
    (see ``_is_retryable_status``), iterates ``LLM_FALLBACK_MODELS`` in
    order. Returns the first successful response, or ``None`` if every
    model fails (caller falls through to local GGUF / None).
    """
    global _last_llm_failure
    if not LLM_BASE_URL:
        _last_llm_failure = "remote LLM not configured (MNEMOSYNE_LLM_BASE_URL unset)"
        return None

    primary = LLM_REMOTE_MODEL or "local"
    candidates: List[tuple[str, str, str, dict]] = [
        (primary, LLM_BASE_URL, LLM_API_KEY, LLM_EXTRA_BODY)]
    for fb in LLM_FALLBACK_MODELS:
        if fb and fb != primary:
            candidates.append((fb, LLM_FALLBACK_BASE_URL or LLM_BASE_URL,
                               LLM_FALLBACK_API_KEY or LLM_API_KEY, LLM_FALLBACK_EXTRA_BODY))

    for model, base_url, api_key, extra_body in candidates:
        text, status, exc = _call_remote_llm_with_model(
            prompt, model, temperature, base_url=base_url, api_key=api_key,
            extra_body=extra_body,
        )
        if text:
            return text
        if isinstance(exc, EmptyAnswer):
            _last_llm_failure = f"{model}: HTTP {status} with no usable choices ({exc})"
        elif exc is not None:
            _last_llm_failure = (
                f"{model}: {type(exc).__name__}: {str(exc) or repr(exc)} "
                f"(timeout={LLM_TIMEOUT:g}s)"
            )
        else:
            _last_llm_failure = f"{model}: HTTP {status} with no usable choices"
        if status is None:
            continue
        if not _is_retryable_status(status):
            return None
    return None


def _summarize_memories(
    memories: List[str], source: str = "", *, prompt_vars: Optional[dict] = None
):
    """Summarize a batch of working-memory items into a single episodic string.

    Fallback chain:

    0. Host-provided LLM backend, only if MNEMOSYNE_HOST_LLM_ENABLED=true,
       MNEMOSYNE_LLM_ENABLED=true, AND a backend is registered. When this
       path is attempted but produces no usable text, the existing remote
       URL is **skipped** — falls through to local GGUF, then None. This
       prevents accidentally routing memory content to a stale
       MNEMOSYNE_LLM_BASE_URL the user forgot to clear.
    1. Remote OpenAI-compatible API (if MNEMOSYNE_LLM_BASE_URL is set
       AND MNEMOSYNE_LLM_ENABLED is not false).
    2. llama-cpp-python (ARM64 + x86_64 native).
    3. ctransformers (x86_64 only, legacy).
    4. Return None → caller falls back to AAAK encoding.

    A reply of exactly NOTHING_DURABLE (any backend) becomes the
    ``_NOTHING_DURABLE`` sentinel: such chunks are dropped, and the sentinel is
    returned only when every chunk was judged not durable.
    """
    # A failure left by an earlier call must not be reported for this one.
    global _last_llm_failure
    _last_llm_failure = None
    if not memories:
        return None

    # Chunk large memory lists to stay within context window limits.
    # chunk_memories_by_budget() respects LLM_N_CTX and safety margins.
    chunks = chunk_memories_by_budget(memories, source=source, prompt_vars=prompt_vars)

    def _summarize_chunk(chunk_memories: List[str], chunk_source: str = ""):
        return _nothing_durable_or(_summarize_chunk_text(chunk_memories, chunk_source))

    def _summarize_chunk_text(chunk_memories: List[str], chunk_source: str = ""):
        """Summarize a single chunk of memories via the fallback chain."""
        host_prompt = _build_host_prompt(chunk_memories, source=chunk_source, prompt_vars=prompt_vars)
        prompt = _build_prompt(chunk_memories, source=chunk_source, prompt_vars=prompt_vars)

        # 0. Host backend.
        attempted, text = _try_host_llm(host_prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.3)
        if attempted:
            if _is_invalid_reasoning_output(text):
                return _INVALID_REASONING_OUTPUT
            if text:
                return text
            raw = _call_local_llm(prompt)
            if raw:
                cleaned = _clean_output(raw)
                if _is_invalid_reasoning_output(cleaned):
                    return _INVALID_REASONING_OUTPUT
                return cleaned if cleaned else None
            return None

        # 1. Remote API (skip if MNEMOSYNE_FORCE_LOCAL=1 or remote call fails).
        if LLM_ENABLED and LLM_BASE_URL and not os.environ.get("MNEMOSYNE_FORCE_LOCAL", "").lower() in ("1", "true", "yes"):
            raw = _call_remote_llm(prompt)
            if raw:
                cleaned = _clean_output(raw)
                if _is_invalid_reasoning_output(cleaned):
                    return _INVALID_REASONING_OUTPUT
                return cleaned if cleaned else None

        # 2. Local LLM (llama-cpp-python or ctransformers fallback).
        raw = _call_local_llm(prompt)
        if raw:
            cleaned = _clean_output(raw)
            if _is_invalid_reasoning_output(cleaned):
                return _INVALID_REASONING_OUTPUT
            return cleaned if cleaned else None
        return None

    # Summarize each chunk individually.
    chunk_summaries = []
    nothing_durable = 0
    for chunk in chunks:
        summary = _summarize_chunk(chunk, chunk_source=source)
        if _is_invalid_reasoning_output(summary):
            return _INVALID_REASONING_OUTPUT
        if _is_nothing_durable(summary):
            nothing_durable += 1
        elif summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return _NOTHING_DURABLE if chunks and nothing_durable == len(chunks) else None

    # If multiple chunks, do a second-pass summary to consolidate chunk summaries.
    if len(chunk_summaries) > 1:
        final = _summarize_chunk(chunk_summaries, chunk_source=f"{source} [chunked {len(chunks)} parts]")
        if _is_invalid_reasoning_output(final):
            return _INVALID_REASONING_OUTPUT
        return final if final else chunk_summaries[0]

    return chunk_summaries[0]


def summarize_memories(memories: List[str], source: str = "", *,
                       prompt_vars: Optional[dict] = None) -> Optional[str]:
    """Public summary API; malformed reasoning and NOTHING_DURABLE degrade to None."""
    summary = _summarize_memories(memories, source=source, prompt_vars=prompt_vars)
    return summary if isinstance(summary, str) else None
