"""LLM-assisted canonical model refresh for Mnemosyne sleep.

This module deliberately does not render or overwrite free-form mental-model
blobs. It asks the sleep LLM for structured candidate updates to canonical
model slots, validates the response, and lets the caller decide whether to
store proposals or apply them.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from mnemosyne.core.filters import _SYSTEM_DERIVED_WRITE_CAPABILITY


DEFAULT_MODEL_CATEGORIES: Set[str] = {
    "model:user",
    "model:workflow",
    "model:project",
    "model:agent",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def sleep_model_refresh_enabled() -> bool:
    """Return whether sleep should run model-refresh inference.

    The default follows sleep itself: if a sleep cycle runs, model refresh runs
    too when an LLM path is available. Deployments can set
    MNEMOSYNE_SLEEP_MODEL_REFRESH_ENABLED=false as an emergency brake.
    """

    return _env_bool("MNEMOSYNE_SLEEP_MODEL_REFRESH_ENABLED", True)


def _allowed_categories_from_env() -> Set[str]:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_CATEGORIES", "").strip()
    if not raw:
        return set(DEFAULT_MODEL_CATEGORIES)
    categories = {part.strip() for part in re.split(r"[,\s]+", raw) if part.strip()}
    return categories or set(DEFAULT_MODEL_CATEGORIES)


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Some models wrap prose around the JSON. Prefer the outermost array/object.
    start_candidates = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if start_candidates:
        start = min(start_candidates)
        end = max(text.rfind("]"), text.rfind("}"))
        if end > start:
            return text[start : end + 1].strip()
    return text


def _coerce_evidence_ids(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    ids: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            ids.append(text)
    return ids


def coerce_confidence(value: Any, default: float) -> float:
    """Return value as a finite float clamped to [0.0, 1.0], or default.

    Confidence reaches this module from two unhardened directions: LLM
    JSON (json.loads round-trips NaN and Infinity, so a model can emit
    them as literals) and persisted metadata_json on legacy banks (any
    JSON type, including strings). Non-numeric and non-finite values
    degrade to the caller's default instead of raising. NaN in
    particular must never survive as a float: it compares False against
    every threshold, so downstream gates of the form
    ``confidence < minimum`` silently pass it. bool is rejected before
    the float attempt: it subclasses int, so ``float(True)`` would
    silently read a JSON ``true`` as full confidence. Finite values are
    clamped to the confidence domain: a persisted ``2.0`` must not
    outrank every in-range proposal or reach the canonical store
    unbounded.
    """
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


def _kebab(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def parse_model_update_proposals(
    raw: str,
    *,
    allowed_categories: Optional[Set[str]] = None,
    principal: Optional[str] = None,
    existing_slots: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Parse and validate LLM candidate updates for canonical model slots.

    Weak-slot guards: a name containing "/" or equal to the principal's full
    or first name is dropped, as is a model:user body that only says who the
    principal is ("The user is Rohit ..."). A name that matches no existing
    "category/name" slot case-insensitively is normalised to kebab-case;
    existing names are kept verbatim so slots never fork. Without
    ``existing_slots`` (unknown) names are left as the model wrote them.
    """

    allowed = allowed_categories or _allowed_categories_from_env()
    principal = (principal or "").strip()
    if principal.lower() in ("", "the user"):
        principal = ""
    principal_names = {principal.lower(), principal.split()[0].lower()} if principal else set()
    identity_body = None
    if principal:
        alts = "|".join(re.escape(p) for p in sorted({principal, principal.split()[0]}, key=len, reverse=True))
        identity_body = re.compile(
            rf"^\s*(?:the\s+user|{alts})(?:'s\s+name)?\s+is\s+(?:called\s+|named\s+)?(?:{alts})\b", re.I)
    existing = None if existing_slots is None else {str(s).lower(): str(s) for s in existing_slots}
    try:
        payload = json.loads(_strip_json_fence(raw))
    except Exception:
        return []
    if isinstance(payload, dict):
        payload = payload.get("proposals") or payload.get("updates") or [payload]
    if not isinstance(payload, list):
        return []

    proposals: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip()
        name = str(item.get("name") or "").strip()
        body = str(item.get("body") or "").strip()
        if not category or not name or not body:
            continue
        if category not in allowed:
            continue
        if "/" in name or name.lower() in principal_names:
            continue
        if identity_body and category == "model:user" and identity_body.search(body):
            continue
        if existing is not None:
            slot = existing.get(f"{category}/{name}".lower())
            name = slot.split("/", 1)[1] if slot is not None else _kebab(name)
            if not name:
                continue
        confidence = coerce_confidence(item.get("confidence", 0.0), 0.0)
        if confidence <= 0.0:
            continue
        evidence_ids = _coerce_evidence_ids(item.get("evidence_ids") or item.get("evidence") or [])
        if not evidence_ids:
            continue
        action = str(item.get("action") or "update").strip().lower()
        if action not in {"update", "keep", "ignore"}:
            action = "update"
        proposals.append({
            "category": category,
            "name": name,
            "body": body,
            "confidence": confidence,
            "evidence_ids": evidence_ids,
            "action": action,
            "reason": str(item.get("reason") or "").strip(),
        })
    return proposals


def _format_memory_lines(items: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in items:
        memory_id = str(item.get("id") or "").strip()
        content = str(item.get("content") or "").strip().replace("\n", " ")
        if memory_id and content:
            lines.append(f"- id={memory_id}: {content}")
    return "\n".join(lines)


# Built-in model-refresh prompt (PROMPTS.md §4.1). Operators override it with
# <profile mnemosyne dir>/model_refresh_prompt.md or sleep_model_refresh_prompt_file.
MODEL_REFRESH_PROMPT = """You maintain the canonical model slots of Hermes, the personal agent of {principal}, during a memory sleep cycle.
Who {principal} is, for recognition only: {principal_card}

Return ONLY a JSON array (a JSON code fence is allowed). Each element has the keys: category, name, body, confidence, evidence_ids, action, reason.
Allowed categories: {categories}
Allowed actions: update (new or changed durable fact), keep (an existing slot is confirmed unchanged), ignore (not durable).
Return [] when nothing durable is present. An empty array is the normal result.

Existing slots (category/name) owned by this profile:
{existing_slots}

What qualifies as a slot body:
- A durable fact that will still be true in months: a stable preference, a standing rule or workflow, how a project or system is set up, how an agent must behave.
- It must be understandable on its own, without the conversation it came from.
- It must add something that is not already in the existing slot of the same name and not already in the description of {principal} above.
- Not durable: task progress, debugging state, one-off requests, deadlines, issue or PR numbers, commit hashes, prices, test or canary messages, anything that reads as a log of what happened today.
- Never secrets, tokens, passwords, keys or account numbers.

Slot naming:
- Prefer updating an existing slot over creating a new one. Reuse the existing name exactly when the subject matches. When you update an existing slot, the body is the complete new text of that slot.
- New names are kebab-case, 1 to 4 words, and name the subject, not the fact: "coding-style", "travel-companions", "backup-policy".
- A model:user name is a facet of {principal}: identity, preferences, family-context, work-context, devices. It is never a person's name and never a path.
- Do not propose any slot whose body only says who {principal} is, where their home directory is, or what their name is. Identity is already known.
- Do not propose a slot about a single tool call, a single file, or one session's outcome.

Evidence and confidence:
- evidence_ids lists the note ids the fact rests on. An update needs at least two notes that state or confirm it, or one explicit standing instruction from {principal} ("always", "from now on", "never", "my preference is").
- confidence 0.9 or above only when {principal} stated the fact explicitly and nothing in the notes contradicts it; 0.6 to 0.8 when it is inferred from behaviour; below 0.6 do not propose it.
- Emit only entries with action "update". A slot that is merely confirmed or a fact that is not durable is omitted, not listed as keep or ignore.
- Family, health and legal matters: factual and neutral, no characterisation of people.

Notes (each line is "id=<note id>: text"; "[USER]" is {principal} unless a "[Name]" tag follows it; "[ASSISTANT]" is Hermes):
{memories}"""


def build_model_refresh_prompt(
    items: Sequence[Dict[str, Any]],
    *,
    allowed_categories: Optional[Iterable[str]] = None,
    prompt_vars: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the operator's model-refresh template, else the built-in one.

    prompt_vars come from local_llm.resolve_prompt_vars (once per sweep); a
    template that does not render falls back to the built-in prompt.
    """
    from mnemosyne.core import local_llm

    categories = sorted(set(allowed_categories or _allowed_categories_from_env()))
    values = {**local_llm.PROMPT_VAR_DEFAULTS, **(prompt_vars or {}),
              "categories": ", ".join(categories), "memories": _format_memory_lines(items)}
    key = local_llm._TEMPLATE_KEYS["model_refresh"]
    if prompt_vars is not None and key in prompt_vars:
        template = prompt_vars[key]
    else:
        template = local_llm._sleep_prompt_template("model_refresh")
    rendered = local_llm._render_prompt(template, values) if template else None
    return (rendered or local_llm._render_prompt(MODEL_REFRESH_PROMPT, values) or "").strip()


def infer_model_update_proposals(
    items: Sequence[Dict[str, Any]],
    *,
    allowed_categories: Optional[Set[str]] = None,
    prompt_vars: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Ask the configured sleep LLM for canonical model-slot proposals."""

    if not sleep_model_refresh_enabled() or not items:
        return []
    allowed = allowed_categories or _allowed_categories_from_env()
    try:
        from mnemosyne.core import local_llm
    except Exception:
        return []
    prompt = build_model_refresh_prompt(items, allowed_categories=allowed, prompt_vars=prompt_vars)

    raw = None
    attempted_host = False
    try:
        attempted_host, raw = local_llm._try_host_llm(  # internal sibling API used by sleep too
            prompt,
            max_tokens=int(os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_MAX_TOKENS", "2048") or "2048"),
            temperature=float(os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_TEMPERATURE", "0.1") or "0.1"),
        )
    except Exception:
        attempted_host = False
        raw = None

    if raw is None and not attempted_host:
        try:
            raw = local_llm._call_remote_llm(prompt, temperature=0.1)
        except Exception:
            raw = None
    if not raw:
        return []
    pv = prompt_vars or {}
    return parse_model_update_proposals(raw, allowed_categories=allowed,
                                        principal=pv.get("principal"),
                                        existing_slots=pv.get("_slot_names"))


def proposal_to_memory_content(proposal: Dict[str, Any]) -> str:
    """Render a pending model-refresh proposal as a compact working-memory row."""

    return (
        "[MODEL_REFRESH_PROPOSAL] "
        f"{proposal.get('category')}::{proposal.get('name')} "
        f"confidence={proposal.get('confidence')}: {proposal.get('body')}"
    )


PROPOSAL_SOURCE = "sleep_model_refresh_proposal"


def auto_apply_enabled() -> bool:
    """Whether sleep may apply validated proposals immediately.

    This defaults ON because model refresh is a sleep-time automation, not a
    human approval queue. Operators can disable it as an emergency brake with
    MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY=false.

    Resolved through the central hot-reload config (config.yaml > env > default)
    so a maintenance operation observes one consistent value at its boundary.
    When verified Dream mode is active (``dream_active``), auto-apply is forced
    off: Dream owns canonical mutations and the sleep path must not race it by
    applying model-refresh proposals directly.
    """
    from mnemosyne.core.config import get_config

    config = get_config()
    if config.get_bool("dream_active", False):
        return False
    return config.get_bool("sleep_model_refresh_auto_apply", True)


def auto_apply_min_confidence() -> float:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY_MIN_CONFIDENCE", "0.90")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.90


def auto_apply_min_evidence() -> int:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_MIN_EVIDENCE", "2")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 2


def auto_apply_conflict_min_confidence() -> float:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_CONFIDENCE", "0.98")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.98


def auto_apply_conflict_min_evidence() -> int:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_EVIDENCE", "3")
    try:
        return max(auto_apply_min_evidence(), int(raw))
    except (TypeError, ValueError):
        return 3


_EPHEMERAL_RE = re.compile(
    r"(\bpr\s*#?\d+\b|\bissue\s*#?\d+\b|\bcommit\s+[0-9a-f]{7,}\b|"
    r"\b[0-9a-f]{12,}\b|\btemporary\b|\btransient\b|\bone[- ]off\b|"
    r"\bdebugging state\b|\btask progress\b|\bphase\s+\d+\s+done\b|"
    r"\bapi[_-]?key\b|\bpassword\b|\bsecret\b|\btoken\b)",
    re.IGNORECASE,
)


def _is_ephemeral_or_sensitive(metadata: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(metadata.get(key) or "")
        for key in ("category", "name", "body", "reason")
    )
    return bool(_EPHEMERAL_RE.search(haystack))


def prepare_proposal_metadata(proposal: Dict[str, Any], *, source_wm_ids: Sequence[str]) -> Dict[str, Any]:
    """Return persisted metadata for a newly inferred proposal."""

    metadata = dict(proposal)
    metadata.setdefault("action", "update")
    metadata["status"] = "pending"
    metadata["source_wm_ids"] = list(source_wm_ids)
    return metadata


def _proposal_row(beam, proposal_id: str) -> Optional[Dict[str, Any]]:
    row = beam.conn.execute(
        "SELECT id, content, source, metadata_json, timestamp, consolidated_at "
        "FROM working_memory WHERE id = ? AND source = ?",
        (proposal_id, PROPOSAL_SOURCE),
    ).fetchone()
    return dict(row) if row is not None else None


def _load_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        metadata = json.loads(row.get("metadata_json") or "{}")
    except Exception:
        metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def _save_metadata(beam, proposal_id: str, metadata: Dict[str, Any]) -> None:
    beam.conn.execute(
        "UPDATE working_memory SET metadata_json = ? WHERE id = ?",
        (json.dumps(metadata, sort_keys=True), proposal_id),
    )
    beam.conn.commit()


def list_model_refresh_proposals(
    beam,
    *,
    status: str = "pending",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List sleep model-refresh proposals stored in working memory."""

    rows = beam.conn.execute(
        "SELECT id, content, source, metadata_json, timestamp, consolidated_at "
        "FROM working_memory WHERE source = ? ORDER BY timestamp DESC LIMIT ?",
        (PROPOSAL_SOURCE, int(limit)),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        metadata = _load_metadata(item)
        item["metadata"] = metadata
        if status and status != "all" and metadata.get("status", "pending") != status:
            continue
        out.append(item)
    return out


def apply_model_refresh_proposal(
    beam,
    proposal_id: str,
    *,
    owner_id: str = "default",
    validator: str = "",
    auto_applied: bool = False,
) -> Dict[str, Any]:
    """Apply one pending proposal to CanonicalStore and mark it applied."""

    row = _proposal_row(beam, proposal_id)
    if row is None:
        raise ValueError(f"model refresh proposal not found: {proposal_id}")
    metadata = _load_metadata(row)
    if metadata.get("status", "pending") != "pending":
        return {"status": metadata.get("status"), "proposal_id": proposal_id, "metadata": metadata}
    if metadata.get("action", "update") != "update":
        raise ValueError("only update proposals can be applied")
    for key in ("category", "name", "body"):
        if not str(metadata.get(key) or "").strip():
            raise ValueError(f"proposal metadata missing {key}")

    from mnemosyne.core.canonical import CanonicalStore

    store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
    canonical = store.remember(
        owner_id,
        metadata["category"],
        metadata["name"],
        metadata["body"],
        source="sleep_model_refresh",
        confidence=coerce_confidence(metadata.get("confidence"), 0.5),
        _write_kind=_SYSTEM_DERIVED_WRITE_CAPABILITY,
    )
    metadata["status"] = "applied"
    metadata["applied_by"] = validator or "system"
    metadata["applied_owner_id"] = owner_id
    metadata["canonical_id"] = canonical.get("id")
    metadata["auto_applied"] = bool(auto_applied)
    _save_metadata(beam, proposal_id, metadata)
    return {"status": "applied", "proposal_id": proposal_id, "canonical": canonical, "metadata": metadata}


def reject_model_refresh_proposal(
    beam,
    proposal_id: str,
    *,
    reason: str = "",
    validator: str = "",
) -> Dict[str, Any]:
    """Reject one pending proposal without touching canonical facts."""

    row = _proposal_row(beam, proposal_id)
    if row is None:
        raise ValueError(f"model refresh proposal not found: {proposal_id}")
    metadata = _load_metadata(row)
    metadata["status"] = "rejected"
    metadata["rejected_by"] = validator or "system"
    metadata["rejection_reason"] = reason
    _save_metadata(beam, proposal_id, metadata)
    return {"status": "rejected", "proposal_id": proposal_id, "metadata": metadata}


def maybe_auto_apply_model_refresh_proposal(
    beam,
    proposal_id: str,
    *,
    owner_id: str = "default",
) -> bool:
    """Validate and automatically resolve one sleep model-refresh proposal.

    Normal product behavior is automated: strong durable candidates are applied;
    weak, ephemeral, unsupported, or unsafe candidates are rejected. Pending rows
    remain only when auto-apply is explicitly disabled by deployment config.
    """

    if not auto_apply_enabled():
        return False
    row = _proposal_row(beam, proposal_id)
    if row is None:
        return False
    metadata = _load_metadata(row)
    if metadata.get("status", "pending") != "pending":
        return False
    if metadata.get("action", "update") != "update":
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="non-update model-refresh action is not durable canonical truth",
            validator="sleep_model_refresh_auto_validation",
        )
        return False
    confidence = coerce_confidence(metadata.get("confidence"), 0.0)

    evidence_ids = [str(x) for x in (metadata.get("evidence_ids") or []) if str(x).strip()]
    source_wm_ids = {str(x) for x in (metadata.get("source_wm_ids") or []) if str(x).strip()}
    if len(evidence_ids) < auto_apply_min_evidence():
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="insufficient evidence for automated model refresh",
            validator="sleep_model_refresh_auto_validation",
        )
        return False
    if source_wm_ids and not set(evidence_ids).issubset(source_wm_ids):
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="proposal cites evidence outside the sleep batch",
            validator="sleep_model_refresh_auto_validation",
        )
        return False
    if _is_ephemeral_or_sensitive(metadata):
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="ephemeral or sensitive content is not eligible for canonical model refresh",
            validator="sleep_model_refresh_auto_validation",
        )
        return False
    if confidence < auto_apply_min_confidence():
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="confidence below automated model-refresh threshold",
            validator="sleep_model_refresh_auto_validation",
        )
        return False

    try:
        from mnemosyne.core.canonical import CanonicalStore
        store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
        current = store.recall(
            owner_id,
            str(metadata.get("category") or ""),
            str(metadata.get("name") or ""),
        )
    except Exception:
        current = None
    if current is not None and str(current.get("body") or "").strip() != str(metadata.get("body") or "").strip():
        if confidence < auto_apply_conflict_min_confidence() or len(evidence_ids) < auto_apply_conflict_min_evidence():
            reject_model_refresh_proposal(
                beam, proposal_id,
                reason="conflicts with current canonical slot without enough supersession evidence",
                validator="sleep_model_refresh_auto_validation",
            )
            return False

    apply_model_refresh_proposal(
        beam,
        proposal_id,
        owner_id=owner_id,
        validator="sleep_model_refresh_auto_apply",
        auto_applied=True,
    )
    return True
