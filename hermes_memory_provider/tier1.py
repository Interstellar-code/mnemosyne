"""Tier 1 MEMORY.md / USER.md passthrough for the matrix-memory provider.

Read + delete only — NEVER writes content to the files.
Migration path: one-time export of entries into the semantic store.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

_SECTION_DELIMITER = "§"


class Tier1Passthrough:
    """Passthrough reader/deleter for MEMORY.md and USER.md."""

    def __init__(self, hermes_home: str | None = None) -> None:
        if hermes_home is None:
            hermes_home = _resolve_hermes_home()
        self._hermes_home = Path(hermes_home)
        memories_dir = self._hermes_home / "memories"
        self.memory_path: Path = memories_dir / "MEMORY.md"
        self.user_path: Path = memories_dir / "USER.md"
        self._marker_path: Path = self._hermes_home / "matrix-memory" / ".migration-v0.1.done"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def recall(self, query: str, *, limit: int = 50) -> list[dict]:
        """Return entries from MEMORY.md and USER.md that substring-match *query*.

        Each result dict: {tier:1, source:"memory"|"user", entry:<text>, path:<str>}
        """
        results: list[dict] = []
        q_lower = query.lower()

        for path, source_label in (
            (self.memory_path, "memory"),
            (self.user_path, "user"),
        ):
            entries = self._read_entries(path)
            for entry in entries:
                if q_lower in entry.lower():
                    results.append(
                        {
                            "tier": 1,
                            "source": source_label,
                            "entry": entry,
                            "path": str(path),
                        }
                    )
                if len(results) >= limit:
                    return results

        return results

    def forget(self, target: str, kind: str) -> dict:
        """Remove the first entry whose text contains *target* from the given file.

        kind must be "memory" or "user".
        Returns: {status:"deleted"|"not_found", kind:<str>, removed:<str|None>}
        """
        path = self._kind_to_path(kind)
        if path is None or not path.exists():
            return {"status": "not_found", "kind": kind, "removed": None}

        text = path.read_text(encoding="utf-8")
        entries = self.split_entries(text)

        t_lower = target.lower()
        match_idx: Optional[int] = None
        for i, entry in enumerate(entries):
            if t_lower in entry.lower():
                match_idx = i
                break

        if match_idx is None:
            return {"status": "not_found", "kind": kind, "removed": None}

        removed = entries[match_idx]
        remaining = entries[:match_idx] + entries[match_idx + 1 :]

        # Reconstruct file preserving the § style if original used it
        new_text = self._reconstruct(text, remaining)
        timestamp = datetime.now(timezone.utc).isoformat()
        audit_line = f"\n<!-- audit: entry deleted at {timestamp} by matrix-memory -->\n"
        path.write_text(new_text + audit_line, encoding="utf-8")

        return {"status": "deleted", "kind": kind, "removed": removed}

    def migrate_once(self, remember_fn: Callable[..., Any]) -> dict:
        """One-time migration of MEMORY.md + USER.md entries into the semantic store.

        Calls remember_fn(content=<str>, source="semantic") for each entry.
        Idempotent via marker file.  MEMORY.md / USER.md are NOT modified.

        Returns: {status:"migrated"|"skipped", count:<int>, marker:<str>}
        """
        marker = self._marker_path
        if marker.exists():
            return {"status": "skipped", "count": 0, "marker": str(marker)}

        count = 0
        for path in (self.memory_path, self.user_path):
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for entry in self.split_entries(text):
                if entry.strip():
                    remember_fn(content=entry, source="semantic")
                    count += 1

        # Write marker
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            f"migration completed at {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )

        return {"status": "migrated", "count": count, "marker": str(marker)}

    @staticmethod
    def split_entries(text: str) -> list[str]:
        """Split *text* into entries using the § → \\n\\n → whole-file chain."""
        if not text:
            return []

        # Primary: split on lines that start with "§" (section delimiter lines)
        lines = text.split("\n")
        section_line_indices = [
            i for i, line in enumerate(lines) if line.lstrip().startswith(_SECTION_DELIMITER)
        ]

        if section_line_indices:
            # Reconstruct chunks delimited by the § lines (each § line starts a new entry)
            chunks: list[str] = []
            boundaries = section_line_indices + [len(lines)]
            # Text before the first § line (if any) goes into a preamble chunk
            if section_line_indices[0] > 0:
                preamble = "\n".join(lines[: section_line_indices[0]]).strip()
                if preamble:
                    chunks.append(preamble)
            for j, start in enumerate(section_line_indices):
                end = boundaries[j + 1]
                chunk = "\n".join(lines[start:end]).strip()
                if chunk:
                    chunks.append(chunk)
            if chunks:
                return chunks

        # Fallback: split on double-newline
        if "\n\n" in text:
            logger.warning(
                "Tier1 split_entries: no § delimiters found; falling back to \\n\\n split."
            )
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            if chunks:
                return chunks

        # Last resort: whole file as one entry
        stripped = text.strip()
        return [stripped] if stripped else []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _kind_to_path(self, kind: str) -> Optional[Path]:
        if kind == "memory":
            return self.memory_path
        if kind == "user":
            return self.user_path
        return None

    def _read_entries(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        return self.split_entries(text)

    def _reconstruct(self, original: str, remaining_entries: list[str]) -> str:
        """Re-join entries using the same delimiter style as the original."""
        if not remaining_entries:
            return ""

        # Detect delimiter style in original
        lines = original.split("\n")
        used_section = any(line.lstrip().startswith(_SECTION_DELIMITER) for line in lines)

        if used_section:
            return "\n".join(remaining_entries)
        else:
            return "\n\n".join(remaining_entries)


def _resolve_hermes_home() -> str:
    """Return the active Hermes home with profile awareness when available.

    Prefer Hermes core's `get_hermes_home()` when this integration is loaded
    inside Hermes proper. Fall back to the historical env/default behavior when
    the provider is imported standalone outside Hermes.
    """
    try:
        from hermes_constants import get_hermes_home

        return str(get_hermes_home())
    except Exception:
        return os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
