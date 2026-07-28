"""
hermes_memory_provider/wiki_bridge.py

One-way markdown -> Mnemosyne bridge for matrix-memory wiki pages.

Spec: matrix-memory-mnemosyne-fork.md §7
- Wiki dir: {hermes_home}/matrix-memory/wiki/
- ONE-WAY: markdown is source of truth; bridge writes INTO Mnemosyne only.
- Primary trigger: inline calls from memory_create_page / memory_update_page handlers.
- Secondary trigger: polling thread checks mtimes every 60s for filesystem edits.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Subdirectories created lazily under wiki_dir
_WIKI_SUBDIRS = ("entities", "concepts", "comparisons", "queries", "raw")


class WikiBridge:
    """One-way bridge: wiki markdown -> Mnemosyne memory store.

    Parameters
    ----------
    hermes_home:
        Root directory for Hermes data.  ``wiki_dir`` will be
        ``{hermes_home}/matrix-memory/wiki/``.  If *None*, falls back to the
        ``HERMES_HOME`` environment variable or ``~/.hermes``.
    remember_fn:
        Callable ``(content: str, source: str, tags: list | None = None) -> Any``.
        Called for each wiki page that is bridged.
    graph_link_fn:
        Callable ``(source_id: str, target_id: str, relationship: str) -> Any``.
        Called for each resolved ``[[wikilink]]``.
    poll_interval:
        Seconds between mtime scans in the polling thread.  Default 60.
    """

    def __init__(
        self,
        hermes_home: str | None = None,
        *,
        remember_fn: Callable[..., Any] | None = None,
        graph_link_fn: Callable[..., Any] | None = None,
        poll_interval: float = 60.0,
    ) -> None:
        if hermes_home is None:
            hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
        self._hermes_home = Path(hermes_home)
        self.wiki_dir: Path = self._hermes_home / "matrix-memory" / "wiki"
        self.remember_fn = remember_fn
        self.graph_link_fn = graph_link_fn
        self.poll_interval = poll_interval

        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # mtime cache for polling: path -> last mtime seen
        self._mtimes: dict[Path, float] = {}

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _ensure_wiki_dir(self) -> bool:
        """Create wiki_dir and standard subdirs.  Returns True on success."""
        try:
            self.wiki_dir.mkdir(parents=True, exist_ok=True)
            for sub in _WIKI_SUBDIRS:
                (self.wiki_dir / sub).mkdir(exist_ok=True)
            return True
        except OSError as exc:
            logger.warning("WikiBridge: cannot create wiki dir %s: %s", self.wiki_dir, exc)
            return False

    def _log_path(self) -> Path:
        return self.wiki_dir / "log.md"

    def _append_log(self, line: str) -> None:
        """Append *line* to log.md; silently no-op if wiki_dir is inaccessible."""
        try:
            self._ensure_wiki_dir()
            with self._log_path().open("a", encoding="utf-8") as fh:
                fh.write(line.rstrip("\n") + "\n")
        except OSError as exc:
            logger.debug("WikiBridge: log write failed: %s", exc)

    # ------------------------------------------------------------------
    # Frontmatter parser (dependency-free)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_frontmatter(text: str) -> tuple[dict, str]:
        """Parse ``---`` YAML-style frontmatter from *text*.

        Returns ``(meta, body)`` where *meta* is a dict of parsed key/value
        pairs and *body* is the remaining text with frontmatter stripped.

        Supported value forms:
        - Scalar:  ``key: value``
        - List (inline):  ``tags: [a, b, c]``
        - List (comma-separated):  ``tags: a, b, c``

        If no frontmatter fence is found, returns ``({}, text)``.
        """
        if not text.startswith("---"):
            return {}, text

        # Find closing fence
        rest = text[3:]
        # Allow the opening --- to be on its own line
        if rest.startswith("\n"):
            rest = rest[1:]

        close = rest.find("\n---")
        if close == -1:
            return {}, text

        fm_block = rest[:close]
        body = rest[close + 4:]  # skip \n---
        if body.startswith("\n"):
            body = body[1:]

        meta: dict[str, Any] = {}
        for raw_line in fm_block.splitlines():
            raw_line = raw_line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue
            if ":" not in raw_line:
                continue
            key, _, val = raw_line.partition(":")
            key = key.strip()
            val = val.strip()
            if not key:
                continue

            # Detect inline list  [a, b, c]
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                items = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                meta[key] = items
            elif "," in val:
                # Comma-separated plain list
                items = [v.strip() for v in val.split(",") if v.strip()]
                meta[key] = items
            else:
                meta[key] = val

        return meta, body

    # ------------------------------------------------------------------
    # Wikilink resolver
    # ------------------------------------------------------------------

    def _build_wikilink_map(self) -> dict[str, str]:
        """Walk wiki_dir and build a case-insensitive ``{name -> relpath}`` map."""
        result: dict[str, str] = {}
        if not self.wiki_dir.exists():
            return result
        for md_path in self.wiki_dir.rglob("*.md"):
            if md_path.name == "log.md":
                continue
            try:
                rel = str(md_path.relative_to(self.wiki_dir))
            except ValueError:
                continue
            # Key: stem (filename without .md), lowercased
            stem = md_path.stem.lower()
            result[stem] = rel
            # Also index by full relative path stem for subdirectory pages
            rel_stem = str(Path(rel).with_suffix("")).lower()
            result[rel_stem] = rel
        return result

    def resolve_wikilinks(self, body: str) -> list[tuple[str, str | None]]:
        """Return ``[(name, relpath_or_None), ...]`` for all ``[[...]]`` in *body*.

        If the target page cannot be found, ``relpath`` is ``None``.
        """
        wikilink_re = re.compile(r"\[\[([^\]]+)\]\]")
        page_map = self._build_wikilink_map()
        results: list[tuple[str, str | None]] = []
        for m in wikilink_re.finditer(body):
            name = m.group(1).strip()
            key = name.lower()
            relpath = page_map.get(key)
            results.append((name, relpath))
        return results

    # ------------------------------------------------------------------
    # Core bridge action
    # ------------------------------------------------------------------

    def bridge(self, path: str) -> dict:
        """Bridge one wiki file into Mnemosyne.

        Parameters
        ----------
        path:
            Path relative to *wiki_dir*.

        Returns
        -------
        dict with keys ``indexed`` (bool), ``links`` (int), ``dead_links`` (int).
        """
        abs_path = self.wiki_dir / path
        if not abs_path.exists():
            logger.debug("WikiBridge.bridge: file not found: %s", abs_path)
            return {"indexed": False, "links": 0, "dead_links": 0}

        try:
            text = abs_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("WikiBridge.bridge: cannot read %s: %s", abs_path, exc)
            return {"indexed": False, "links": 0, "dead_links": 0}

        meta, body = self.parse_frontmatter(text)
        tags: list[str] | None = meta.get("tags")  # type: ignore[assignment]
        if isinstance(tags, str):
            tags = [tags]

        indexed = False
        if self.remember_fn is not None:
            try:
                self.remember_fn(content=body, source="wiki", tags=tags)
                indexed = True
            except Exception as exc:
                logger.error("WikiBridge.bridge: remember_fn failed for %s: %s", path, exc)

        # Resolve wikilinks and call graph_link_fn for resolved ones
        links_count = 0
        dead_links_count = 0
        link_results = self.resolve_wikilinks(body)
        for name, relpath in link_results:
            if relpath is not None:
                links_count += 1
                if self.graph_link_fn is not None:
                    try:
                        self.graph_link_fn(path, relpath, "references")
                    except Exception as exc:
                        logger.error(
                            "WikiBridge.bridge: graph_link_fn failed %s->%s: %s",
                            path, relpath, exc,
                        )
            else:
                dead_links_count += 1
                self._append_log(
                    f"[dead_link] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                    f"path={path} link=[[{name}]]"
                )

        self._append_log(
            f"[bridge] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"path={path} indexed={indexed} links={links_count} dead_links={dead_links_count}"
        )

        return {"indexed": indexed, "links": links_count, "dead_links": dead_links_count}

    # ------------------------------------------------------------------
    # Tool handler surfaces
    # ------------------------------------------------------------------

    def create_page(self, path: str, content: str) -> dict:
        """Write a new wiki page and bridge it into Mnemosyne.

        Returns ``{status, path, indexed, links}``.
        """
        self._ensure_wiki_dir()
        abs_path = self.wiki_dir / path
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.error("WikiBridge.create_page: write failed %s: %s", abs_path, exc)
            return {"status": "error", "path": path, "indexed": False, "links": 0}

        result = self.bridge(path)
        return {
            "status": "created",
            "path": path,
            "indexed": result["indexed"],
            "links": result["links"],
        }

    def update_page(self, path: str, find: str, replace: str) -> dict:
        """Edit an existing wiki page (find/replace) then re-bridge.

        Returns ``{status, path, indexed, links}``.
        """
        abs_path = self.wiki_dir / path
        if not abs_path.exists():
            return {"status": "not_found", "path": path, "indexed": False, "links": 0}

        try:
            original = abs_path.read_text(encoding="utf-8")
            updated = original.replace(find, replace, 1)
            abs_path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            logger.error("WikiBridge.update_page: edit failed %s: %s", abs_path, exc)
            return {"status": "error", "path": path, "indexed": False, "links": 0}

        result = self.bridge(path)
        return {
            "status": "updated",
            "path": path,
            "indexed": result["indexed"],
            "links": result["links"],
        }

    def show_page(self, path: str) -> dict:
        """Read-only render of a wiki page.

        Returns ``{status, path, content}``.
        """
        abs_path = self.wiki_dir / path
        if not abs_path.exists():
            return {"status": "not_found", "path": path, "content": ""}
        try:
            content = abs_path.read_text(encoding="utf-8")
            return {"status": "ok", "path": path, "content": content}
        except OSError as exc:
            logger.warning("WikiBridge.show_page: read failed %s: %s", abs_path, exc)
            return {"status": "error", "path": path, "content": ""}

    # ------------------------------------------------------------------
    # Polling thread
    # ------------------------------------------------------------------

    def _scan_once(self) -> None:
        """Scan wiki_dir for .md files whose mtime has advanced; bridge them."""
        if not self.wiki_dir.exists():
            return
        for md_path in self.wiki_dir.rglob("*.md"):
            if md_path.name == "log.md":
                continue
            try:
                mtime = md_path.stat().st_mtime
            except OSError:
                continue
            prev = self._mtimes.get(md_path)
            if prev is None or mtime > prev:
                self._mtimes[md_path] = mtime
                if prev is not None:
                    # Only bridge on change, not on first discovery
                    try:
                        rel = str(md_path.relative_to(self.wiki_dir))
                        self.bridge(rel)
                    except Exception as exc:
                        logger.error("WikiBridge polling scan error: %s", exc)
                else:
                    # Record current mtime on first discovery (no bridge)
                    pass

    def _poll_loop(self) -> None:
        """Daemon thread body: scan, sleep, repeat until stop_event is set."""
        # Initial discovery pass to record baseline mtimes
        if self.wiki_dir.exists():
            for md_path in self.wiki_dir.rglob("*.md"):
                if md_path.name == "log.md":
                    continue
                try:
                    self._mtimes[md_path] = md_path.stat().st_mtime
                except OSError:
                    pass

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.poll_interval)
            if self._stop_event.is_set():
                break
            self._scan_once()

    def start_polling(self) -> None:
        """Spawn the background mtime-scan daemon thread."""
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="wiki-bridge-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def stop_polling(self) -> None:
        """Signal the polling thread to stop and wait for it."""
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None
