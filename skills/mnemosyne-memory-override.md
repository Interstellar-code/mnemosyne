---
name: mnemosyne-memory-override
description: Tier-1 memory tool remains the sole writer of MEMORY.md/USER.md; durable searchable facts prefer Tier-2 mnemosyne_remember. Policy inverted from v0.1.
version: 0.2.0
trigger:
  - memory
  - mnemosyne
platforms:
  - hermes
  - codex
  - claude-code
  - cursor
---

> **Superseded by the matrix-memory contract (v0.2).** This file is retained as the
> host skill-hook location so agents and platform bindings that discover skills by
> path (`skills/mnemosyne-memory-override.md`) continue to load the correct policy.
> The v0.1 override (which banned the built-in memory tool) is **inverted** here —
> the built-in memory tool is now sacred and must NOT be deprecated or bypassed.
> See `skills/matrix-memory/SKILL.md` for the full three-tier discipline.

---

# Mnemosyne Memory Override — v0.2 Policy (Inverted)

## What changed from v0.1

v0.1 banned the built-in `memory` tool and routed everything to `mnemosyne_remember`.
That policy is abolished. Under the matrix-memory contract:

- Tier 1 (MEMORY.md / USER.md) is **sacred** and owned exclusively by the built-in
  `memory` tool.
- The contract does NOT expose a Tier 1 write path. Agents must never add or replace
  Tier 1 entries through the contract layer.
- `mnemosyne_remember` (Tier 2) is preferred for searchable, durable knowledge —
  but this is **advisory discipline**, not a ban on Tier 1.

---

## Current Policy Rules

### Rule 1 — Built-in memory tool is the sole Tier-1 writer

The built-in `memory` tool (writing to MEMORY.md / USER.md) is **not deprecated**.
It remains the correct and only mechanism for persisting user identity, profile data,
and stable preferences that the user has explicitly asked to remember.

Do NOT route Tier-1 writes through `mnemosyne_remember`. Do NOT call
`mnemosyne_forget` as a replacement for `memory(action="remove")` on Tier-1 entries
unless you are deleting via the contract layer (which exposes only deletes, not adds).

### Rule 2 — Durable searchable facts prefer Tier 2

For knowledge that benefits from search, consolidation, or knowledge-graph indexing
(project conventions, learned facts, episodic context), prefer:

```
mnemosyne_remember(content=<fact>, importance=<0.7–1.0>, source=<type>, tags=[...])
```

This is **advisory**: the built-in memory tool is not forbidden, but `mnemosyne_remember`
is the better tool for anything a future session would want to search or recall.

### Rule 3 — Contract does not expose a Tier-1 write path (structural)

The matrix-memory contract intentionally omits a Tier-1 write tool. This is a
structural decision, not a workaround. If you find yourself needing to write to
Tier 1 through the contract, you are misrouting — use the built-in `memory` tool
directly.

### Rule 4 — Tier-1 deletes go through the contract

Removing a Tier-1 entry via the contract layer:

```
mnemosyne_forget(kind="memory", key=<key>)   # removes from MEMORY.md
mnemosyne_forget(kind="user", key=<key>)     # removes from USER.md
```

This is the only Tier-1 mutation the contract exposes.

### Rule 5 — Follow the three-tier discipline for all other routing

For the full routing table, dry_run + confirm_token discipline, audit logging, and
the evergreen test, read:

```
skills/matrix-memory/SKILL.md
```

---

## Summary Routing Table

| What to store | Tool | Tier |
|---------------|------|------|
| Identity / profile (user-requested) | built-in `memory` tool | 1 |
| Durable fact / convention / episode | `mnemosyne_remember` | 2 |
| Long-form human-readable article | `wiki_create_page` / `wiki_update_page` | 3 |
| Delete Tier-1 entry | `mnemosyne_forget(kind="memory"\|"user")` | 1 |

---

*Full discipline: `skills/matrix-memory/SKILL.md`*
