---
name: matrix-memory
description: Three-tier memory discipline for agents using Mnemosyne — routes durable facts, episodic memory, and human-browseable knowledge to the correct tier with dry_run + confirm_token safety discipline.
version: 0.2.0
author: hermes-switch
license: MIT
platforms:
  - hermes
  - codex
  - claude-code
  - cursor
metadata:
  spec_version: "0.2"
  replaces: "mnemosyne-memory-override v0.1"
  tier_count: 3
---

# Matrix-Memory Discipline Skill (v0.2)

This skill defines the binding rules every agent MUST follow when deciding how and
where to store information. Read it once; internalize the routing table; follow the
dry_run discipline on every write.

---

## 1. The Three-Tier Model

There are exactly **three tiers**. `tier=1`, `tier=2`, and `tier=3` are the only
valid values. `tier=4` does NOT exist and is NEVER accepted. (v0.1.0 used a
four-tier model; that model is abolished.)

### Tier 1 — Identity & Profile (MEMORY.md / USER.md)

- **Files**: `MEMORY.md` and `USER.md` (the built-in memory system files).
- **Owner**: The built-in `memory` tool. **Only** the built-in `memory` tool
  writes to Tier 1. The matrix-memory contract does NOT expose a Tier 1 write path.
- **Agent rule**: READ and DELETE only through the contract. Never call
  `memory(action="add"|"replace")` to add new entries on behalf of the contract
  layer. If an entry must be removed, use `mnemosyne_forget(kind="memory"|"user")`.
- **Contents**: Stable user identity, preferences, long-term profile facts that the
  user has explicitly asked to persist (name, timezone, communication style, etc.).
- **Why read-only for agents**: Tier 1 is sacred. Its character limit and simplicity
  are features — it loads fast and always-on. Agents that silently grow it produce
  noise the user cannot easily audit.

### Tier 2 — Episodic + Semantic Memory (Mnemosyne)

- **Backend**: Mnemosyne MCP (`mnemosyne_remember`, `mnemosyne_recall`,
  `mnemosyne_forget`, `mnemosyne_invalidate`, …).
- **Contents**: Durable facts, learned knowledge, project conventions, episodic
  context, entity triples, working scratchpad state.
- **Agent rule**: New durable facts that are NOT identity/profile belong here.
  Call `mnemosyne_remember` with appropriate `importance`, `source`, and `tags`.
- **Advantages over Tier 1**: Full-text + vector search, knowledge-graph triples,
  automatic consolidation, importance-weighted aging.

### Tier 3 — Human-Browseable Wiki (Karpathy Markdown Wiki)

- **Backend**: Wiki MCP (`wiki_create_page`, `wiki_update_page`, `wiki_search`, …).
- **Contents**: Long-form knowledge articles, how-to guides, decision logs, and
  reference pages intended to be read by humans.
- **Relationship to Tier 2**: One-way bridge only. Tier 3 pages are imported into
  Tier 2 at sync time; agents do NOT write to Tier 3 to "update Mnemosyne". If you
  want the wiki and Mnemosyne to agree, update the wiki page and let the bridge sync.
- **Agent rule**: Create or update wiki pages for human-browseable knowledge.
  Never use wiki pages as a substitute for `mnemosyne_remember`.

---

## 2. The Evergreen Test

Before every write, ask:

> **"Is this durable? Would I want it next session?"**

| Answer | Action |
|--------|--------|
| Yes, durable + identity/profile (user asked you to remember it) | Tier 1 via built-in `memory` tool — NOT the contract |
| Yes, durable + searchable knowledge/fact/convention | Tier 2 via `mnemosyne_remember` |
| Yes, durable + long-form human-readable article | Tier 3 via `wiki_create_page` or `wiki_update_page` |
| No, ephemeral session state | Do NOT persist; if needed, use in-memory state only |

If you reach for `memory(action="add")` for a searchable fact, stop. Use
`mnemosyne_remember` instead.

---

## 3. Routing Rules

### Rule R1 — New durable facts

```
mnemosyne_remember(
    content=<fact>,
    importance=<0.7–1.0>,
    source=<"fact"|"preference"|"convention"|"episode">,
    tags=[<relevant>, <tags>]
)
```

Use for: project conventions, learned preferences, episodic context, entity facts.
Do NOT use for: ephemeral todos, session flags, current-task scratchpad (use
`mnemosyne_scratchpad_set` for that).

### Rule R2 — Human-browseable knowledge pages

```
wiki_create_page(title=<title>, content=<markdown>)
wiki_update_page(page_id=<id>, content=<updated_markdown>)
```

Use for: reference articles, how-to guides, architecture decision records, anything
a human would navigate and read rather than search.

### Rule R3 — Tier 1 deletes only

```
mnemosyne_forget(kind="memory", key=<key>)   # removes from MEMORY.md
mnemosyne_forget(kind="user", key=<key>)     # removes from USER.md
```

Never write to Tier 1 through the contract. Deletes are the sole permitted mutation.

### Rule R4 — Recall before writing

Always query before writing. Avoid duplicates:

```
mnemosyne_recall(query=<what you're about to write>, top_k=5)
```

If an existing memory covers the fact, update it (via `mnemosyne_remember` with
the same `source` and higher `importance`) instead of creating a duplicate.

---

## 4. dry_run + confirm_token Discipline (spec §8)

This is the safety contract for every destructive or write operation.

### Affected operations (require the two-call pattern)

- `mnemosyne_remember` (in agent context)
- `mnemosyne_forget`
- `mnemosyne_invalidate`
- `mnemosyne_import`
- `mnemosyne_sync_pull`
- `mnemosyne_scratchpad_clear`
- `wiki_create_page`
- `wiki_update_page`

### The two-call pattern

**Call 1 — Dry run (always first):**

```
mnemosyne_remember(
    content=<fact>,
    importance=0.8,
    source="fact",
    dry_run=true          # ← mandatory in agent context
)
```

The tool returns a preview and a `confirm_token`. Record the token.

**Call 2 — Confirm (within 5 minutes, single use):**

```
mnemosyne_remember(
    content=<fact>,
    importance=0.8,
    source="fact",
    confirm_token=<token_from_call_1>
)
```

### Rules

1. Every write defaults `dry_run=true` in agent context. Never skip the dry run.
2. The `confirm_token` is single-use. If confirmation fails (expired or used), run
   the dry run again to get a fresh token.
3. The `confirm_token` TTL is **5 minutes**. Do not queue tokens for later use.
4. If the dry-run preview shows unexpected content, abort — do not confirm.
5. Batch related facts into one call where the tool supports it; do not confirm
   partial batches.

### Violation consequence

Skipping `dry_run=true` or reusing a confirm_token is a contract violation. The
binding (Hermes or otherwise) may reject the call. On non-Hermes bindings, the
safety check is advisory (see §5 Trust Boundary), but the two-call pattern is still
required discipline.

---

## 5. Audit Rule

**Every write MUST be logged.**

After each confirmed write, append a one-line entry to `log.md`:

```
[<ISO-8601 timestamp>] <tier> | <operation> | <summary of what was written> | confirm_token=<token>
```

Example:

```
[2026-06-19T08:22:00Z] tier=2 | mnemosyne_remember | Learned user prefers 2-space indent in TS | confirm_token=abc123
[2026-06-19T08:23:00Z] tier=3 | wiki_create_page | Created "TypeScript Style Guide" page | confirm_token=def456
```

If `log.md` does not exist in the working directory, create it before the first
write. Never omit the log entry even for low-importance facts.

---

## 6. Trust Boundary (spec §4 Rule 4)

- **On Hermes**: The contract layer enforces all safety rules at the binding level.
  Tool calls that violate the contract are rejected server-side.
- **On non-Hermes bindings** (Codex, Claude Code, Cursor, etc.): Safety is
  **advisory and best-effort**. The binding does not enforce the contract server-side.
  You MUST follow the two-call pattern and audit rule by discipline alone.
- **Never call Mnemosyne MCP tools directly.** Always route through the contract
  layer (the skill functions / MCP facade). Calling `mnemosyne_remember` directly,
  bypassing the contract, voids the safety guarantees on all bindings.

---

## 7. Quick Reference

| What you want to store | Tool | Tier |
|------------------------|------|------|
| Identity / profile (user asked) | built-in `memory` tool | 1 |
| Durable fact / convention / episode | `mnemosyne_remember` | 2 |
| Long-form human-readable article | `wiki_create_page` / `wiki_update_page` | 3 |
| Delete a Tier 1 entry | `mnemosyne_forget(kind="memory"|"user")` | 1 |
| Delete a Tier 2 memory | `mnemosyne_forget(kind="episodic"|"semantic")` | 2 |
| Delete a Tier 3 page | `wiki_delete_page` | 3 |
| Recall / search | `mnemosyne_recall` | 2 |
| Scratchpad (ephemeral working state) | `mnemosyne_scratchpad_set` | 2 |

---

## 8. Cheat Sheet — Decision Tree

```
New information to store?
│
├─ Is it identity/profile that the USER explicitly asked you to remember?
│   └─ YES → built-in memory tool (Tier 1). NOT the contract.
│
├─ Is it a durable fact, learned knowledge, or episodic context?
│   └─ YES → mnemosyne_remember (Tier 2). Run dry_run first.
│
├─ Is it long-form, human-browseable knowledge (article / guide / ADR)?
│   └─ YES → wiki_create_page or wiki_update_page (Tier 3). Run dry_run first.
│
└─ Is it ephemeral (current task state, temp flags)?
    └─ YES → Do NOT persist. Use in-memory state only.
```

---

## 9. Anti-Patterns (Never Do These)

- **Never** call `memory(action="add"|"replace")` through the contract for searchable facts.
- **Never** skip `dry_run=true` on the first call of a destructive operation.
- **Never** reuse or cache a `confirm_token` across multiple calls.
- **Never** use `tier=4` — it does not exist in this contract.
- **Never** write to Tier 3 (wiki) as a proxy for updating Tier 2 (Mnemosyne).
- **Never** call Mnemosyne MCP tools directly — always go through the contract layer.
- **Never** omit the `log.md` audit entry after a confirmed write.

---

*See also: `mem:mnemosyne-memory-override` — the host skill-hook location, policy
inverted from v0.1 to reflect this three-tier discipline.*
