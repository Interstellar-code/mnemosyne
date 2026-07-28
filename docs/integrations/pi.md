# Mnemosyne for Pi

[Pi](https://pi.dev/) is a minimal terminal coding harness. This guide shows you how to add Mnemosyne memory to Pi.

## Install

1. Make sure the Mnemosyne CLI is installed:

```bash
pip install mnemosyne-memory
```

2. Install the Pi package globally:

```bash
pi install npm:@mnemosyne-oss/pi-mnemosyne
```

Or add it to `.pi/settings.json` in a project for project-local use:

```json
{
  "packages": [
    "npm:@mnemosyne-oss/pi-mnemosyne"
  ]
}
```

3. Restart or reload Pi (`/reload`).

## What you get

A Pi extension that registers these tools:

| Tool | Purpose |
|------|---------|
| `mnemosyne_remember` | Store a memory |
| `mnemosyne_recall` | Search memories by semantic similarity |
| `mnemosyne_forget` | Delete a memory by ID |
| `mnemosyne_stats` | Show memory statistics |
| `mnemosyne_sleep` | Consolidate old memories into summaries |

Plus a Pi skill (`/skill:mnemosyne`) with best practices and examples.

## Quickstart

After installing, try:

```
Remember that I prefer vitest over jest for testing.
Recall my testing preferences.
```

Or invoke the skill:

```
/skill:mnemosyne
```

## Configuration

The extension uses the default Mnemosyne data directory (`~/.hermes/mnemosyne/data`). Set the `MNEMOSYNE_DATA_DIR` environment variable to change it.

## Source code

<https://github.com/mnemosyne-oss/pi-mnemosyne>
