#!/usr/bin/env python3
"""Generate Mnemosyne docs from actual provider code.

Usage:
    python3 scripts/generate-docs.py

Output:
  - src/app/(docs)/api/tool-schema/page.mdx (full route page, overwritten)
  - Config table injected into configuration/page.mdx via marker replacement
  - src/api/.provider-metadata.json (build tooling data)
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hermes_memory_provider import ALL_TOOL_SCHEMAS, MnemosyneMemoryProvider


def _version():
    try:
        import mnemosyne

        return mnemosyne.__version__
    except Exception:
        return "unknown"


def _config_schema():
    p = MnemosyneMemoryProvider.__new__(MnemosyneMemoryProvider)
    return p.get_config_schema()


def _esc(v):
    """Escape angle brackets for MDX."""
    return str(v).replace("<", "&lt;").replace(">", "&gt;")


def _gen_tool_schema(schemas, version):
    lines = []
    lines.append("---")
    lines.append('title: Tool Schema Reference')
    lines.append('description: Auto-generated reference for all Mnemosyne tool schemas.')
    lines.append('generated: true')
    lines.append('source: hermes_memory_provider/__init__.py:568')
    lines.append('code_version: ' + version)
    lines.append('---')
    lines.append('')
    lines.append('This page is **auto-generated** from the actual tool schema definitions')
    lines.append('in the provider code. Do not hand-edit.')
    lines.append('')
    lines.append('There are **' + str(len(schemas)) + '** tools available.')
    lines.append('')
    lines.append('## Tool Index')
    lines.append('')
    for s in schemas:
        name = s.get("function", {}).get("name", s.get("name", "unnamed"))
        desc = s.get("description", "") or s.get("function", {}).get("description", "")
        lines.append('- **`' + name + '`** — ' + desc[:120])
    lines.append('')
    lines.append('## Schema Definitions')
    lines.append('')
    for s in schemas:
        name = s.get("function", {}).get("name", s.get("name", "unnamed"))
        desc = s.get("description", "") or s.get("function", {}).get("description", "")
        lines.append('### ' + name)
        lines.append('')
        if desc:
            lines.append(desc)
            lines.append('')
        lines.append("```json")
        lines.append(json.dumps(s, indent=2))
        lines.append("```")
        lines.append('')
    return '\n'.join(lines)


def _gen_config_table(schema):
    lines = []
    lines.append('| Key | Default | Description |')
    lines.append('|-----|---------|-------------|')
    for e in schema:
        key = e['key']
        d = _esc(e.get('default', ''))
        desc = _esc(e.get('description', ''))
        choices = e.get('choices', [])
        if choices:
            desc += ' Choices: ' + ' | '.join(choices)
        lines.append('| `' + key + '` | `' + d + '` | ' + desc + ' |')
    return '\n'.join(lines)


def _inject_config_table(config_page_path, table):
    marker_open = '{/* GENERATED: config-table */}'
    marker_close = '{/* /GENERATED: config-table */}'

    callout = (
        marker_open + '\n\n'
        '<Callout type="info" title="Auto-generated from code">\n'
        'This table is generated from `MnemosyneMemoryProvider.get_config_schema()`.\n'
        'If the provider adds or changes config keys, this updates automatically.\n'
        '</Callout>\n\n'
        + table + '\n\n'
        + marker_close
    )

    with open(config_page_path, 'r') as f:
        content = f.read()

    if marker_open in content:
        pattern = re.escape(marker_open) + r'.*?' + re.escape(marker_close)
        result = re.sub(pattern, callout, content, count=1, flags=re.DOTALL)
        if result == content:
            # No change needed (already up to date)
            return
    else:
        anchor = 'ignore_patterns:              # Content patterns to skip during remember()\n'
        anchor += '      - "be ACTIVE"               # Skill refinement boilerplate\n'
        anchor += '      - "nothing to change"       # No-op responses\n'
        anchor += '      - "skill.*refined"          # Wildcard match\n'
        anchor += '```\n'
        if anchor not in content:
            print('  ERROR: could not find insertion point in config page')
            print('  (looked for end of YAML example block)')
            sys.exit(1)
        result = content.replace(anchor, anchor + '\n' + callout + '\n\n', 1)

    with open(config_page_path, 'w') as f:
        f.write(result)


def _write_website_mirror(docs_root, version, schema):
    """Write tool schema + config + metadata to the website sibling repo.
    Gracefully skips if the sibling repo doesn't exist (CI runners, bare checkouts).
    """
    if not os.path.isdir(docs_root):
        print(f'  ⚠️  Sibling docs repo not found — skipping website mirror')
        return

    # 1. Tool schema page
    tool_page = os.path.join(
        docs_root, 'app/(docs)', 'api', 'tool-schema', 'page.mdx'
    )
    os.makedirs(os.path.dirname(tool_page), exist_ok=True)
    with open(tool_page, 'w') as f:
        f.write(_gen_tool_schema(ALL_TOOL_SCHEMAS, version))
    print('  Tool schema page (' + str(len(ALL_TOOL_SCHEMAS)) + ' tools)')

    # 2. Config table injection
    config_page = os.path.join(
        docs_root, 'app/(docs)', 'getting-started', 'configuration', 'page.mdx'
    )
    _inject_config_table(config_page, _gen_config_table(schema))
    print('  Config table injected (' + str(len(schema)) + ' keys)')

    # 3. Metadata
    meta = {
        'version': version,
        'tool_count': len(ALL_TOOL_SCHEMAS),
        'config_keys': [e['key'] for e in schema],
        'tool_names': [
            s.get('function', {}).get('name', s.get('name', 'unnamed'))
            for s in ALL_TOOL_SCHEMAS
        ],
        'pypi_package': 'mnemosyne-memory',
        'provider_class': 'MnemosyneMemoryProvider',
    }
    meta_dir = os.path.join(docs_root, 'api')
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, '.provider-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('  Provider metadata')
    print('')


def main():
    version = _version()
    schema = _config_schema()

    # Use absolute paths relative to known repo locations
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_script_dir)

    # Canonical copies: always write to docs/api/ inside this repo
    canonical_root = os.path.join(_repo_root, 'docs', 'api')
    os.makedirs(canonical_root, exist_ok=True)

    tool_schema_dest = os.path.join(canonical_root, 'tool-schema.mdx')
    with open(tool_schema_dest, 'w') as f:
        f.write(_gen_tool_schema(ALL_TOOL_SCHEMAS, version))

    config_dest = os.path.join(canonical_root, 'configuration.mdx')
    with open(config_dest, 'w') as f:
        f.write(_gen_config_table(schema))

    print('  Canonical docs written to docs/api/')

    # Website mirror (sibling repo — skip if unavailable)
    docs_root = os.path.normpath(os.path.join(_repo_root, '..', 'mnemosyne-docs', 'src'))
    _write_website_mirror(docs_root, version, schema)


if __name__ == '__main__':
    main()
