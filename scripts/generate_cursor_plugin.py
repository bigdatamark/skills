#!/usr/bin/env -S uv run
# [MODULE DOC] Generate the Cursor plugin manifest and MCP config from existing repo metadata.
# Behavior: derives .cursor-plugin/plugin.json from .claude-plugin/plugin.json plus discovered
# skills, and .mcp.json from the first MCP server in gemini-extension.json (or a default); writes
# them, or with --check only verifies they are up to date.
# CLI arguments: --check (flag) - do not write; exit 1 listing outdated files if any differ.
# Env vars: none.
# Constants: ROOT (repo root), CLAUDE_PLUGIN_MANIFEST, GEMINI_EXTENSION, CURSOR_PLUGIN_DIR,
# CURSOR_PLUGIN_MANIFEST, CURSOR_MCP_CONFIG (file paths under ROOT); DEFAULT_MCP_SERVER_NAME and
# DEFAULT_MCP_URL (fallback MCP server); PLUGIN_NAME_RE (lowercase alnum with inner dots/hyphens).
# Top-level code: runs main when executed as a script.
# Outputs: writes/validates the two files; prints status; exit code 1 on --check mismatch.
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate Cursor plugin artifacts from existing repo metadata.

Outputs:
- .cursor-plugin/plugin.json
- .mcp.json

Design goals:
- Keep Claude + Cursor metadata in sync.
- Reuse .claude-plugin/plugin.json as primary metadata source.
- Discover skills from skills/*/SKILL.md.
- Reuse MCP URL from gemini-extension.json when available.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLAUDE_PLUGIN_MANIFEST = ROOT / ".claude-plugin" / "plugin.json"
GEMINI_EXTENSION = ROOT / "gemini-extension.json"
CURSOR_PLUGIN_DIR = ROOT / ".cursor-plugin"
CURSOR_PLUGIN_MANIFEST = CURSOR_PLUGIN_DIR / "plugin.json"
CURSOR_MCP_CONFIG = ROOT / ".mcp.json"

DEFAULT_MCP_SERVER_NAME = "huggingface-skills"
DEFAULT_MCP_URL = "https://huggingface.co/mcp?login"

PLUGIN_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")


def load_json(path: Path) -> dict:
    # [FUNCTION DOC] HELPER: Load a required UTF-8 JSON file.
    # Inputs: path - file to read.
    # Outputs: the parsed JSON value (expected to be a mapping).
    # Exceptions: FileNotFoundError with a descriptive message if missing; JSON errors propagate.
    # Side effects: reads the file.
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_frontmatter(text: str) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Parse simple "key: value" lines from a leading frontmatter block.
    # Behavior: finds the first block delimited by "---" lines (searching from a line start anywhere,
    # not only the file start); for each line containing a colon, splits on the first colon and
    # stores the stripped key and value. Lines without a colon are skipped; no YAML semantics.
    # Inputs: text - file content.
    # Outputs: mapping from key to raw string value; empty if no block is found.
    # Exceptions: none.
    # Side effects: none.
    match = re.search(r"^---\s*\n(.*?)\n---\s*", text, re.DOTALL)
    if not match:
        return {}
    data: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def collect_skills() -> list[str]:
    # [FUNCTION DOC] Collect skill names from skills/*/SKILL.md, sorted by file path.
    # Behavior: parses each SKILL.md frontmatter and keeps the non-empty "name" values; files without
    # a name are skipped.
    # Inputs: none.
    # Outputs: list of skill name strings.
    # Exceptions: file read errors propagate.
    # Side effects: reads files under ROOT/skills.
    skills: list[str] = []
    for skill_md in sorted(ROOT.glob("skills/*/SKILL.md")):
        meta = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = meta.get("name", "").strip()
        if not name:
            continue
        skills.append(name)
    return skills


def validate_plugin_name(name: str) -> None:
    # [FUNCTION DOC] HELPER: Validate a plugin name against PLUGIN_NAME_RE.
    # Inputs: name - plugin name.
    # Outputs: none.
    # Exceptions: ValueError with an explanatory message when invalid.
    # Side effects: none.
    if not PLUGIN_NAME_RE.match(name):
        raise ValueError(
            "Invalid plugin name in .claude-plugin/plugin.json: "
            f"'{name}'. Must be lowercase and match {PLUGIN_NAME_RE.pattern}"
        )


def build_cursor_plugin_manifest() -> dict:
    # [FUNCTION DOC] Build the Cursor plugin manifest mapping.
    # Behavior: loads the Claude plugin manifest; requires a non-empty string "name" and validates it;
    # requires at least one discovered skill; starts the manifest with name, "skills" -> "skills" and
    # "mcpServers" -> ".mcp.json"; copies description, version, author, homepage, repository,
    # license, keywords and logo when present in the source.
    # Inputs: none.
    # Outputs: the manifest mapping.
    # Exceptions: FileNotFoundError if the Claude manifest is missing; ValueError for a missing or
    # invalid name or when no skills are found.
    # Side effects: reads files.
    # KNOWN ISSUE: the skill list is only used for the emptiness check; it is not written out.
    src = load_json(CLAUDE_PLUGIN_MANIFEST)

    name = src.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(".claude-plugin/plugin.json must define a non-empty 'name'")
    validate_plugin_name(name)

    skills = collect_skills()
    if not skills:
        raise ValueError("No skills discovered under skills/*/SKILL.md")

    manifest: dict = {"name": name, "skills": "skills", "mcpServers": ".mcp.json"}

    # Copy optional metadata fields when present.
    for key in [
        "description",
        "version",
        "author",
        "homepage",
        "repository",
        "license",
        "keywords",
        "logo",
    ]:
        if key in src:
            manifest[key] = src[key]

    return manifest


def extract_mcp_from_gemini() -> tuple[str, str]:
    # [FUNCTION DOC] Return the MCP server name and URL from gemini-extension.json.
    # Behavior: falls back to the defaults when the file is absent, has no non-empty "mcpServers"
    # mapping, or its first server config is not a mapping. Otherwise uses the first server's name
    # and its "url", else "httpUrl", else the default URL; a non-string or blank URL also becomes the
    # default.
    # Inputs: none.
    # Outputs: a pair (server name, URL).
    # Exceptions: JSON errors propagate.
    # Side effects: reads gemini-extension.json.
    """Return (server_name, url) from gemini-extension when available."""
    if not GEMINI_EXTENSION.exists():
        return DEFAULT_MCP_SERVER_NAME, DEFAULT_MCP_URL

    data = load_json(GEMINI_EXTENSION)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        return DEFAULT_MCP_SERVER_NAME, DEFAULT_MCP_URL

    # Use first configured server as source of truth.
    server_name = next(iter(servers.keys()))
    server_cfg = servers[server_name]
    if not isinstance(server_cfg, dict):
        return DEFAULT_MCP_SERVER_NAME, DEFAULT_MCP_URL

    url = server_cfg.get("url") or server_cfg.get("httpUrl") or DEFAULT_MCP_URL
    if not isinstance(url, str) or not url.strip():
        url = DEFAULT_MCP_URL

    return server_name, url


def build_mcp_config() -> dict:
    # [FUNCTION DOC] Build the .mcp.json mapping with one HTTP MCP server.
    # Inputs: none.
    # Outputs: mapping {"mcpServers": {<name>: {"type": "http", "url": <url>}}}.
    # Exceptions: propagated from extract_mcp_from_gemini.
    # Side effects: reads gemini-extension.json.
    server_name, url = extract_mcp_from_gemini()
    return {
        "mcpServers": {
            server_name: {
                "type": "http",
                "url": url,
            }
        }
    }


def render_json(data: dict) -> str:
    # [FUNCTION DOC] HELPER: Serialize a mapping as 2-space-indented JSON with a trailing newline.
    # Inputs: data - mapping. Outputs: the JSON text (non-ASCII kept).
    # Exceptions: serialization errors propagate.
    # Side effects: none.
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_or_check(path: Path, content: str, check: bool) -> bool:
    # [FUNCTION DOC] HELPER: Write a file if its content differs, or only compare in check mode.
    # Behavior: reads current content (absent file counts as different); identical -> True; in check
    # mode a difference returns False; otherwise creates parent dirs, writes content and returns True.
    # Inputs: path - target file; content - desired text; check - boolean check-only mode.
    # Outputs: boolean "up to date (now)".
    # Exceptions: OS errors propagate.
    # Side effects: may create directories and write the file.
    """Return True when file is already up-to-date (or after writing in non-check mode)."""
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return True

    if check:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    # [FUNCTION DOC] CLI entry point: generate or check the Cursor plugin manifest and MCP config.
    # Behavior: parses --check, renders both files, writes or checks each. In check mode prints the
    # outdated relative paths and a fix hint to stderr and exits 1, or prints an up-to-date message.
    # Otherwise prints "Wrote ..." for both files.
    # Inputs: command-line arguments.
    # Outputs: printed messages; exit code 1 on check failure.
    # Exceptions: SystemExit on check failure; builder errors propagate.
    # Side effects: file writes (non-check mode), printing.
    parser = argparse.ArgumentParser(description="Generate Cursor plugin manifest + MCP config")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generated files are up-to-date without writing changes.",
    )
    args = parser.parse_args()

    plugin_manifest = render_json(build_cursor_plugin_manifest())
    mcp_config = render_json(build_mcp_config())

    ok_plugin = write_or_check(CURSOR_PLUGIN_MANIFEST, plugin_manifest, check=args.check)
    ok_mcp = write_or_check(CURSOR_MCP_CONFIG, mcp_config, check=args.check)

    if args.check:
        outdated = []
        if not ok_plugin:
            outdated.append(str(CURSOR_PLUGIN_MANIFEST.relative_to(ROOT)))
        if not ok_mcp:
            outdated.append(str(CURSOR_MCP_CONFIG.relative_to(ROOT)))

        if outdated:
            print("Generated Cursor artifacts are out of date:", file=sys.stderr)
            for item in outdated:
                print(f"  - {item}", file=sys.stderr)
            print("Run: uv run scripts/generate_cursor_plugin.py", file=sys.stderr)
            sys.exit(1)

        print("Cursor plugin artifacts are up to date.")
        return

    print(f"Wrote {CURSOR_PLUGIN_MANIFEST.relative_to(ROOT)}")
    print(f"Wrote {CURSOR_MCP_CONFIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
