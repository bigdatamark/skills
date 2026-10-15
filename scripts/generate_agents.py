#!/usr/bin/env -S uv run
# [MODULE DOC] Generate agentsmd/AGENTS.md, the internal marketplace and the README skills table.
# Behavior: discovers skills from skills/*/SKILL.md frontmatter, renders AGENTS.md from a template,
# regenerates .claude-plugin/marketplace-internal.json, validates the curated client marketplaces
# and the internal marketplace, and rewrites the README table between marker comments.
# CLI arguments: none. Env vars: none.
# Constants: ROOT (repo root); TEMPLATE_PATH (scripts/AGENTS_TEMPLATE.md); OUTPUT_PATH
# (agentsmd/AGENTS.md); INTERNAL_MARKETPLACE_PATH; EXTERNAL_MARKETPLACE_PATHS (Claude and Cursor
# client marketplaces); PLUGIN_MANIFEST_PATH (source of the version); README_PATH;
# REQUIRED_MARKETPLACE_SKILLS (the only skills client marketplaces may and must expose: hf-cli);
# README_TABLE_START / README_TABLE_END (HTML comment markers around the README table).
# Top-level code: runs main when executed as a script.
# Outputs: writes AGENTS.md, marketplace-internal.json and README.md; prints progress; exits 1 on
# validation errors.
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate AGENTS.md and marketplace metadata from SKILL.md frontmatter.

Also validates that client marketplace entries stay curated, validates that
the internal Hub marketplace contains every skill, and updates README.md.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "scripts" / "AGENTS_TEMPLATE.md"
OUTPUT_PATH = ROOT / "agentsmd" / "AGENTS.md"
INTERNAL_MARKETPLACE_PATH = ROOT / ".claude-plugin" / "marketplace-internal.json"
EXTERNAL_MARKETPLACE_PATHS = [
    ROOT / ".claude-plugin" / "marketplace.json",
    ROOT / ".cursor-plugin" / "marketplace.json",
]
PLUGIN_MANIFEST_PATH = ROOT / ".claude-plugin" / "plugin.json"
README_PATH = ROOT / "README.md"
REQUIRED_MARKETPLACE_SKILLS = {"hf-cli"}

# Markers for the auto-generated skills table in README
README_TABLE_START = "<!-- BEGIN_SKILLS_TABLE -->"
README_TABLE_END = "<!-- END_SKILLS_TABLE -->"


def load_template() -> str:
    # [FUNCTION DOC] HELPER: Read the AGENTS.md template text.
    # Inputs: none. Outputs: template string.
    # Exceptions: file errors propagate.
    # Side effects: reads TEMPLATE_PATH.
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def parse_frontmatter(text: str) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Parse a minimal YAML-like frontmatter block without dependencies.
    # Behavior: finds the first block delimited by "---" lines. Walks its lines: lines without a colon
    # are skipped; others split on the first colon into stripped key/value. A value of ">" or "|"
    # starts a block scalar: following lines that are empty or indented are stripped and joined with
    # single spaces (empty ones dropped) until an unindented non-empty line. A value wrapped in
    # matching single or double quotes is unquoted. Later keys overwrite earlier ones.
    # Inputs: text - file content.
    # Outputs: mapping from key to string value; empty if no block.
    # Exceptions: none. Side effects: none.
    # KNOWN ISSUE: "|" literal blocks are folded into one line like ">"; block indicators with
    # modifiers (e.g. ">-") are kept as literal values; nested keys appear as top-level keys.
    """Parse a minimal YAML-ish frontmatter block without external deps."""
    match = re.search(r"^---\s*\n(.*?)\n---\s*", text, re.DOTALL)
    if not match:
        return {}
    data: dict[str, str] = {}
    lines = match.group(1).splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if ":" not in line:
            index += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value in {">", "|"}:
            folded: list[str] = []
            index += 1
            while index < len(lines):
                continuation = lines[index]
                if continuation and not continuation.startswith((" ", "\t")):
                    break
                folded.append(continuation.strip())
                index += 1
            data[key] = " ".join(part for part in folded if part)
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        data[key] = value
        index += 1
    return data


def concise_description(description: str) -> str:
    # [FUNCTION DOC] HELPER: Shorten a description to its first sentence for the README.
    # Behavior: collapses whitespace, then cuts at the first whitespace that follows ".", "!" or "?";
    # returns the whole collapsed text if there is no such break.
    # Inputs: description - text. Outputs: the shortened string.
    # Exceptions: none. Side effects: none.
    """Return a compact README description from activation-oriented frontmatter."""
    description = " ".join(description.split())
    sentence = re.search(r"(?<=[.!?])\s+", description)
    if sentence:
        return description[: sentence.start()]
    return description


def collect_skills() -> list[dict[str, str]]:
    # [FUNCTION DOC] Discover skills from skills/*/SKILL.md frontmatter.
    # Behavior: parses each file; skips files lacking a name or description; records name,
    # description and the skill directory path relative to ROOT; sorts by lowercased name.
    # Inputs: none.
    # Outputs: list of mappings with keys name, description, path.
    # Exceptions: file read errors propagate.
    # Side effects: reads files.
    skills: list[dict[str, str]] = []
    for skill_md in ROOT.glob("skills/*/SKILL.md"):
        meta = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = meta.get("name")
        description = meta.get("description")
        if not name or not description:
            continue
        skills.append(
            {
                "name": name,
                "description": description,
                "path": skill_md.parent.relative_to(ROOT).as_posix(),
            }
        )
    # Keep deterministic order for consistent output
    return sorted(skills, key=lambda s: s["name"].lower())


def render(template: str, skills: list[dict[str, str]]) -> str:
    # [FUNCTION DOC] Render the AGENTS template, expanding the skills loop block.
    # Behavior: replaces every section between the skills loop open and close tags with one copy of
    # its inner text per skill (leading/trailing newlines of the block trimmed), substituting the name,
    # description and path placeholders, copies joined by newlines. Other text is untouched.
    # Inputs: template - template text; skills - list of skill mappings.
    # Outputs: rendered text.
    # Exceptions: KeyError if a skill mapping lacks a used key.
    # Side effects: none.
    """Very small Mustache-like renderer that only supports a single skills loop."""
    def repl(match: re.Match[str]) -> str:
        # [FUNCTION DOC] HELPER: Regex replacement callback producing the repeated block for all skills.
        # Inputs: match - regex match whose first group is the loop body.
        # Outputs: the rendered blocks joined by newlines.
        # Exceptions: KeyError on missing skill keys.
        # Side effects: none (reads skills from the enclosing scope).
        block = match.group(1).strip("\n")
        rendered_blocks = []
        for skill in skills:
            rendered = (
                block.replace("{{name}}", skill["name"])
                .replace("{{description}}", skill["description"])
                .replace("{{path}}", skill["path"])
            )
            rendered_blocks.append(rendered)
        return "\n".join(rendered_blocks)

    # Render loop blocks
    content = re.sub(r"{{#skills}}(.*?){{/skills}}", repl, template, flags=re.DOTALL)
    return content


def load_json(path: Path) -> dict:
    # [FUNCTION DOC] HELPER: Load a JSON file, returning an empty mapping when it does not exist.
    # Inputs: path. Outputs: parsed value or empty mapping.
    # Exceptions: JSON/OS errors propagate.
    # Side effects: reads the file.
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_json(data: dict) -> str:
    # [FUNCTION DOC] HELPER: Serialize to 2-space-indented JSON (non-ASCII kept) plus trailing newline.
    # Inputs: data - mapping. Outputs: JSON text.
    # Exceptions: serialization errors propagate. Side effects: none.
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def load_marketplace(path: Path = INTERNAL_MARKETPLACE_PATH) -> dict:
    # [FUNCTION DOC] HELPER: Load a marketplace JSON file that must exist.
    # Inputs: path - marketplace file (default the internal marketplace).
    # Outputs: parsed mapping.
    # Exceptions: FileNotFoundError if missing; JSON errors propagate.
    # Side effects: reads the file.
    """Load marketplace.json and return parsed structure."""
    if not path.exists():
        raise FileNotFoundError(f"marketplace.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def marketplace_descriptions(path: Path) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Map plugin name to description from a marketplace file.
    # Behavior: loads the file (missing -> empty); if "plugins" is not a list returns empty; keeps
    # entries that are mappings with string name and description.
    # Inputs: path - marketplace file.
    # Outputs: mapping from name to description.
    # Exceptions: JSON errors propagate. Side effects: reads the file.
    data = load_json(path)
    plugins = data.get("plugins", [])
    if not isinstance(plugins, list):
        return {}

    descriptions: dict[str, str] = {}
    for plugin in plugins:
        if not isinstance(plugin, dict):
            continue
        name = plugin.get("name")
        description = plugin.get("description")
        if isinstance(name, str) and isinstance(description, str):
            descriptions[name] = description
    return descriptions


def build_internal_marketplace(skills: list[dict[str, str]]) -> dict:
    # [FUNCTION DOC] Build the full internal marketplace mapping covering every skill.
    # Behavior: reads the plugin manifest version (default "0.0.0") and existing internal descriptions;
    # returns a mapping with fixed name/owner/metadata description, the version, and one plugin per
    # skill with name, source "./<path>", skills "./" and description preserved from the existing
    # internal marketplace or else the concise frontmatter description.
    # Inputs: skills - list of skill mappings.
    # Outputs: the marketplace mapping.
    # Exceptions: JSON errors propagate.
    # Side effects: reads files.
    """Build the full marketplace used by hf skills, MCP, and discovery."""
    plugin_manifest = load_json(PLUGIN_MANIFEST_PATH)
    existing_descriptions = marketplace_descriptions(INTERNAL_MARKETPLACE_PATH)

    return {
        "name": "huggingface-skills",
        "owner": {
            "name": "Hugging Face",
        },
        "metadata": {
            "description": (
                "Full Hugging Face Skills manifest for hf skills, MCP, and discovery. "
                "Client plugin marketplaces use curated manifests."
            ),
            "version": plugin_manifest.get("version", "0.0.0"),
        },
        "plugins": [
            {
                "name": skill["name"],
                "source": f"./{skill['path']}",
                "skills": "./",
                "description": existing_descriptions.get(
                    skill["name"],
                    concise_description(skill["description"]),
                ),
            }
            for skill in skills
        ],
    }


def write_internal_marketplace(skills: list[dict[str, str]]) -> None:
    # [FUNCTION DOC] Write the generated internal marketplace JSON to disk.
    # Inputs: skills - list of skill mappings.
    # Outputs: none.
    # Exceptions: propagated.
    # Side effects: overwrites .claude-plugin/marketplace-internal.json.
    INTERNAL_MARKETPLACE_PATH.write_text(
        render_json(build_internal_marketplace(skills)),
        encoding="utf-8",
    )


def generate_readme_table(skills: list[dict[str, str]]) -> str:
    # [FUNCTION DOC] Generate the README markdown table of skills.
    # Behavior: loads the internal marketplace and indexes plugins by source; for each skill looks up
    # its plugin by "./<path>" and emits a row with the plugin name (else skill name), plugin
    # description (else concise description) and a link to the skill's SKILL.md.
    # Inputs: skills - list of skill mappings.
    # Outputs: table text (header, separator, rows) joined by newlines.
    # Exceptions: FileNotFoundError if the internal marketplace is missing; KeyError if a plugin entry
    # lacks "source".
    # Side effects: reads the marketplace file.
    """Generate the skills table for README.md using marketplace.json names."""
    marketplace = load_marketplace()
    plugins = {p["source"]: p for p in marketplace.get("plugins", [])}

    lines = [
        "| Name | Description | Documentation |",
        "|------|-------------|---------------|",
    ]

    for skill in skills:
        source = f"./{skill['path']}"
        plugin = plugins.get(source, {})
        name = plugin.get("name", skill["name"])
        description = plugin.get("description") or concise_description(skill["description"])
        doc_link = f"[SKILL.md]({skill['path']}/SKILL.md)"
        lines.append(f"| `{name}` | {description} | {doc_link} |")

    return "\n".join(lines)


def update_readme(skills: list[dict[str, str]]) -> bool:
    # [FUNCTION DOC] Replace the README skills table between the marker comments.
    # Behavior: warns to stderr and returns False if README is missing, either marker is missing, or
    # the end marker precedes the start marker. Otherwise replaces everything between the markers with
    # a newline, the generated table and a newline, and writes the file.
    # Inputs: skills - list of skill mappings.
    # Outputs: True if written, False otherwise.
    # Exceptions: propagated from table generation and file I/O.
    # Side effects: rewrites README.md; prints warnings.
    """
    Update the README.md skills table between markers.
    Returns True if the file was updated, False if markers not found.
    """
    if not README_PATH.exists():
        print(f"Warning: README.md not found at {README_PATH}", file=sys.stderr)
        return False

    content = README_PATH.read_text(encoding="utf-8")

    start_idx = content.find(README_TABLE_START)
    end_idx = content.find(README_TABLE_END)

    if start_idx == -1 or end_idx == -1:
        print(
            f"Warning: README.md markers not found. Add {README_TABLE_START} and "
            f"{README_TABLE_END} to enable table generation.",
            file=sys.stderr,
        )
        return False

    if end_idx < start_idx:
        print("Warning: README.md markers are in wrong order.", file=sys.stderr)
        return False

    table = generate_readme_table(skills)
    new_content = (
        content[: start_idx + len(README_TABLE_START)]
        + "\n"
        + table
        + "\n"
        + content[end_idx:]
    )

    README_PATH.write_text(new_content, encoding="utf-8")
    return True


def validate_external_marketplace(skills: list[dict[str, str]], path: Path) -> list[str]:
    # [FUNCTION DOC] Validate a curated client marketplace against discovered skills.
    # Behavior: for each plugin, reports duplicate sources, duplicate names, sources with no SKILL.md,
    # and name mismatches with the skill at that source. Then reports required skills that are missing
    # and any exposed skill not in REQUIRED_MARKETPLACE_SKILLS.
    # Inputs: skills - list of skill mappings; path - marketplace file.
    # Outputs: list of error message strings (empty means valid).
    # Exceptions: FileNotFoundError if the file is missing; AttributeError if a plugin entry is not a
    # mapping.
    # Side effects: reads the file.
    # KNOWN ISSUE: plugin entries and the plugins list are not type-checked, unlike the internal
    # validator, so malformed input crashes instead of producing an error message.
    """
    Validate client marketplace entries against discovered skills.

    Client marketplace manifests intentionally expose a small install-time
    surface. The remaining repo skills are installed through
    `hf skills add <skill-name>` or consumed through skill-aware clients over
    CLI/MCP.
    Returns list of error messages (empty = passed).
    """
    errors: list[str] = []
    marketplace = load_marketplace(path)
    plugins = marketplace.get("plugins", [])

    # Build lookups (normalize paths: skill uses "skills/x", marketplace uses "./skills/x")
    skill_by_source = {f"./{s['path']}": s for s in skills}
    plugin_by_source: dict[str, dict] = {}
    plugin_names: set[str] = set()

    # Check: every marketplace plugin has a corresponding skill with matching name.
    for plugin in plugins:
        source = plugin.get("source")
        name = plugin.get("name")

        if source in plugin_by_source:
            errors.append(f"{path}: duplicate marketplace source '{source}'")
        else:
            plugin_by_source[source] = plugin

        if name in plugin_names:
            errors.append(f"{path}: duplicate marketplace skill name '{name}'")
        else:
            plugin_names.add(name)

        if source not in skill_by_source:
            errors.append(
                f"{path}: marketplace plugin '{name}' at '{source}' has no SKILL.md"
            )
            continue

        skill = skill_by_source[source]
        if name != skill["name"]:
            errors.append(
                f"Name mismatch at '{source}': "
                f"SKILL.md='{skill['name']}', marketplace.json='{name}'"
            )

    missing_required = REQUIRED_MARKETPLACE_SKILLS - plugin_names
    for name in sorted(missing_required):
        errors.append(f"{path}: required marketplace skill '{name}' is missing")

    extra = plugin_names - REQUIRED_MARKETPLACE_SKILLS
    for name in sorted(extra):
        errors.append(f"{path}: client marketplace must not expose '{name}'")

    return errors


def validate_internal_marketplace(skills: list[dict[str, str]]) -> list[str]:
    # [FUNCTION DOC] Validate that the internal marketplace lists every skill once.
    # Behavior: errors if plugins is not a list; per entry reports non-mappings, duplicate sources,
    # sources without SKILL.md and name mismatches; finally reports skills with no entry.
    # Inputs: skills - list of skill mappings.
    # Outputs: list of error message strings.
    # Exceptions: FileNotFoundError if missing; JSON errors propagate.
    # Side effects: reads the file.
    """Validate the internal marketplace includes every discovered skill."""
    errors: list[str] = []
    marketplace = load_marketplace(INTERNAL_MARKETPLACE_PATH)
    plugins = marketplace.get("plugins", [])
    if not isinstance(plugins, list):
        return [f"{INTERNAL_MARKETPLACE_PATH}: plugins must be a list"]

    skill_by_source = {f"./{s['path']}": s for s in skills}
    plugin_by_source: dict[str, dict] = {}

    for plugin in plugins:
        if not isinstance(plugin, dict):
            errors.append(f"{INTERNAL_MARKETPLACE_PATH}: plugin entries must be objects")
            continue

        source = plugin.get("source")
        name = plugin.get("name")
        if source in plugin_by_source:
            errors.append(f"{INTERNAL_MARKETPLACE_PATH}: duplicate source '{source}'")
        else:
            plugin_by_source[source] = plugin

        skill = skill_by_source.get(source)
        if skill is None:
            errors.append(f"{INTERNAL_MARKETPLACE_PATH}: '{name}' at '{source}' has no SKILL.md")
        elif name != skill["name"]:
            errors.append(
                f"{INTERNAL_MARKETPLACE_PATH}: name mismatch at '{source}': "
                f"SKILL.md='{skill['name']}', marketplace='{name}'"
            )

    missing = set(skill_by_source) - set(plugin_by_source)
    for source in sorted(missing):
        errors.append(f"{INTERNAL_MARKETPLACE_PATH}: missing skill at '{source}'")

    return errors


def main() -> None:
    # [FUNCTION DOC] Entry point: generate AGENTS.md and internal marketplace, validate, update README.
    # Behavior: renders and writes AGENTS.md (creating its directory), writes the internal marketplace,
    # collects validation errors from both client marketplaces and the internal one; on errors prints
    # them to stderr and exits 1; otherwise prints success and updates the README table.
    # Inputs: none.
    # Outputs: printed progress; exit code 1 on validation errors.
    # Exceptions: SystemExit on validation failure; other errors propagate.
    # Side effects: writes AGENTS.md, marketplace-internal.json and README.md.
    # KNOWN ISSUE: the internal marketplace is regenerated right before it is validated, so that
    # validation can essentially never fail (except for duplicate names in skill directories).
    template = load_template()
    skills = collect_skills()
    output = render(template, skills)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} with {len(skills)} skills.")
    write_internal_marketplace(skills)
    print(f"Wrote {INTERNAL_MARKETPLACE_PATH} with {len(skills)} skills.")

    # Validate marketplace manifests
    errors = [
        error
        for marketplace_path in EXTERNAL_MARKETPLACE_PATHS
        for error in validate_external_marketplace(skills, marketplace_path)
    ]
    errors.extend(validate_internal_marketplace(skills))
    if errors:
        print("\nMarketplace validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
    print("Marketplace validation passed.")

    # Update README.md skills table
    if update_readme(skills):
        print(f"Updated {README_PATH} skills table.")


if __name__ == "__main__":
    main()
