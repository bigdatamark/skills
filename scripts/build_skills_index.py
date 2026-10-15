#!/usr/bin/env -S uv run
# [MODULE DOC] Build Agent Finder search artifacts (NDJSON documents, AI catalog, manifest).
# Behavior: for every skills/*/SKILL.md creates one whole-skill search document (including text of
# supporting files) plus one document per heading section chunk, then writes them as NDJSON with
# an AI catalog and a manifest into an output directory.
# CLI arguments: --repo-root PATH (default current directory) - skills repo root; --out-dir PATH
# (required) - output directory; --branch NAME - source branch recorded in the manifest (default
# from git, else "main").
# Env vars: none.
# Constants: SCHEMA_VERSION (manifest schema 1); SOURCE_REPO / SOURCE_URL (GitHub identity);
# DEFAULT_INDEX ("hf_skills" index name); MARKETPLACE_PATH (internal marketplace used for version
# and descriptions); MAX_CHARS / OVERLAP_CHARS (section chunk size and overlap);
# MAX_SUPPORTING_CHARS (total supporting text budget per skill); SUPPORTING_SUFFIXES (text file
# extensions included); FRONTMATTER_RE; NON_ID_RE (runs of non-alphanumerics for ids); MARKDOWN (a
# shared markdown parser).
# Top-level code: runs main when executed as a script.
# Outputs: hf-skills.ndjson, ai-catalog.json, manifest.json and _SUCCESS in the output directory.
# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown-it-py==4.2.0", "pyyaml==6.0.3"]
# ///
"""Build Agent Finder search artifacts for Hugging Face Skills."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from markdown_it import MarkdownIt

SCHEMA_VERSION = 1
SOURCE_REPO = "huggingface/skills"
SOURCE_URL = "https://github.com/huggingface/skills.git"
DEFAULT_INDEX = "hf_skills"
MARKETPLACE_PATH = ".claude-plugin/marketplace-internal.json"
MAX_CHARS = 12_000
OVERLAP_CHARS = 800
MAX_SUPPORTING_CHARS = 200_000
SUPPORTING_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
NON_ID_RE = re.compile(r"[^a-z0-9]+")
MARKDOWN = MarkdownIt()


@dataclass(frozen=True)
class Section:
    # [CLASS DOC] Immutable record of one heading-delimited section of a SKILL.md body.
    # Fields: title - heading text (or fallback title); heading_path - list of ancestor heading texts
    # ending with this heading; content - section text including its heading line; ordinal - 0-based
    # index among non-empty sections.
    # Behavior: plain frozen dataclass, no methods.
    title: str
    heading_path: list[str]
    content: str
    ordinal: int


def run_git(repo_root: Path, *args: str) -> str:
    # [FUNCTION DOC] HELPER: Run a git command against a repository and return stripped stdout.
    # Inputs: repo_root - repository path (passed with -C); args - git arguments.
    # Outputs: stdout text, stripped.
    # Exceptions: CalledProcessError on failure; FileNotFoundError if git is missing.
    # Side effects: spawns git; stderr is discarded.
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def git_value(repo_root: Path, *args: str, default: str) -> str:
    # [FUNCTION DOC] HELPER: Run a git command, returning a default when it fails.
    # Inputs: repo_root; args - git arguments; default - keyword-only fallback string.
    # Outputs: git stdout or the default.
    # Exceptions: swallows CalledProcessError and FileNotFoundError.
    # Side effects: spawns git.
    try:
        return run_git(repo_root, *args)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


def utc_now() -> str:
    # [FUNCTION DOC] HELPER: Return the current UTC time as ISO-8601 with seconds precision and a Z suffix.
    # Inputs: none. Outputs: timestamp string. Exceptions: none. Side effects: reads the clock.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_skill(path: Path) -> tuple[dict[str, Any], str]:
    # [FUNCTION DOC] Read a SKILL.md and split it into frontmatter and body.
    # Behavior: if the file starts with a frontmatter block, YAML-parses it (empty -> empty mapping)
    # and returns it with the remaining text; without frontmatter returns an empty mapping and the
    # whole text.
    # Inputs: path - SKILL.md file.
    # Outputs: pair (metadata mapping, body text).
    # Exceptions: ValueError if the frontmatter is not a mapping; YAML and file errors propagate.
    # Side effects: reads the file.
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"Frontmatter must be a mapping: {path}")
    return meta, text[match.end() :]


def clean_heading(raw: str) -> str:
    # [FUNCTION DOC] HELPER: Collapse whitespace runs to single spaces and strip.
    # Inputs: raw - text. Outputs: cleaned text. Exceptions: none. Side effects: none.
    return re.sub(r"\s+", " ", raw).strip()


def heading_text(inline: Any) -> str:
    # [FUNCTION DOC] HELPER: Extract plain heading text from a markdown inline token.
    # Behavior: missing token -> empty string; token without children -> its cleaned content;
    # otherwise concatenates text and inline code content, turns soft/hard breaks into spaces, and
    # includes the content of any other child that has content; then cleans whitespace.
    # Inputs: inline - a markdown-it inline token or nothing.
    # Outputs: heading string.
    # Exceptions: none. Side effects: none.
    if inline is None:
        return ""
    if not inline.children:
        return clean_heading(inline.content)

    parts: list[str] = []
    for child in inline.children:
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
        elif child.content:
            parts.append(child.content)
    return clean_heading("".join(parts))


def split_sections(body: str, fallback_title: str) -> list[Section]:
    # [FUNCTION DOC] Split a markdown body into heading-delimited sections.
    # Behavior: parses the body with the markdown parser; text before the first heading forms a
    # section titled fallback_title with an empty heading path. At each heading token with a source
    # line map, flushes the pending section (from its start line up to the heading line), then pops
    # stack entries at the same or deeper level, pushes this heading, and starts a new section at the
    # heading line. Finally flushes to the end. Empty (whitespace-only) sections are dropped and do
    # not consume an ordinal.
    # Inputs: body - markdown text; fallback_title - title for preamble text.
    # Outputs: list of Section records in document order.
    # Exceptions: none expected. Side effects: none.
    lines = body.splitlines()
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current_title = fallback_title
    current_path: list[str] = []
    current_start = 0
    ordinal = 0

    def flush(end: int) -> None:
        # [FUNCTION DOC] HELPER: Emit the pending section ending before a given line.
        # Behavior: joins lines from the current start to end, strips; if non-empty appends a Section
        # with the current title, a copy of the heading path and the next ordinal, then increments it.
        # Inputs: end - exclusive 0-based line index.
        # Outputs: none.
        # Exceptions: none.
        # Side effects: mutates the enclosing sections list and ordinal counter.
        nonlocal ordinal, current_start
        content = "\n".join(lines[current_start:end]).strip()
        if not content:
            return
        sections.append(
            Section(
                title=current_title,
                heading_path=list(current_path),
                content=content,
                ordinal=ordinal,
            )
        )
        ordinal += 1

    tokens = MARKDOWN.parse(body)
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue

        heading_start = token.map[0]
        flush(heading_start)

        level = int(token.tag[1:])
        inline = tokens[index + 1] if index + 1 < len(tokens) else None
        title = heading_text(inline if inline and inline.type == "inline" else None)
        stack = [(depth, text) for depth, text in stack if depth < level]
        stack.append((level, title))
        current_title = title
        current_path = [text for _, text in stack]
        current_start = heading_start

    flush(len(lines))
    return sections


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    # [FUNCTION DOC] Split text into chunks of at most max_chars with overlap.
    # Behavior: text that fits is returned as one chunk. Otherwise repeatedly takes a window of
    # max_chars from start; if not at the end, moves the end back to the last blank line or newline
    # in the window when that lies past the window's midpoint. Stripped non-empty chunks are kept.
    # The next start is end minus overlap_chars, but always advances by at least one character.
    # Inputs: text; max_chars - positive window size; overlap_chars - overlap between chunks.
    # Outputs: list of chunk strings.
    # Exceptions: none. Side effects: none.
    # KNOWN ISSUE: the boundary uses the larger of the last blank-line and last newline positions,
    # which is always the last newline, so preferring paragraph breaks has no effect.
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def stable_id(rel_path: str, ordinal: int, part: int) -> str:
    # [FUNCTION DOC] HELPER: Build a deterministic id for a section chunk document.
    # Behavior: combines SOURCE_REPO, path, 3-digit ordinal and 3-digit part, lowercases, replaces
    # non-alphanumeric runs with hyphens and trims hyphens.
    # Inputs: rel_path; ordinal; part - integers.
    # Outputs: id string. Exceptions: none. Side effects: none.
    source = f"{SOURCE_REPO}-{rel_path}-{ordinal:03d}-{part:03d}".lower()
    return NON_ID_RE.sub("-", source).strip("-")


def stable_skill_id(skill: str) -> str:
    # [FUNCTION DOC] HELPER: Build a deterministic id for a whole-skill document.
    # Inputs: skill - skill directory name.
    # Outputs: slug of SOURCE_REPO + "-skill-" + skill (as in stable_id).
    # Exceptions: none. Side effects: none.
    source = f"{SOURCE_REPO}-skill-{skill}".lower()
    return NON_ID_RE.sub("-", source).strip("-")


def github_url(rel_path: str) -> str:
    # [FUNCTION DOC] HELPER: GitHub blob URL on main for a repo-relative path.
    # Inputs: rel_path. Outputs: URL string. Exceptions: none. Side effects: none.
    return f"https://github.com/{SOURCE_REPO}/blob/main/{rel_path}"


def github_skill_url(skill: str) -> str:
    # [FUNCTION DOC] HELPER: GitHub tree URL on main for a skill directory.
    # Inputs: skill - directory name. Outputs: URL string. Exceptions: none. Side effects: none.
    return f"https://github.com/{SOURCE_REPO}/tree/main/skills/{skill}"


def raw_github_url(rel_path: str) -> str:
    # [FUNCTION DOC] HELPER: raw.githubusercontent.com URL on main for a repo-relative path.
    # Inputs: rel_path. Outputs: URL string. Exceptions: none. Side effects: none.
    return f"https://raw.githubusercontent.com/{SOURCE_REPO}/main/{rel_path}"


def load_marketplace(repo_root: Path) -> dict[str, Any]:
    # [FUNCTION DOC] Load the internal marketplace JSON, or an empty mapping if absent.
    # Inputs: repo_root.
    # Outputs: parsed mapping.
    # Exceptions: ValueError if the JSON top level is not an object; JSON errors propagate.
    # Side effects: reads the file.
    path = repo_root / MARKETPLACE_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object: {MARKETPLACE_PATH}")
    return data


def marketplace_version(marketplace: dict[str, Any]) -> str | None:
    # [FUNCTION DOC] HELPER: Return the marketplace metadata version if it is a string.
    # Inputs: marketplace - mapping.
    # Outputs: version string or nothing (None).
    # Exceptions: none. Side effects: none.
    metadata = marketplace.get("metadata")
    if not isinstance(metadata, dict):
        return None
    version = metadata.get("version")
    return version if isinstance(version, str) else None


def marketplace_descriptions(marketplace: dict[str, Any]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Map plugin name to description from a marketplace mapping.
    # Behavior: returns empty if "plugins" is not a list; keeps mapping entries with string name and
    # description.
    # Inputs: marketplace - mapping. Outputs: name-to-description mapping.
    # Exceptions: none. Side effects: none.
    plugins = marketplace.get("plugins")
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


def read_supporting_files(skill_dir: Path, repo_root: Path) -> tuple[list[str], str]:
    # [FUNCTION DOC] Collect text of a skill's supporting files within a size budget.
    # Behavior: walks the skill directory recursively in sorted order; considers regular files other
    # than SKILL.md whose lowercase suffix is in SUPPORTING_SUFFIXES; skips files that are not valid
    # UTF-8; stops once the MAX_SUPPORTING_CHARS budget is used up; skips blank files; truncates text
    # to the remaining budget; records the repo-relative path and a "# <path>" titled block.
    # Inputs: skill_dir - skill directory; repo_root - for relative paths.
    # Outputs: pair (list of included relative paths, blocks joined by blank lines).
    # Exceptions: UnicodeDecodeError is swallowed per file; other OS errors propagate.
    # Side effects: reads files.
    # KNOWN ISSUE: ignored directories such as node_modules are not excluded.
    files: list[str] = []
    chunks: list[str] = []
    total = 0

    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file() or path.name == "SKILL.md" or path.suffix.lower() not in SUPPORTING_SUFFIXES:
            continue

        rel_path = path.relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        remaining = MAX_SUPPORTING_CHARS - total
        if remaining <= 0:
            break

        text = text.strip()
        if not text:
            continue

        if len(text) > remaining:
            text = text[:remaining].rstrip()

        files.append(rel_path)
        chunks.append(f"# {rel_path}\n\n{text}")
        total += len(text)

    return files, "\n\n".join(chunks)


def build_documents(repo_root: Path, marketplace: dict[str, Any], updated_at: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Build all search documents for every skill.
    # Behavior: for each skills/*/SKILL.md (sorted): loads frontmatter/body; skill is the directory
    # name; skill_name and description come from frontmatter (falling back to the directory name /
    # empty); marketplace description falls back to the frontmatter description; skill_meta is the
    # frontmatter minus name and description. Emits one "skill" document (ordinal 0, part 0) whose
    # text joins name, descriptions, path, body and supporting content, then one "skill_section"
    # document per chunk of each section, whose text joins name, path, heading path, title and
    # chunk. All documents carry repo, URLs, updated_at and, when known, the marketplace version.
    # Inputs: repo_root; marketplace - mapping; updated_at - timestamp string.
    # Outputs: list of document mappings.
    # Exceptions: propagated from loading/parsing.
    # Side effects: reads files.
    documents: list[dict[str, Any]] = []
    version = marketplace_version(marketplace)
    descriptions = marketplace_descriptions(marketplace)

    for skill_md in sorted((repo_root / "skills").glob("*/SKILL.md")):
        meta, body = load_skill(skill_md)
        rel_path = skill_md.relative_to(repo_root).as_posix()
        skill = skill_md.parent.name
        skill_name = str(meta.get("name") or skill)
        skill_description = str(meta.get("description") or "")
        marketplace_description = descriptions.get(skill, skill_description)
        skill_meta = {k: v for k, v in meta.items() if k not in {"name", "description"}}
        url = github_url(rel_path)
        raw_url = raw_github_url(rel_path)
        supporting_files, supporting_content = read_supporting_files(skill_md.parent, repo_root)

        text = "\n".join(
            value
            for value in [
                skill_name,
                marketplace_description,
                skill_description,
                rel_path,
                body.strip(),
                supporting_content,
            ]
            if value
        )

        document = {
            "id": stable_skill_id(skill),
            "repo": SOURCE_REPO,
            "skill": skill,
            "skill_name": skill_name,
            "skill_description": skill_description,
            "marketplace_description": marketplace_description,
            "skill_meta": skill_meta,
            "path": rel_path,
            "url": url,
            "raw_url": raw_url,
            "kind": "skill",
            "title": skill_name,
            "heading_path": [],
            "content": body.strip(),
            "supporting_files": supporting_files,
            "supporting_content": supporting_content,
            "text": text,
            "ordinal": 0,
            "part": 0,
            "updated_at": updated_at,
        }
        if version:
            document["version"] = version
        documents.append(document)

        for section in split_sections(body, skill_name):
            for part, content in enumerate(chunk_text(section.content, MAX_CHARS, OVERLAP_CHARS)):
                title = section.title or skill_name
                heading_text = " > ".join(section.heading_path)
                section_text = "\n".join(
                    value
                    for value in [skill_name, rel_path, heading_text, title, content]
                    if value
                )
                section_document = {
                    "id": stable_id(rel_path, section.ordinal, part),
                    "repo": SOURCE_REPO,
                    "skill": skill,
                    "skill_name": skill_name,
                    "skill_description": skill_description,
                    "marketplace_description": marketplace_description,
                    "skill_meta": skill_meta,
                    "path": rel_path,
                    "url": url,
                    "raw_url": raw_url,
                    "kind": "skill_section",
                    "title": title,
                    "heading_path": section.heading_path,
                    "content": content,
                    "supporting_files": [],
                    "supporting_content": "",
                    "text": section_text,
                    "ordinal": section.ordinal,
                    "part": part,
                    "updated_at": updated_at,
                }
                if version:
                    section_document["version"] = version
                documents.append(section_document)

    return documents


def build_catalog(
    documents: list[dict[str, Any]],
    marketplace: dict[str, Any],
    updated_at: str,
) -> dict[str, Any]:
    # [FUNCTION DOC] Build the AI catalog mapping with one entry per distinct skill.
    # Behavior: iterates documents keeping the first per skill; each entry has the GitHub tree URL as
    # identifier, display name, media type "application/ai-skill", URL, description (marketplace
    # description else skill description), tags, updatedAt and metadata (source, path, rawUrl); when a
    # version is known adds it and a "version:<v>" tag. Wraps entries with spec version and host info.
    # Inputs: documents; marketplace - mapping; updated_at - timestamp.
    # Outputs: catalog mapping.
    # Exceptions: KeyError if documents lack expected keys. Side effects: none.
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    version = marketplace_version(marketplace)
    descriptions = marketplace_descriptions(marketplace)
    for doc in documents:
        skill = doc["skill"]
        if skill in seen:
            continue
        seen.add(skill)
        description = descriptions.get(skill, doc["skill_description"])
        entry = {
            "identifier": github_skill_url(skill),
            "displayName": doc["skill_name"],
            "mediaType": "application/ai-skill",
            "url": doc["url"],
            "description": description,
            "tags": ["huggingface", "skill"],
            "updatedAt": updated_at,
            "metadata": {
                "source": SOURCE_REPO,
                "path": doc["path"],
                "rawUrl": doc["raw_url"],
            },
        }
        if version:
            entry["version"] = version
            entry["tags"].append(f"version:{version}")
        entries.append(entry)

    return {
        "specVersion": "1.0",
        "host": {
            "displayName": "Hugging Face Skills",
            "identifier": f"https://github.com/{SOURCE_REPO}",
            "documentationUrl": "https://github.com/huggingface/skills",
            "logoUrl": "https://huggingface.co/favicon.ico",
        },
        "entries": entries,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: Write a mapping as 2-space-indented UTF-8 JSON with a trailing newline.
    # Inputs: path; data. Outputs: none.
    # Exceptions: OS/serialization errors propagate. Side effects: overwrites the file.
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_ndjson(path: Path, documents: list[dict[str, Any]]) -> None:
    # [FUNCTION DOC] HELPER: Write documents as newline-delimited JSON with sorted keys.
    # Inputs: path; documents - list of mappings. Outputs: none.
    # Exceptions: OS/serialization errors propagate. Side effects: overwrites the file.
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")


def build_artifacts(repo_root: Path, out_dir: Path, branch: str | None) -> None:
    # [FUNCTION DOC] Build and write all Agent Finder artifacts.
    # Behavior: resolves the repo root and creates the output directory; reads the HEAD commit
    # (default "unknown"), the branch (explicit, else git's current branch, else "main"; a detached
    # "HEAD" becomes the explicit branch or "main") and the HEAD commit date as updated_at (default
    # now); loads the marketplace, builds documents, and writes hf-skills.ndjson, ai-catalog.json,
    # manifest.json (schema, source info, generation time, document count, index name, artifact
    # names) and finally a _SUCCESS file containing the generation time.
    # Inputs: repo_root; out_dir; branch - optional name.
    # Outputs: none.
    # Exceptions: propagated (git failures are defaulted).
    # Side effects: git subprocesses, file reads and writes.
    repo_root = repo_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = git_value(repo_root, "rev-parse", "HEAD", default="unknown")
    source_branch = branch or git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD", default="main")
    if source_branch == "HEAD":
        source_branch = branch or "main"

    updated_at = git_value(repo_root, "show", "-s", "--format=%cI", "HEAD", default=utc_now())
    marketplace = load_marketplace(repo_root)
    documents = build_documents(repo_root, marketplace, updated_at)
    generated_at = utc_now()

    write_ndjson(out_dir / "hf-skills.ndjson", documents)
    write_json(out_dir / "ai-catalog.json", build_catalog(documents, marketplace, updated_at))
    write_json(
        out_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_repo": SOURCE_REPO,
            "source_url": SOURCE_URL,
            "source_branch": source_branch,
            "source_commit": commit,
            "generated_at": generated_at,
            "document_count": len(documents),
            "index": DEFAULT_INDEX,
            "artifacts": {
                "catalog": "ai-catalog.json",
                "documents": "hf-skills.ndjson",
            },
        },
    )
    (out_dir / "_SUCCESS").write_text(generated_at + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC] HELPER: Parse command-line arguments (see module block).
    # Inputs: process arguments. Outputs: parsed namespace.
    # Exceptions: SystemExit on invalid arguments. Side effects: may print usage.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="skills repo root")
    parser.add_argument("--out-dir", type=Path, required=True, help="artifact output directory")
    parser.add_argument("--branch", help="source branch name, default from git or main")
    return parser.parse_args()


def main() -> None:
    # [FUNCTION DOC] Entry point: parse arguments and build the artifacts.
    # Inputs: command line. Outputs: none.
    # Exceptions: propagated. Side effects: as build_artifacts.
    args = parse_args()
    build_artifacts(args.repo_root, args.out_dir, args.branch)


if __name__ == "__main__":
    main()
