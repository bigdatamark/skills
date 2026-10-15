#!/usr/bin/env -S uv run
# [MODULE DOC] CLI to check, bump and enforce plugin/extension manifest versions.
# Behavior: keeps one semantic version in sync across every plugin/extension manifest of the repo
# and, in CI, requires that version to increase whenever published content changes.
# Constants: ROOT is the repository root (parent of the scripts directory). VERSION_PATHS lists
# (manifest file, key path) pairs holding a version: .claude-plugin/plugin.json "version",
# .claude-plugin/marketplace.json and marketplace-internal.json "metadata.version",
# .cursor-plugin/plugin.json "version", .cursor-plugin/marketplace.json "metadata.version",
# gemini-extension.json "version". BUMP_RELEVANT_PATHS are repo-relative paths (a trailing slash
# means a directory prefix) whose change requires a bump. GENERATED_PATHS are generated files that
# never require a bump on their own. SEMVER_RE matches MAJOR.MINOR.PATCH with an optional
# pre-release/build suffix introduced by a hyphen or plus.
# CLI arguments (subcommands, one required):
#   check - verify all manifest versions are valid semver and identical; exit 0 or 1.
#   bump [major|minor|patch] (default patch) [--set VERSION] - bump the canonical version (from
#     .claude-plugin/plugin.json) or set an explicit one, and write it to every manifest.
#   check-bump BASE_REF - exit 1 if bump-relevant files changed between BASE_REF and HEAD
#     without the canonical version increasing.
#   bump-if-needed BASE_REF [major|minor|patch] - like check-bump but also considers
#     uncommitted/untracked files and bumps all manifests automatically when needed.
# Env vars: none. Top-level code: defines constants; runs main when executed as a script.
# Side effects: reads/writes JSON manifests under ROOT and runs git subprocesses in ROOT.
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Manage plugin/extension manifest versions."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

VERSION_PATHS = [
    (ROOT / ".claude-plugin" / "plugin.json", ("version",)),
    (ROOT / ".claude-plugin" / "marketplace.json", ("metadata", "version")),
    (ROOT / ".claude-plugin" / "marketplace-internal.json", ("metadata", "version")),
    (ROOT / ".cursor-plugin" / "plugin.json", ("version",)),
    (ROOT / ".cursor-plugin" / "marketplace.json", ("metadata", "version")),
    (ROOT / "gemini-extension.json", ("version",)),
]

BUMP_RELEVANT_PATHS = (
    "skills/",
    ".claude-plugin/plugin.json",
    ".claude-plugin/marketplace.json",
    ".claude-plugin/marketplace-internal.json",
    ".cursor-plugin/marketplace.json",
    "gemini-extension.json",
    "scripts/AGENTS_TEMPLATE.md",
    "scripts/generate_agents.py",
    "scripts/generate_cursor_plugin.py",
    "scripts/publish.sh",
)

GENERATED_PATHS = (
    "agentsmd/AGENTS.md",
    "README.md",
    ".claude-plugin/marketplace-internal.json",
    ".cursor-plugin/plugin.json",
    ".mcp.json",
)

SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")


def rel(path: Path) -> str:
    # [FUNCTION DOC] HELPER: Return a path relative to the repo root as a forward-slash string.
    # Behavior: computes the path relative to ROOT and converts it to POSIX form.
    # Inputs: path - an absolute filesystem path located under ROOT.
    # Outputs: the relative POSIX path string.
    # Exceptions: ValueError (from the path library) if the path is not under ROOT.
    # Side effects: none.
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Read a UTF-8 JSON file and return the parsed value.
    # Inputs: path - file to read.
    # Outputs: the parsed JSON value (expected to be an object/mapping).
    # Exceptions: file-not-found / OS errors and JSON decode errors propagate.
    # Side effects: reads the file.
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: Write a mapping to a file as pretty JSON.
    # Behavior: serializes with 2-space indentation, non-ASCII characters kept as-is, then appends a
    # trailing newline and writes the text as UTF-8, overwriting the file.
    # Inputs: path - destination file; data - the mapping to serialize.
    # Outputs: none (returns nothing).
    # Exceptions: OS errors and serialization errors propagate.
    # Side effects: overwrites the file at path.
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    # [FUNCTION DOC] HELPER: Look up a value in nested mappings by a sequence of keys.
    # Behavior: starting at data, descends one key at a time; if the current value is not a mapping or
    # lacks the key, raises KeyError whose message is the keys joined with dots.
    # Inputs: data - root mapping; keys - ordered sequence of string keys.
    # Outputs: the value found at the end of the key path (any type).
    # Exceptions: KeyError when any key along the path is missing.
    # Side effects: none.
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(".".join(keys))
        cur = cur[key]
    return cur


def set_nested(data: dict[str, Any], keys: tuple[str, ...], value: str) -> None:
    # [FUNCTION DOC] HELPER: Set a value in nested mappings, creating intermediate mappings as needed.
    # Behavior: for every key except the last, replaces a missing or non-mapping entry with an empty
    # mapping and descends into it; then assigns value under the last key.
    # Inputs: data - root mapping (mutated in place); keys - non-empty key path; value - string to set.
    # Outputs: none.
    # Exceptions: IndexError if keys is empty.
    # Side effects: mutates data in place (may overwrite non-mapping intermediates).
    cur: Any = data
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def read_versions() -> dict[str, str]:
    # [FUNCTION DOC] Collect the version string from every existing manifest in VERSION_PATHS.
    # Behavior: for each (path, keys) pair, skips files that do not exist, loads the JSON, reads the
    # nested key path, requires the value to be a string, and records it under the label
    # "<relative path>:<dotted keys>".
    # Inputs: none.
    # Outputs: a mapping from location label to version string, in VERSION_PATHS order.
    # Exceptions: TypeError if a version is not a string; KeyError if the key path is missing;
    # JSON/OS errors propagate.
    # Side effects: reads manifest files.
    versions: dict[str, str] = {}
    for path, keys in VERSION_PATHS:
        if not path.exists():
            continue
        value = get_nested(load_json(path), keys)
        if not isinstance(value, str):
            raise TypeError(f"{rel(path)} {'.'.join(keys)} must be a string")
        versions[f"{rel(path)}:{'.'.join(keys)}"] = value
    return versions


def canonical_version() -> str:
    # [FUNCTION DOC] Return the canonical version from .claude-plugin/plugin.json.
    # Inputs: none.
    # Outputs: the value of the top-level "version" key (not type-checked).
    # Exceptions: file/JSON errors propagate; KeyError if "version" is missing.
    # Side effects: reads the manifest file.
    return get_nested(load_json(ROOT / ".claude-plugin" / "plugin.json"), ("version",))


def parse_semver(version: str) -> tuple[int, int, int]:
    # [FUNCTION DOC] Parse a semantic version string into its numeric core.
    # Behavior: matches SEMVER_RE; any pre-release/build suffix is accepted but discarded.
    # Inputs: version - the version string.
    # Outputs: a triple of integers (major, minor, patch).
    # Exceptions: ValueError if the string is not a valid semantic version.
    # Side effects: none.
    # KNOWN ISSUE: because suffixes are dropped, comparisons treat 1.2.3-rc1 and 1.2.3 as equal.
    match = SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"Expected semantic version MAJOR.MINOR.PATCH, got: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump_version(version: str, part: str) -> str:
    # [FUNCTION DOC] Compute the next version from a current version and a bump spec.
    # Behavior: parses the current version; "major" -> (major+1).0.0, "minor" -> major.(minor+1).0,
    # "patch" -> major.minor.(patch+1); any other part must itself be a valid semver and is returned
    # unchanged as an explicit version.
    # Inputs: version - current semver string; part - "major", "minor", "patch" or an explicit semver.
    # Outputs: the new version string.
    # Exceptions: ValueError if version is not semver, or part is neither a keyword nor semver.
    # Side effects: none.
    major, minor, patch = parse_semver(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if not SEMVER_RE.match(part):
        raise ValueError(f"Bump must be major, minor, patch, or explicit semver; got: {part}")
    return part


def set_all_versions(version: str) -> None:
    # [FUNCTION DOC] Write one version into every existing manifest listed in VERSION_PATHS.
    # Behavior: validates the version as semver first, then for each existing manifest loads it, sets
    # the nested key path (creating intermediate objects if needed) and rewrites the file.
    # Inputs: version - the semver string to set.
    # Outputs: none.
    # Exceptions: ValueError for an invalid version; file/JSON errors propagate.
    # Side effects: rewrites manifest files (reformatting them as 2-space-indented JSON).
    parse_semver(version)
    for path, keys in VERSION_PATHS:
        if not path.exists():
            continue
        data = load_json(path)
        set_nested(data, keys, version)
        write_json(path, data)


def check_versions() -> int:
    # [FUNCTION DOC] Check that all manifest versions are valid semver and identical.
    # Behavior: reads all versions; if any is not semver prints them to stderr and returns 1; if more
    # than one distinct value exists prints every location and a fix hint to stderr and returns 1;
    # otherwise prints the shared version (or "none" if no manifests exist) and returns 0.
    # Inputs: none.
    # Outputs: exit code integer 0 (ok) or 1 (problem); messages on stdout/stderr.
    # Exceptions: errors from read_versions propagate.
    # Side effects: reads manifests; prints.
    versions = read_versions()
    bad_semver = [f"{where}={version}" for where, version in versions.items() if not SEMVER_RE.match(version)]
    if bad_semver:
        print("Manifest versions must be semantic versions:", file=sys.stderr)
        for item in bad_semver:
            print(f"  - {item}", file=sys.stderr)
        return 1

    unique = sorted(set(versions.values()))
    if len(unique) > 1:
        print("Plugin/extension manifest versions are out of sync:", file=sys.stderr)
        for where, version in versions.items():
            print(f"  - {where} = {version}", file=sys.stderr)
        print("Run: uv run scripts/plugin_versions.py bump patch", file=sys.stderr)
        return 1

    print(f"Plugin manifest versions are in sync: {unique[0] if unique else 'none'}")
    return 0


def git(*args: str) -> str:
    # [FUNCTION DOC] HELPER: Run a git command in the repo root and return its stripped stdout.
    # Inputs: args - git arguments as separate strings.
    # Outputs: the command's standard output as text with surrounding whitespace removed.
    # Exceptions: subprocess.CalledProcessError on non-zero exit; FileNotFoundError if git is absent.
    # Side effects: spawns a git subprocess with working directory ROOT.
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_lines(*args: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Run a git command and return its non-empty output lines.
    # Inputs: args - git arguments.
    # Outputs: a list of non-empty output lines.
    # Exceptions: same as git.
    # Side effects: spawns a git subprocess.
    out = git(*args)
    return [line for line in out.splitlines() if line]


def changed_paths(base_ref: str, include_worktree: bool = False) -> list[str]:
    # [FUNCTION DOC] List repo paths changed relative to a base ref.
    # Behavior: collects names from a three-dot diff between base_ref and HEAD (changes since the merge
    # base); if include_worktree is true also adds unstaged changes, staged changes and untracked
    # non-ignored files. Deduplicates and sorts.
    # Inputs: base_ref - git ref to compare against; include_worktree - boolean, default false.
    # Outputs: sorted list of repo-relative path strings.
    # Exceptions: subprocess.CalledProcessError if a git command fails (e.g. unknown ref).
    # Side effects: runs git subprocesses.
    paths = set(git_lines("diff", "--name-only", f"{base_ref}...HEAD"))
    if include_worktree:
        paths.update(git_lines("diff", "--name-only"))
        paths.update(git_lines("diff", "--cached", "--name-only"))
        paths.update(git_lines("ls-files", "--others", "--exclude-standard"))
    return sorted(paths)


def requires_bump(path: str) -> bool:
    # [FUNCTION DOC] Decide whether a changed path requires a version bump.
    # Behavior: generated paths never require one; otherwise a path requires a bump if it exactly
    # equals an entry of BUMP_RELEVANT_PATHS or starts with one of its directory-prefix entries
    # (those ending in a slash).
    # Inputs: path - repo-relative POSIX path.
    # Outputs: boolean.
    # Exceptions: none.
    # Side effects: none.
    if path in GENERATED_PATHS:
        return False
    return path in BUMP_RELEVANT_PATHS or any(path.startswith(prefix) for prefix in BUMP_RELEVANT_PATHS if prefix.endswith("/"))


def version_at(ref: str) -> str | None:
    # [FUNCTION DOC] Return the canonical plugin version as it was at a git ref.
    # Behavior: reads .claude-plugin/plugin.json at the given ref via git show, parses it, and returns
    # its "version" if that is a string.
    # Inputs: ref - git ref.
    # Outputs: the version string, or nothing (None) if the file does not exist at that ref or the
    # version is not a string.
    # Exceptions: CalledProcessError from git is caught and turns into None; JSON decode errors and
    # KeyError for a missing "version" key propagate.
    # Side effects: runs a git subprocess.
    try:
        text = git("show", f"{ref}:.claude-plugin/plugin.json")
    except subprocess.CalledProcessError:
        return None
    data = json.loads(text)
    value = get_nested(data, ("version",))
    return value if isinstance(value, str) else None


def check_bump(base_ref: str) -> int:
    # [FUNCTION DOC] CI check: fail when bump-relevant content changed without a version increase.
    # Behavior: finds committed changes since base_ref that require a bump; if none, prints a message
    # and returns 0. Otherwise compares the version at base_ref with the current canonical version:
    # missing base version counts as bumped; unparsable versions print the error and return 1; a
    # strictly greater current version prints "Version bumped" and returns 0; otherwise prints the
    # versions, the relevant paths and a fix hint to stderr and returns 1.
    # Inputs: base_ref - git ref to compare against.
    # Outputs: exit code 0 or 1; messages on stdout/stderr.
    # Exceptions: ValueError from parsing is caught; git/file errors propagate.
    # Side effects: runs git; reads manifests; prints.
    # KNOWN ISSUE: when the base version is missing it prints "Version bumped: None -> X", which is
    # misleading since nothing was compared.
    relevant = [path for path in changed_paths(base_ref) if requires_bump(path)]
    if not relevant:
        print("No version-relevant published content changed.")
        return 0

    before = version_at(base_ref)
    after = canonical_version()
    if before is None:
        print(f"Version bumped: {before} -> {after}")
        return 0

    try:
        before_semver = parse_semver(before)
        after_semver = parse_semver(after)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if after_semver > before_semver:
        print(f"Version bumped: {before} -> {after}")
        return 0

    print("Published plugin content changed without an increased version:", file=sys.stderr)
    print(f"  - version: {before} -> {after}", file=sys.stderr)
    for path in relevant:
        print(f"  - {path}", file=sys.stderr)
    print("Run: uv run scripts/plugin_versions.py bump patch && ./scripts/publish.sh", file=sys.stderr)
    return 1


def bump_if_needed(base_ref: str, part: str = "patch") -> int:
    # [FUNCTION DOC] Automatically bump all manifest versions when relevant content changed.
    # Behavior: like check_bump but includes staged, unstaged and untracked files. If nothing relevant
    # changed, or the base version cannot be found, prints a message and returns 0 unchanged. If the
    # versions cannot be parsed prints the error and returns 1. If the current version is already
    # greater than the base version, prints so and returns 0. Otherwise bumps the base version by part
    # and writes it into every manifest, printing the change.
    # Inputs: base_ref - git ref; part - "major", "minor" or "patch" (default "patch"; an explicit
    # semver is also accepted by bump_version).
    # Outputs: exit code 0 or 1; messages on stdout/stderr.
    # Exceptions: ValueError from parsing is caught; other errors propagate.
    # Side effects: runs git; may rewrite all manifest files.
    relevant = [path for path in changed_paths(base_ref, include_worktree=True) if requires_bump(path)]
    if not relevant:
        print("No version-relevant published content changed; version unchanged.")
        return 0

    before = version_at(base_ref)
    after = canonical_version()
    if before is None:
        print("Base manifest version not found; version unchanged.")
        return 0

    try:
        before_semver = parse_semver(before)
        after_semver = parse_semver(after)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if after_semver > before_semver:
        print(f"Version already bumped: {before} -> {after}")
        return 0

    next_version = bump_version(before, part)
    set_all_versions(next_version)
    print(f"Version bumped: {before} -> {next_version}")
    return 0


def main() -> None:
    # [FUNCTION DOC] CLI entry point: parse subcommand arguments and dispatch.
    # Behavior: builds the argparse parser with subcommands check, bump, check-bump and bump-if-needed
    # (see module block). check/check-bump/bump-if-needed exit with their returned code. bump uses
    # --set when given, else bumps the canonical version by part, writes all manifests and prints
    # the new version.
    # Inputs: command-line arguments.
    # Outputs: none returned; process exit code via SystemExit for the check-style commands.
    # Exceptions: SystemExit (argparse errors and deliberate exits); errors from helpers propagate.
    # Side effects: as per the dispatched command (git, manifest reads/writes, printing).
    # KNOWN ISSUE: with bump --set, the positional part argument is silently ignored.
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="Check that all manifest versions are valid and in sync")

    bump = sub.add_parser("bump", help="Bump or set all manifest versions")
    bump.add_argument("part", choices=["major", "minor", "patch"], nargs="?", default="patch")
    bump.add_argument("--set", dest="set_version", help="Set an explicit semantic version")

    changed = sub.add_parser("check-bump", help="Require a version bump when published content changed")
    changed.add_argument("base_ref")

    auto = sub.add_parser("bump-if-needed", help="Bump manifest versions when published content changed")
    auto.add_argument("base_ref")
    auto.add_argument("part", choices=["major", "minor", "patch"], nargs="?", default="patch")

    args = parser.parse_args()

    if args.cmd == "check":
        raise SystemExit(check_versions())
    if args.cmd == "check-bump":
        raise SystemExit(check_bump(args.base_ref))
    if args.cmd == "bump-if-needed":
        raise SystemExit(bump_if_needed(args.base_ref, args.part))
    if args.cmd == "bump":
        version = args.set_version or bump_version(canonical_version(), args.part)
        set_all_versions(version)
        print(f"Set plugin manifest versions to {version}")
        return


if __name__ == "__main__":
    main()
