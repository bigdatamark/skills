#!/usr/bin/env -S uv run
# [MODULE DOC] Build static skills distribution artifacts for bucket publishing.
# Behavior: loads and validates every skills/<name>/SKILL.md, copies all skill files to
# <out>/<name>/, writes a single SKILL.md or a deterministic .tar.gz archive per skill for legacy
# consumers, and writes index.json (legacy), skills.json (SEP-2640 entries with per-file URI,
# SHA-256 digest and size), ai-catalog.json, manifest.json and a _SUCCESS marker.
# CLI arguments: --skills-dir PATH (default "skills"); --out-dir PATH (required; deleted and
# recreated); --uri-prefix PREFIX (default "skill://") used to build resource URIs.
# Env vars: none.
# Constants: FRONTMATTER_RE; SKILL_NAME_RE (lowercase alnum/hyphen, 1-64 chars, no leading or
# trailing hyphen); IGNORE_DIRS (node_modules) and IGNORE_FILES (.DS_Store) excluded from
# packages; SOURCE_REPO; ARCHIVE_MEDIA_TYPE ("application/gzip").
# Top-level code: runs main when executed as a script.
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml==6.0.3"]
# ///
"""Build static skills distribution artifacts.

Single-file skills are emitted as ``<out>/<name>/SKILL.md`` for easy audit.
Multi-file skills are emitted as ``<out>/<name>.tar.gz`` archives to keep the
skill package atomic for legacy consumers. ``index.json`` retains the legacy
archive-oriented format.

``skills.json`` contains the current SEP-2640 skill-entry shape. Each entry has
the URI of its ``SKILL.md``, its complete parsed frontmatter, and a ``resources``
manifest containing the URI, raw-byte SHA-256 digest, and byte size of every file
in the published skill directory. All selected skill files are emitted under
``<out>/<name>/``; the bucket sync workflow publishes that expanded tree and its
manifest together.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
IGNORE_DIRS = {"node_modules"}
IGNORE_FILES = {".DS_Store"}
SOURCE_REPO = "huggingface/skills"
ARCHIVE_MEDIA_TYPE = "application/gzip"


@dataclass(frozen=True)
class Skill:
	# [CLASS DOC] Immutable record describing one validated skill.
	# Fields: name - skill name; description - stripped description; frontmatter - full parsed
	# frontmatter mapping; dir - skill directory; files - sorted list of files to publish.
	# Behavior: plain frozen dataclass, no methods.
	name: str
	description: str
	frontmatter: dict[str, Any]
	dir: Path
	files: list[Path]


def utc_now() -> str:
	# [FUNCTION DOC] HELPER: Current UTC time as ISO-8601 with seconds precision and a Z suffix.
	# Inputs: none. Outputs: timestamp string. Exceptions: none. Side effects: reads the clock.
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_skill_md(path: Path) -> dict[str, Any]:
	# [FUNCTION DOC] Parse the YAML frontmatter of a SKILL.md file.
	# Behavior: requires the file to start with a "---" delimited block; YAML-parses it (empty ->
	# empty mapping) and requires a mapping.
	# Inputs: path - SKILL.md.
	# Outputs: frontmatter mapping.
	# Exceptions: ValueError if frontmatter is missing or not a mapping; YAML/file errors propagate.
	# Side effects: reads the file.
	text = path.read_text(encoding="utf-8")
	match = FRONTMATTER_RE.match(text)
	if not match:
		raise ValueError(f"{path} must start with YAML frontmatter")
	meta = yaml.safe_load(match.group(1)) or {}
	if not isinstance(meta, dict):
		raise ValueError(f"{path} frontmatter must be a mapping")
	return meta


def validate_skill_name(name: str, skill_dir: Path) -> None:
	# [FUNCTION DOC] Validate a skill name and that it matches its directory name.
	# Behavior: rejects names not matching SKILL_NAME_RE or containing a double hyphen, and names that
	# differ from the directory name.
	# Inputs: name; skill_dir - skill directory.
	# Outputs: none.
	# Exceptions: ValueError on violation. Side effects: none.
	if not SKILL_NAME_RE.fullmatch(name) or "--" in name:
		raise ValueError(f"{skill_dir}: invalid skill name {name!r}")
	if name != skill_dir.name:
		raise ValueError(f"{skill_dir}: SKILL.md name {name!r} must match directory name")


def validate_frontmatter(meta: dict[str, Any], skill_md: Path) -> None:
	# [FUNCTION DOC] Validate optional/required frontmatter field types and lengths.
	# Behavior: description must be a string of 1-1024 characters; license, if present, a string;
	# compatibility, if present, a string of 1-500 characters; metadata, if present, a mapping of
	# string keys to string values; allowed-tools, if present, a string.
	# Inputs: meta - frontmatter mapping; skill_md - path used in messages.
	# Outputs: none.
	# Exceptions: ValueError naming the first violated rule. Side effects: none.
	description = meta.get("description")
	if not isinstance(description, str) or not 1 <= len(description) <= 1024:
		raise ValueError(f"{skill_md}: description must contain 1-1024 characters")
	license_value = meta.get("license")
	if license_value is not None and not isinstance(license_value, str):
		raise ValueError(f"{skill_md}: license must be a string")
	compatibility = meta.get("compatibility")
	if compatibility is not None and (
		not isinstance(compatibility, str) or not 1 <= len(compatibility) <= 500
	):
		raise ValueError(f"{skill_md}: compatibility must contain 1-500 characters")
	metadata = meta.get("metadata")
	if metadata is not None and (
		not isinstance(metadata, dict)
		or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items())
	):
		raise ValueError(f"{skill_md}: metadata must map string keys to string values")
	allowed_tools = meta.get("allowed-tools")
	if allowed_tools is not None and not isinstance(allowed_tools, str):
		raise ValueError(f"{skill_md}: allowed-tools must be a string")


def rel_archive_path(skill_dir: Path, path: Path) -> str:
	# [FUNCTION DOC] HELPER: Return a safe forward-slash path of a file relative to its skill dir.
	# Inputs: skill_dir; path - file inside it.
	# Outputs: relative POSIX path.
	# Exceptions: ValueError if the path is absolute-looking or contains a ".." segment; ValueError
	# from the path library if not under skill_dir.
	# Side effects: none.
	rel = path.relative_to(skill_dir).as_posix()
	if rel.startswith("/") or ".." in rel.split("/"):
		raise ValueError(f"{path}: unsafe archive path {rel!r}")
	return rel


def discover_files(skill_dir: Path) -> list[Path]:
	# [FUNCTION DOC] List the publishable files of a skill directory.
	# Behavior: walks recursively in sorted order, skipping anything inside an ignored directory,
	# ignored file names, and symlinks; keeps regular files after checking their relative path is
	# safe.
	# Inputs: skill_dir.
	# Outputs: sorted list of file paths.
	# Exceptions: ValueError for unsafe paths.
	# Side effects: filesystem traversal.
	files: list[Path] = []
	for path in sorted(skill_dir.rglob("*")):
		rel_parts = path.relative_to(skill_dir).parts
		if any(part in IGNORE_DIRS for part in rel_parts):
			continue
		if path.name in IGNORE_FILES:
			continue
		if path.is_symlink():
			continue
		if path.is_file():
			rel_archive_path(skill_dir, path)
			files.append(path)
	return files


def load_skills(skills_dir: Path) -> list[Skill]:
	# [FUNCTION DOC] Load and validate all skills in a skills directory.
	# Behavior: for each <dir>/SKILL.md (sorted) parses frontmatter, requires non-empty string name
	# and non-blank description, validates name and frontmatter, discovers files and requires SKILL.md
	# among them, and builds a Skill record.
	# Inputs: skills_dir.
	# Outputs: list of Skill records.
	# Exceptions: ValueError for any validation failure.
	# Side effects: reads files.
	skills: list[Skill] = []
	for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
		meta = parse_skill_md(skill_md)
		name = meta.get("name")
		description = meta.get("description")
		if not isinstance(name, str) or not name:
			raise ValueError(f"{skill_md}: missing required name")
		if not isinstance(description, str) or not description.strip():
			raise ValueError(f"{skill_md}: missing required description")
		validate_skill_name(name, skill_md.parent)
		validate_frontmatter(meta, skill_md)
		files = discover_files(skill_md.parent)
		if skill_md not in files:
			raise ValueError(f"{skill_md}: SKILL.md was not included in discovered files")
		skills.append(
			Skill(
				name=name,
				description=description.strip(),
				frontmatter=meta,
				dir=skill_md.parent,
				files=files,
			)
		)
	return skills


def sha256_and_size(path: Path) -> tuple[str, int]:
	# [FUNCTION DOC] Hash a file and count its bytes in one streaming pass.
	# Behavior: reads the file in 1 MiB binary chunks, updating a SHA-256 hash and a byte count.
	# Inputs: path.
	# Outputs: pair (hex digest, size in bytes).
	# Exceptions: OS errors propagate. Side effects: reads the file.
	"""Hash and count the same raw bytes in a single streaming pass."""
	h = hashlib.sha256()
	size = 0
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			h.update(chunk)
			size += len(chunk)
	return h.hexdigest(), size


def sha256_hex(path: Path) -> str:
	# [FUNCTION DOC] HELPER: Return only the SHA-256 hex digest of a file.
	# Inputs: path. Outputs: hex digest string.
	# Exceptions: OS errors propagate. Side effects: reads the file.
	return sha256_and_size(path)[0]


def write_skill_md(skill: Skill, out_dir: Path) -> Path:
	# [FUNCTION DOC] Copy a skill's SKILL.md to <out>/<name>/SKILL.md.
	# Inputs: skill; out_dir.
	# Outputs: path of the written file.
	# Exceptions: OS errors propagate.
	# Side effects: creates the directory and writes the file (overwriting).
	target_dir = out_dir / skill.name
	target_dir.mkdir(parents=True, exist_ok=True)
	target = target_dir / "SKILL.md"
	shutil.copyfile(skill.dir / "SKILL.md", target)
	return target


def tar_info(name: str, data: bytes, mode: int) -> tarfile.TarInfo:
	# [FUNCTION DOC] HELPER: Build a normalized tar header for deterministic archives.
	# Behavior: sets the name, size, mode, and zeroes mtime, uid, gid and owner/group names.
	# Inputs: name - member path; data - bytes (for size); mode - permission bits.
	# Outputs: tar header object.
	# Exceptions: none. Side effects: none.
	info = tarfile.TarInfo(name)
	info.size = len(data)
	info.mtime = 0
	info.mode = mode
	info.uid = 0
	info.gid = 0
	info.uname = ""
	info.gname = ""
	return info


def write_archive(skill: Skill, out_dir: Path) -> Path:
	# [FUNCTION DOC] Write a reproducible <out>/<name>.tar.gz containing all skill files.
	# Behavior: opens a gzip stream with no file name and zero mtime, wraps a tar writer, and adds
	# each skill file in order under its relative path with normalized header and original
	# permission bits.
	# Inputs: skill; out_dir.
	# Outputs: archive path.
	# Exceptions: ValueError for unsafe paths; OS errors propagate.
	# Side effects: writes the archive file.
	target = out_dir / f"{skill.name}.tar.gz"
	with target.open("wb") as raw:
		with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
			with tarfile.open(fileobj=gz, mode="w") as tar:
				for path in skill.files:
					rel = rel_archive_path(skill.dir, path)
					data = path.read_bytes()
					tar.addfile(
						tar_info(rel, data, path.stat().st_mode & 0o777),
						fileobj=io.BytesIO(data),
					)
	return target


def uri_join(prefix: str, path: str) -> str:
	# [FUNCTION DOC] HELPER: Join a URI prefix and a path.
	# Behavior: a prefix ending in "://" is concatenated directly; otherwise trailing slashes are
	# stripped and a single slash inserted.
	# Inputs: prefix; path. Outputs: joined URI. Exceptions: none. Side effects: none.
	if prefix.endswith("://"):
		return prefix + path
	return prefix.rstrip("/") + "/" + path


def resource_uri(uri_prefix: str, skill_name: str, relative_path: str) -> str:
	# [FUNCTION DOC] HELPER: Build a percent-encoded resource URI for a file of a skill.
	# Behavior: percent-encodes the skill name and each path segment (slashes encoded within names)
	# and joins them under the prefix.
	# Inputs: uri_prefix; skill_name; relative_path - slash-separated.
	# Outputs: URI string. Exceptions: none. Side effects: none.
	encoded_path = "/".join(quote(part, safe="") for part in relative_path.split("/"))
	return uri_join(uri_prefix, f"{quote(skill_name, safe='')}/{encoded_path}")


def write_skill_files(skill: Skill, out_dir: Path, uri_prefix: str) -> dict[str, Any]:
	# [FUNCTION DOC] Copy every skill file to <out>/<name>/ and build its SEP-2640 entry.
	# Behavior: for each file copies it (with metadata) to the expanded tree, then hashes and sizes
	# the copied bytes and records uri, "sha256:<hex>" digest and size; verifies SKILL.md's URI is
	# among the resources.
	# Inputs: skill; out_dir; uri_prefix.
	# Outputs: mapping with uri (of SKILL.md), frontmatter and resources list.
	# Exceptions: ValueError if SKILL.md is missing from resources or paths are unsafe.
	# Side effects: creates directories and copies files.
	resources = []
	for path in skill.files:
		relative_path = rel_archive_path(skill.dir, path)
		target = out_dir / skill.name / relative_path
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(path, target)
		digest, size = sha256_and_size(target)
		resources.append(
			{
				"uri": resource_uri(uri_prefix, skill.name, relative_path),
				"digest": f"sha256:{digest}",
				"size": size,
			}
		)

	skill_uri = resource_uri(uri_prefix, skill.name, "SKILL.md")
	if not any(resource["uri"] == skill_uri for resource in resources):
		raise ValueError(f"{skill.dir}: resource manifest does not include SKILL.md")

	return {
		"uri": skill_uri,
		"frontmatter": skill.frontmatter,
		"resources": resources,
	}


def build_distribution(skills_dir: Path, out_dir: Path, uri_prefix: str) -> None:
	# [FUNCTION DOC] Build the complete distribution into out_dir.
	# Behavior: deletes out_dir if it exists and recreates it; loads skills; for each skill writes the
	# expanded file tree and SEP-2640 entry. A skill consisting only of SKILL.md gets a legacy index
	# entry (url, digest, frontmatter) and a markdown catalog entry; others get a .tar.gz archive, an
	# index entry with an archives list, and an archive catalog entry. Then writes index.json,
	# skills.json, ai-catalog.json (host info and entries), manifest.json (schema 1, generation time,
	# source, count, prefix, artifact names) and a _SUCCESS file with the generation time.
	# Inputs: skills_dir; out_dir; uri_prefix.
	# Outputs: none.
	# Exceptions: ValueError from validation; OS errors propagate.
	# Side effects: recursively deletes out_dir; writes many files.
	# KNOWN ISSUE: for single-file skills SKILL.md is copied a second time over the copy already
	# made by write_skill_files (harmless but redundant).
	if out_dir.exists():
		shutil.rmtree(out_dir)
	out_dir.mkdir(parents=True)

	skills = load_skills(skills_dir)
	index_entries: list[dict[str, Any]] = []
	skill_entries: list[dict[str, Any]] = []
	catalog_entries: list[dict[str, Any]] = []

	for skill in skills:
		skill_entries.append(write_skill_files(skill, out_dir, uri_prefix))
		if len(skill.files) == 1 and skill.files[0].name == "SKILL.md":
			skill_md_path = write_skill_md(skill, out_dir)
			skill_md_url = uri_join(uri_prefix, f"{skill.name}/SKILL.md")
			skill_md_digest = f"sha256:{sha256_hex(skill_md_path)}"

			index_entries.append(
				{
					"url": skill_md_url,
					"digest": skill_md_digest,
					"frontmatter": skill.frontmatter,
				}
			)
			catalog_url = skill_md_url
			catalog_media_type = "text/markdown"
			catalog_digest = skill_md_digest
			catalog_path = f"skills/{skill.name}/SKILL.md"
		else:
			archive_path = write_archive(skill, out_dir)
			archive_url = uri_join(uri_prefix, f"{skill.name}.tar.gz")
			archive_digest = f"sha256:{sha256_hex(archive_path)}"

			index_entries.append(
				{
					"frontmatter": skill.frontmatter,
					"archives": [
						{
							"url": archive_url,
							"mimeType": ARCHIVE_MEDIA_TYPE,
							"digest": archive_digest,
						}
					],
				}
			)
			catalog_url = archive_url
			catalog_media_type = ARCHIVE_MEDIA_TYPE
			catalog_digest = archive_digest
			catalog_path = f"skills/{skill.name}.tar.gz"

		catalog_entries.append(
			{
				"identifier": f"urn:huggingface:skill:{skill.name}",
				"displayName": skill.name,
				"mediaType": catalog_media_type,
				"url": catalog_url,
				"description": skill.description,
				"tags": ["huggingface", "skill"],
				"metadata": {
					"digest": catalog_digest,
					"source": SOURCE_REPO,
					"path": catalog_path,
				},
			}
		)

	generated_at = utc_now()
	write_json(out_dir / "index.json", {"skills": index_entries})
	write_json(out_dir / "skills.json", {"skills": skill_entries})
	write_json(
		out_dir / "ai-catalog.json",
		{
			"specVersion": "1.0",
			"host": {
				"displayName": "Hugging Face Skills",
				"identifier": f"https://github.com/{SOURCE_REPO}",
				"documentationUrl": f"https://github.com/{SOURCE_REPO}",
				"logoUrl": "https://huggingface.co/favicon.ico",
			},
			"entries": catalog_entries,
		},
	)
	write_json(
		out_dir / "manifest.json",
		{
			"schema_version": 1,
			"generated_at": generated_at,
			"source_repo": SOURCE_REPO,
			"skill_count": len(skills),
			"uri_prefix": uri_prefix,
			"artifacts": {
				"skills_index": "index.json",
				"skills_catalog": "skills.json",
				"ai_catalog": "ai-catalog.json",
			},
		},
	)
	(out_dir / "_SUCCESS").write_text(generated_at + "\n", encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
	# [FUNCTION DOC] HELPER: Write a mapping as 2-space-indented UTF-8 JSON with a trailing newline.
	# Inputs: path; value. Outputs: none.
	# Exceptions: OS/serialization errors propagate. Side effects: overwrites the file.
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
	# [FUNCTION DOC] HELPER: Parse command-line arguments (see module block).
	# Inputs: process arguments. Outputs: parsed namespace.
	# Exceptions: SystemExit on invalid arguments. Side effects: may print usage.
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--skills-dir", type=Path, default=Path("skills"), help="directory containing skill subdirectories")
	parser.add_argument("--out-dir", type=Path, required=True, help="distribution output directory")
	parser.add_argument("--uri-prefix", default="skill://", help="URI prefix used in generated indexes")
	return parser.parse_args()


def main() -> None:
	# [FUNCTION DOC] Entry point: parse arguments and build the distribution with resolved paths.
	# Inputs: command line. Outputs: none.
	# Exceptions: propagated. Side effects: as build_distribution.
	args = parse_args()
	build_distribution(args.skills_dir.resolve(), args.out_dir.resolve(), args.uri_prefix)


if __name__ == "__main__":
	main()
