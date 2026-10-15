#!/usr/bin/env -S uv run
# [MODULE DOC] Unit tests for build_skill_distribution.py.
# Behavior: loads the builder script as a module by file path and tests manifest contents,
# hashing across chunk boundaries, and metadata validation, using temporary directories.
# CLI arguments: none (runs unittest's main when executed). Env vars: none.
# Constants: SCRIPT_PATH (sibling builder script); DIGEST_RE (valid "sha256:<64 hex>" digest).
# Top-level code: imports the builder at import time into the global "builder" and registers it
# in the module registry under "build_skill_distribution".
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml==6.0.3"]
# ///

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote


SCRIPT_PATH = Path(__file__).with_name("build_skill_distribution.py")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def load_builder() -> ModuleType:
	# [FUNCTION DOC] HELPER: Import build_skill_distribution.py from its file path.
	# Behavior: creates an import spec, registers the module in the module registry and executes it.
	# Inputs: none.
	# Outputs: the loaded module.
	# Exceptions: RuntimeError if no spec/loader can be created; import errors propagate.
	# Side effects: adds the module to the global module registry.
	spec = importlib.util.spec_from_file_location("build_skill_distribution", SCRIPT_PATH)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"could not load {SCRIPT_PATH}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


builder = load_builder()


def write_skill(skill_dir: Path, description: str = "Test skill") -> None:
	# [FUNCTION DOC] HELPER: Create a minimal skill directory with a SKILL.md fixture.
	# Behavior: creates the directory (parents allowed, must not exist) and writes frontmatter with
	# name equal to the directory name, the given description and license Apache-2.0, then a heading.
	# Inputs: skill_dir; description - default "Test skill".
	# Outputs: none.
	# Exceptions: FileExistsError if the directory exists.
	# Side effects: writes to the filesystem.
	skill_dir.mkdir(parents=True)
	(skill_dir / "SKILL.md").write_text(
		f"---\nname: {skill_dir.name}\ndescription: {description}\nlicense: Apache-2.0\n---\n\n# Test\n",
		encoding="utf-8",
	)


class BuildSkillDistributionTest(unittest.TestCase):
	# [CLASS DOC] Test case covering the distribution builder.
	# Behavior: groups end-to-end and unit tests; no fixtures beyond temporary directories.
	def test_writes_complete_sep_2640_resource_manifests(self) -> None:
		# [FUNCTION DOC] End-to-end check of skills.json resource manifests and legacy outputs.
		# Behavior: builds a single-file skill and a multi-file skill (CRLF text, unicode, empty and binary
		# files with a space in the name, plus ignored node_modules and .DS_Store), runs the builder, and
		# asserts: exactly two entries keyed by name; each entry has only uri/frontmatter/resources with
		# the expected SKILL.md URI and license; resources are unique, include SKILL.md and match the
		# expected path set; each resource has only uri/digest/size, an integer size, a valid digest that
		# matches the published bytes, which equal the source bytes; the binary file's digest is exact;
		# ignored entries are not published; index.json has two skills; the multi archive exists; the
		# manifest names skills.json as the catalog.
		# Inputs: none. Outputs: none.
		# Exceptions: assertion failures.
		# Side effects: temporary directory I/O.
		with tempfile.TemporaryDirectory() as temp_dir:
			root = Path(temp_dir)
			skills_dir = root / "skills"
			out_dir = root / "out"

			single = skills_dir / "single"
			write_skill(single, "Single-file skill")

			multi = skills_dir / "multi"
			write_skill(multi, "Multi-file skill")
			(multi / "references").mkdir()
			(multi / "references" / "guide.md").write_text("# Guide\r\n", encoding="utf-8", newline="")
			(multi / "references" / "unicode.md").write_text("Café 🤗\n", encoding="utf-8")
			(multi / "assets").mkdir()
			(multi / "assets" / "empty.bin").write_bytes(b"")
			binary_payload = b"line1\r\nline2\x00\xff\n"
			(multi / "assets" / "raw data.bin").write_bytes(binary_payload)
			(multi / "node_modules").mkdir()
			(multi / "node_modules" / "ignored.js").write_text("ignored", encoding="utf-8")
			(multi / ".DS_Store").write_bytes(b"ignored")

			builder.build_distribution(skills_dir, out_dir, "skill://")

			payload = json.loads((out_dir / "skills.json").read_text(encoding="utf-8"))
			entries = {entry["frontmatter"]["name"]: entry for entry in payload["skills"]}
			self.assertEqual(set(entries), {"multi", "single"})

			for skill_name, expected_paths in {
				"single": {"SKILL.md"},
				"multi": {
					"SKILL.md",
					"assets/raw data.bin",
					"assets/empty.bin",
					"references/guide.md",
					"references/unicode.md",
				},
			}.items():
				entry = entries[skill_name]
				self.assertEqual(set(entry), {"uri", "frontmatter", "resources"})
				self.assertEqual(entry["uri"], f"skill://{skill_name}/SKILL.md")
				self.assertEqual(entry["frontmatter"]["license"], "Apache-2.0")

				resources = entry["resources"]
				self.assertEqual(len(resources), len(expected_paths))
				self.assertEqual(len({resource["uri"] for resource in resources}), len(resources))
				self.assertIn(entry["uri"], {resource["uri"] for resource in resources})

				actual_paths = {
					unquote(resource["uri"].split(f"skill://{skill_name}/", maxsplit=1)[1])
					for resource in resources
				}
				self.assertEqual(actual_paths, expected_paths)

				for resource in resources:
					self.assertEqual(set(resource), {"uri", "digest", "size"})
					self.assertIs(type(resource["size"]), int)
					self.assertRegex(resource["digest"], DIGEST_RE)
					relative_uri = resource["uri"].split(f"skill://{skill_name}/", maxsplit=1)[1]
					relative_path = unquote(relative_uri)
					source_bytes = (skills_dir / skill_name / relative_path).read_bytes()
					published_path = out_dir / skill_name / relative_path
					self.assertEqual(published_path.read_bytes(), source_bytes)
					expected_digest = "sha256:" + hashlib.sha256(published_path.read_bytes()).hexdigest()
					self.assertEqual(resource["digest"], expected_digest)
					self.assertEqual(resource["size"], len(source_bytes))

			binary_resource = next(
				resource for resource in entries["multi"]["resources"] if resource["uri"].endswith("raw%20data.bin")
			)
			self.assertEqual(binary_resource["digest"], "sha256:" + hashlib.sha256(binary_payload).hexdigest())
			self.assertFalse((out_dir / "multi" / "node_modules").exists())
			self.assertFalse((out_dir / "multi" / ".DS_Store").exists())

			legacy = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
			self.assertEqual(len(legacy["skills"]), 2)
			self.assertTrue((out_dir / "multi.tar.gz").is_file())

			manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
			self.assertEqual(manifest["artifacts"]["skills_catalog"], "skills.json")

	def test_hash_and_size_across_chunk_boundaries(self) -> None:
		# [FUNCTION DOC] Check hashing of a file larger than one read chunk.
		# Behavior: writes slightly over 2 MiB of bytes and asserts the digest and size match a direct
		# hash of the payload and its length.
		# Inputs: none. Outputs: none.
		# Exceptions: assertion failures. Side effects: temporary file I/O.
		with tempfile.TemporaryDirectory() as temp_dir:
			path = Path(temp_dir) / "large.bin"
			payload = b"\x00\xff" * (1024 * 1024 + 1)
			path.write_bytes(payload)
			self.assertEqual(builder.sha256_and_size(path), (hashlib.sha256(payload).hexdigest(), len(payload)))

	def test_rejects_non_string_metadata_values(self) -> None:
		# [FUNCTION DOC] Check metadata validation.
		# Behavior: writes a skill whose metadata maps a key to a list and asserts building raises
		# ValueError with the "metadata must map string keys to string values" message.
		# Inputs: none. Outputs: none.
		# Exceptions: assertion failures. Side effects: temporary directory I/O.
		with tempfile.TemporaryDirectory() as temp_dir:
			root = Path(temp_dir)
			skill_dir = root / "skills" / "invalid"
			skill_dir.mkdir(parents=True)
			(skill_dir / "SKILL.md").write_text(
				"---\n"
				"name: invalid\n"
				"description: Invalid metadata fixture\n"
				"metadata:\n"
				"  tags:\n"
				"    - one\n"
				"---\n",
				encoding="utf-8",
			)
			with self.assertRaisesRegex(ValueError, "metadata must map string keys to string values"):
				builder.build_distribution(root / "skills", root / "out", "skill://")


if __name__ == "__main__":
	unittest.main()
