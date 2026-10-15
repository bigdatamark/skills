#!/usr/bin/env -S uv run
# [MODULE DOC] CLI for publishing research papers on the Hugging Face Hub.
# Behavior overview: checks/indexes arXiv papers on huggingface.co/papers, links papers into a
#   repo README, renders research articles from local Markdown templates, fetches arXiv metadata
#   and generates BibTeX citations.
# CLI arguments: subcommands index/check (--arxiv-id required); link (--repo-id required,
#   --repo-type model|dataset|space default model, --arxiv-id, --arxiv-ids comma list,
#   --citation, --create-pr); create (--template, --title, --output required; --authors,
#   --abstract); info (--arxiv-id, --format json|text default json); citation (--arxiv-id,
#   --format bibtex|apa|mla default bibtex); search (--query).
# Env vars: HF_TOKEN (also loaded from a .env file via load_dotenv); falls back to the cached
#   Hub token.
# Top-level executed code: imports third-party deps, printing a hint and exiting 1 if any is
#   missing; loads .env into the environment; as a script runs main.
# Side effects: network calls to huggingface.co and export.arxiv.org, Hub README commits, local
#   file writes for created articles, stdout output.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
#     "pyyaml",
#     "requests",
#     "python-dotenv",
# ]
# ///
"""
Paper Manager for Hugging Face Hub
Manages paper indexing, linking, authorship, and article creation.
"""

import argparse
import os
import sys
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from huggingface_hub import HfApi, hf_hub_download, get_token
    import yaml
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Tip: run this script with `uv run scripts/paper_manager.py ...`.")
    sys.exit(1)

# Load environment variables
load_dotenv()


class PaperManager:
    # [CLASS DOC] Groups paper operations against the Hub and arXiv around one authenticated client.
    # Behavior: holds the token and an HfApi client; class constants hold regexes for modern
    #   (YYMM.NNNN[N][vN]) and legacy (category/YYMMNNN[vN]) arXiv ids; static helpers clean ids and
    #   sanitize/escape untrusted text.
    """Manages paper publishing operations on Hugging Face Hub."""

    def __init__(self, hf_token: Optional[str] = None):
        # [FUNCTION DOC] Resolve the Hub token and create the API client.
        # Behavior: uses the given token, else HF_TOKEN, else the locally cached token; prints a warning
        #   if none found; creates an HfApi with that token.
        # Inputs: hf_token (optional string, default none).
        # Outputs: none.
        # Exceptions: none.
        # Side effects: reads env/cached token, stdout warning.
        """Initialize Paper Manager with HF token."""
        self.token = hf_token or os.getenv("HF_TOKEN") or get_token()
        if not self.token:
            print("Warning: No HF_TOKEN found. Some operations will fail.")
        self.api = HfApi(token=self.token)

    def index_paper(self, arxiv_id: str) -> Dict[str, Any]:
        # [FUNCTION DOC] Check whether a paper is indexed on HF Papers and advise how to index it.
        # Behavior: cleans the id (error dict on invalid); GETs the HF paper page (10 s timeout); status
        #   200 means already indexed; otherwise reports it is not indexed and tells the user to visit
        #   the URL. It does not itself trigger indexing.
        # Inputs: arxiv_id (string, may include prefixes/URLs).
        # Outputs: mapping with status "exists" (+url), "not_indexed" (+url, action), or "error"
        #   (+message); also prints messages.
        # Exceptions: ValueError and request errors are caught and turned into error results.
        # Side effects: network request, stdout.
        """
        Index a paper on Hugging Face from arXiv.

        Args:
            arxiv_id: arXiv identifier (e.g., "2301.12345")

        Returns:
            dict: Status information
        """
        # Clean and validate arXiv ID
        try:
            arxiv_id = self._clean_arxiv_id(arxiv_id)
        except ValueError as e:
            print(f"Error: {e}")
            return {"status": "error", "message": str(e)}

        print(f"Indexing paper {arxiv_id} on Hugging Face...")

        # Check if paper exists
        paper_url = f"https://huggingface.co/papers/{arxiv_id}"

        try:
            response = requests.get(paper_url, timeout=10)
            if response.status_code == 200:
                print(f"✓ Paper already indexed at {paper_url}")
                return {"status": "exists", "url": paper_url}
            else:
                print(f"Paper not indexed. Visit {paper_url} to trigger indexing.")
                print("The paper will be automatically indexed when you first visit the URL.")
                return {"status": "not_indexed", "url": paper_url, "action": "visit_url"}
        except requests.RequestException as e:
            print(f"Error checking paper status: {e}")
            return {"status": "error", "message": str(e)}

    def check_paper(self, arxiv_id: str) -> Dict[str, Any]:
        # [FUNCTION DOC] Report whether a paper page exists on HF Papers.
        # Behavior: cleans the id; GETs the paper page (10 s timeout); 200 returns exists true with HF
        #   and arXiv URLs; other statuses return exists false with an index URL and message.
        # Inputs: arxiv_id (string).
        # Outputs: mapping with exists flag and either URLs/ids, a message, or an error string.
        # Exceptions: invalid ids and request errors are caught and returned as error fields.
        # Side effects: network request.
        """
        Check if a paper exists on Hugging Face.

        Args:
            arxiv_id: arXiv identifier

        Returns:
            dict: Paper status and metadata
        """
        try:
            arxiv_id = self._clean_arxiv_id(arxiv_id)
        except ValueError as e:
            return {"exists": False, "error": str(e)}
        paper_url = f"https://huggingface.co/papers/{arxiv_id}"

        try:
            response = requests.get(paper_url, timeout=10)
            if response.status_code == 200:
                return {
                    "exists": True,
                    "url": paper_url,
                    "arxiv_id": arxiv_id,
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}"
                }
            else:
                return {
                    "exists": False,
                    "arxiv_id": arxiv_id,
                    "index_url": paper_url,
                    "message": f"Visit {paper_url} to index this paper"
                }
        except requests.RequestException as e:
            return {"exists": False, "error": str(e)}

    def link_paper_to_repo(
        self,
        repo_id: str,
        arxiv_id: str,
        repo_type: str = "model",
        citation: Optional[str] = None,
        create_pr: bool = False
    ) -> Dict[str, Any]:
        # [FUNCTION DOC] Add a paper reference section to a Hub repo's README and commit it.
        # Behavior: cleans the id; downloads README.md of the repo; inserts the paper section via
        #   _add_paper_to_readme; if create_pr is set, prints that PRs are unimplemented; uploads the new
        #   README as a direct commit "Add paper reference: arXiv:<id>"; prints success with URLs.
        # Inputs: repo_id (string); arxiv_id (string); repo_type (string, default "model");
        #   citation (optional string); create_pr (bool, default false).
        # Outputs: mapping with status "success" plus paper/repo URLs and id, or "error" with message.
        # Exceptions: all exceptions (including invalid id, missing README, upload failures) are caught,
        #   printed and returned as error results.
        # Side effects: network download/upload, a commit to the remote repo, stdout.
        # KNOWN ISSUE: create_pr is ignored; the change is always committed directly to the main branch.
        # KNOWN ISSUE: if the paper is already referenced, the unchanged README is still uploaded and
        #   success reported.
        """
        Link a paper to a model/dataset/space repository.

        Args:
            repo_id: Repository identifier (e.g., "username/repo-name")
            arxiv_id: arXiv identifier
            repo_type: Type of repository ("model", "dataset", or "space")
            citation: Optional full citation text
            create_pr: Create a PR instead of direct commit

        Returns:
            dict: Operation status
        """
        try:
            arxiv_id = self._clean_arxiv_id(arxiv_id)
        except ValueError as e:
            print(f"Error: {e}")
            return {"status": "error", "message": str(e)}

        print(f"Linking paper {arxiv_id} to {repo_type} {repo_id}...")

        try:
            # Download current README
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type=repo_type,
                token=self.token
            )

            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse or create YAML frontmatter
            updated_content = self._add_paper_to_readme(content, arxiv_id, citation)

            # Upload updated README
            commit_message = f"Add paper reference: arXiv:{arxiv_id}"

            if create_pr:
                # Create PR (not implemented in basic version)
                print("PR creation not yet implemented. Committing directly.")

            self.api.upload_file(
                path_or_fileobj=updated_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                token=self.token
            )

            paper_url = f"https://huggingface.co/papers/{arxiv_id}"
            repo_url = f"https://huggingface.co/{repo_id}"

            print(f"✓ Successfully linked paper to repository")
            print(f"  Paper: {paper_url}")
            print(f"  Repo: {repo_url}")

            return {
                "status": "success",
                "paper_url": paper_url,
                "repo_url": repo_url,
                "arxiv_id": arxiv_id
            }

        except Exception as e:
            print(f"Error linking paper: {e}")
            return {"status": "error", "message": str(e)}

    def _add_paper_to_readme(
        self,
        content: str,
        arxiv_id: str,
        citation: Optional[str] = None
    ) -> str:
        # [FUNCTION DOC] HELPER: Return README text with a marked "Paper" section inserted after the YAML.
        # Behavior: if the content starts with a YAML front-matter block, returns it unchanged when the id
        #   already appears anywhere; otherwise splits after the front matter. Without front matter,
        #   prepends an empty front-matter block. Builds a section between paper-manager start/end HTML
        #   comments with a "## Paper" heading, a sentence saying "This model" (if "model" appears in the
        #   text) or "This work", arXiv and HF links, and an optional sanitized BibTeX citation block.
        # Inputs: content (README text); arxiv_id (string); citation (optional string).
        # Outputs: updated README text.
        # Exceptions: none.
        # Side effects: prints a message when already referenced.
        # KNOWN ISSUE: the duplicate check only runs when front matter exists, so READMEs without it get
        #   a duplicate section on repeated runs.
        """
        Add paper reference to README content.

        Args:
            content: Current README content
            arxiv_id: arXiv identifier
            citation: Optional citation text

        Returns:
            str: Updated README content
        """
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
        hf_paper_url = f"https://huggingface.co/papers/{arxiv_id}"

        # Check if YAML frontmatter exists
        yaml_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(yaml_pattern, content, re.DOTALL)

        if match:
            # YAML exists, check if paper already referenced
            if arxiv_id in content:
                print(f"Paper {arxiv_id} already referenced in README")
                return content

            # Add to existing content (after YAML)
            yaml_end = match.end()
            before = content[:yaml_end]
            after = content[yaml_end:]
        else:
            # No YAML, add minimal frontmatter
            yaml_content = "---\n---\n\n"
            before = yaml_content
            after = content

        # Add paper reference section with boundary markers
        paper_section = "\n<!-- paper-manager:start -->\n"
        paper_section += f"## Paper\n\n"
        paper_section += f"This {'model' if 'model' in content.lower() else 'work'} is based on research presented in:\n\n"
        paper_section += f"**[View on arXiv]({arxiv_url})** | "
        paper_section += f"**[View on Hugging Face]({hf_paper_url})**\n\n"

        if citation:
            safe_citation = self._sanitize_text(citation)
            paper_section += f"### Citation\n\n```bibtex\n{safe_citation}\n```\n\n"

        paper_section += "<!-- paper-manager:end -->\n"

        # Insert after YAML, before main content
        updated_content = before + paper_section + after

        return updated_content

    def create_research_article(
        self,
        template: str,
        title: str,
        output: str,
        authors: Optional[str] = None,
        abstract: Optional[str] = None
    ) -> Dict[str, Any]:
        # [FUNCTION DOC] Render a research article from a Markdown template to a file.
        # Behavior: loads templates/<template>.md next to the scripts folder (error result if missing);
        #   computes today's date and sanitized title/authors/abstract (defaults "Your Name" and
        #   "Abstract to be written..."). If the template has front matter, replaces TITLE/AUTHORS
        #   placeholders there with YAML-escaped values and DATE raw, and in the body replaces
        #   TITLE/AUTHORS/ABSTRACT/DATE with sanitized values; otherwise replaces all placeholders with
        #   sanitized values. Writes the result to the output path.
        # Inputs: template (name); title (string); output (file path); authors (optional comma list);
        #   abstract (optional string).
        # Outputs: mapping with status "success", output and template, or status "error" with message.
        # Exceptions: file write/read errors propagate.
        # Side effects: reads template file, writes/overwrites output file, stdout.
        # KNOWN ISSUE: an ABSTRACT placeholder inside front matter is not replaced.
        """
        Create a research article from template.

        Args:
            template: Template name ("standard", "modern", "arxiv", "ml-report")
            title: Paper title
            output: Output filename
            authors: Comma-separated author names
            abstract: Abstract text

        Returns:
            dict: Creation status
        """
        print(f"Creating research article with '{template}' template...")

        # Load template
        template_dir = Path(__file__).parent.parent / "templates"
        template_file = template_dir / f"{template}.md"

        if not template_file.exists():
            return {
                "status": "error",
                "message": f"Template '{template}' not found at {template_file}"
            }

        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # Prepare safe values for different contexts
        date_str = datetime.now().strftime("%Y-%m-%d")
        safe_title_body = self._sanitize_text(title)
        authors_val = authors if authors else "Your Name"
        safe_authors_body = self._sanitize_text(authors_val)
        abstract_val = abstract if abstract else "Abstract to be written..."
        safe_abstract_body = self._sanitize_text(abstract_val)

        # Split frontmatter from body for context-aware escaping
        fm_pattern = r'^(---\s*\n)(.*?\n)(---\s*\n)'
        fm_match = re.match(fm_pattern, template_content, re.DOTALL)

        if fm_match:
            fm_open, fm_body, fm_close = fm_match.group(1), fm_match.group(2), fm_match.group(3)
            body = template_content[fm_match.end():]

            # YAML-escape values in frontmatter
            fm_body = fm_body.replace("{{TITLE}}", self._escape_yaml_value(title))
            fm_body = fm_body.replace("{{AUTHORS}}", self._escape_yaml_value(authors_val))
            fm_body = fm_body.replace("{{DATE}}", date_str)

            # Sanitize values in body
            body = body.replace("{{TITLE}}", safe_title_body)
            body = body.replace("{{AUTHORS}}", safe_authors_body)
            body = body.replace("{{ABSTRACT}}", safe_abstract_body)
            body = body.replace("{{DATE}}", date_str)

            content = fm_open + fm_body + fm_close + body
        else:
            # No frontmatter — sanitize everything
            content = template_content.replace("{{TITLE}}", safe_title_body)
            content = content.replace("{{DATE}}", date_str)
            content = content.replace("{{AUTHORS}}", safe_authors_body)
            content = content.replace("{{ABSTRACT}}", safe_abstract_body)

        # Write output
        with open(output, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ Research article created at {output}")

        return {
            "status": "success",
            "output": output,
            "template": template
        }

    def get_arxiv_info(self, arxiv_id: str) -> Dict[str, Any]:
        # [FUNCTION DOC] Fetch a paper's title, authors and abstract from the arXiv Atom API.
        # Behavior: cleans the id; GETs the export.arxiv.org query (10 s timeout, raising on HTTP error);
        #   extracts title, all name tags and summary with regexes; drops the first name; sanitizes
        #   values; returns with arXiv abs and PDF URLs.
        # Inputs: arxiv_id (string).
        # Outputs: mapping with arxiv_id, title, authors list, abstract, arxiv_url, pdf_url; or with an
        #   "error" string.
        # Exceptions: all exceptions are caught and returned as the error field.
        # Side effects: network request.
        # KNOWN ISSUE: the first title tag in the Atom feed is the feed title ("arXiv Query: ..."), not the
        #   paper title, so the returned title is wrong.
        # KNOWN ISSUE: skipping the first name tag drops the paper's real first author, since the feed
        #   has no feed-level author name.
        """
        Fetch paper information from arXiv API.

        Args:
            arxiv_id: arXiv identifier

        Returns:
            dict: Paper metadata
        """
        try:
            arxiv_id = self._clean_arxiv_id(arxiv_id)
        except ValueError as e:
            return {"error": str(e)}
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"

        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()

            # Parse XML response (simplified)
            content = response.text

            # Extract basic info with regex (proper XML parsing would be better)
            title_match = re.search(r'<title>(.*?)</title>', content, re.DOTALL)
            authors_matches = re.findall(r'<name>(.*?)</name>', content)
            summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)

            # Sanitize all text extracted from the external API
            raw_title = title_match.group(1).strip() if title_match else None
            raw_authors = authors_matches[1:] if len(authors_matches) > 1 else []
            raw_abstract = summary_match.group(1).strip() if summary_match else None

            return {
                "arxiv_id": arxiv_id,
                "title": self._sanitize_text(raw_title) if raw_title else None,
                "authors": [self._sanitize_text(a) for a in raw_authors],
                "abstract": self._sanitize_text(raw_abstract) if raw_abstract else None,
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            }
        except Exception as e:
            return {"error": str(e)}

    def generate_citation(
        self,
        arxiv_id: str,
        format: str = "bibtex"
    ) -> str:
        # [FUNCTION DOC] Generate a citation string for an arXiv paper.
        # Behavior: cleans the id and fetches arXiv info (error strings on failure). For "bibtex", builds
        #   key "arxiv<id with dots as underscores>", joins authors with " and ", derives the year from
        #   the first two digits of the id (below 50 means 20xx else 19xx), escapes braces in title and
        #   authors, and formats an article entry. Other formats return a not-implemented message.
        # Inputs: arxiv_id (string); format (string, default "bibtex").
        # Outputs: citation text or an error/not-implemented message string.
        # Exceptions: invalid ids are caught and returned as text; others propagate.
        # Side effects: network request.
        # KNOWN ISSUE: legacy ids such as category/YYMMNNN make the year parse fail with an uncaught
        #   ValueError.
        # KNOWN ISSUE: a missing title is stored as none (key present), so the "Untitled" default never
        #   applies and brace escaping crashes; empty author lists likewise skip the "Unknown" default.
        # KNOWN ISSUE: "apa" and "mla" are accepted by the CLI but not implemented.
        """
        Generate citation for a paper.

        Args:
            arxiv_id: arXiv identifier
            format: Citation format ("bibtex", "apa", "mla")

        Returns:
            str: Formatted citation
        """
        try:
            arxiv_id = self._clean_arxiv_id(arxiv_id)
        except ValueError as e:
            return f"Error: {e}"

        info = self.get_arxiv_info(arxiv_id)

        if "error" in info:
            return f"Error fetching paper info: {info['error']}"

        if format == "bibtex":
            # Generate BibTeX citation
            key = f"arxiv{arxiv_id.replace('.', '_')}"
            raw_authors = " and ".join(info.get("authors", ["Unknown"]))
            raw_title = info.get("title", "Untitled")
            year = arxiv_id.split(".")[0][:2]  # Extract year from ID (simplified)
            year = f"20{year}" if int(year) < 50 else f"19{year}"

            # Escape BibTeX structural characters in untrusted values
            safe_title = raw_title.replace('{', r'\{').replace('}', r'\}')
            safe_authors = raw_authors.replace('{', r'\{').replace('}', r'\}')

            citation = f"""@article{{{key},
  title={{{safe_title}}},
  author={{{safe_authors}}},
  journal={{arXiv preprint arXiv:{arxiv_id}}},
  year={{{year}}}
}}"""
            return citation

        return f"Format '{format}' not yet implemented"

    # Patterns for valid arXiv IDs
    _ARXIV_ID_MODERN = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')
    _ARXIV_ID_LEGACY = re.compile(r'^[a-zA-Z\-]+/\d{7}(v\d+)?$')

    @staticmethod
    def _clean_arxiv_id(arxiv_id: str) -> str:
        # [FUNCTION DOC] HELPER: Normalize and validate an arXiv identifier.
        # Behavior: trims whitespace, removes a leading "arxiv:" prefix (any case), removes an arxiv.org
        #   abs/pdf URL prefix, removes ".pdf", then requires a modern or legacy id format.
        # Inputs: arxiv_id (string).
        # Outputs: cleaned id string.
        # Exceptions: ValueError if the cleaned id is not a valid format.
        # Side effects: none.
        """Clean, normalize, and validate arXiv ID.

        Raises:
            ValueError: If the cleaned ID does not match a valid arXiv format.
        """
        # Remove common prefixes and whitespace
        arxiv_id = arxiv_id.strip()
        arxiv_id = re.sub(r'^(arxiv:|arXiv:)', '', arxiv_id, flags=re.IGNORECASE)
        arxiv_id = re.sub(r'https?://arxiv\.org/(abs|pdf)/', '', arxiv_id)
        arxiv_id = arxiv_id.replace('.pdf', '')

        # Validate format
        if not (PaperManager._ARXIV_ID_MODERN.match(arxiv_id)
                or PaperManager._ARXIV_ID_LEGACY.match(arxiv_id)):
            raise ValueError(
                f"Invalid arXiv ID: {arxiv_id!r}. "
                "Expected format: YYMM.NNNNN[vN] or category/YYMMNNN[vN]"
            )

        return arxiv_id

    @staticmethod
    def _escape_yaml_value(value: str) -> str:
        # [FUNCTION DOC] HELPER: Quote a string as a double-quoted YAML scalar.
        # Behavior: escapes backslashes then double quotes, and wraps the result in double quotes.
        # Inputs: value (string).
        # Outputs: quoted string.
        # Exceptions: none.
        # Side effects: none.
        # KNOWN ISSUE: newlines are not escaped, so multi-line values can still break the YAML layout.
        """Escape a string for safe use as a YAML scalar value.

        Wraps in double quotes and escapes internal quotes and backslashes
        to prevent YAML injection via crafted titles/authors.
        """
        value = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{value}"'

    @staticmethod
    def _sanitize_text(text: str) -> str:
        # [FUNCTION DOC] HELPER: Clean untrusted text for safe inclusion in Markdown/YAML output.
        # Behavior: removes control characters except tab, newline and carriage return; collapses runs
        #   of non-newline whitespace to one space; limits blank lines to one; escapes triple backticks;
        #   backslash-escapes "---" at line starts; trims.
        # Inputs: text (string).
        # Outputs: sanitized string.
        # Exceptions: none.
        # Side effects: none.
        """Sanitize untrusted text for safe inclusion in Markdown/YAML output.

        Normalizes whitespace, strips control characters, and neutralizes
        markdown code-fence breakout and YAML document delimiters.
        """
        # Remove control characters (keep newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize whitespace runs (collapse multiple spaces/tabs, preserve single newlines)
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Neutralize markdown code fence breakout
        text = text.replace('```', r'\`\`\`')
        # Neutralize YAML document delimiters at line start
        text = re.sub(r'^---', r'\\---', text, flags=re.MULTILINE)
        return text.strip()


def main():
    # [FUNCTION DOC] CLI entry point: parse arguments and run the chosen subcommand.
    # Behavior: builds the parser with all subcommands; with no command prints help and exits 1.
    #   Creates a PaperManager, then: index/check print JSON results; link gathers ids from
    #   --arxiv-id and --arxiv-ids (error and exit 1 if none) and links each, printing each result;
    #   create renders an article and prints the JSON result; info prints JSON or a text summary;
    #   citation prints the citation; search only prints a placeholder and an HF Papers search URL.
    # Inputs: none (command-line arguments).
    # Outputs: stdout text/JSON; exit code 1 on missing command or ids, else 0.
    # Exceptions: none caught here beyond those handled by the manager methods.
    # Side effects: network, Hub commits, file writes as per subcommand.
    # KNOWN ISSUE: the search command is unimplemented and the query is not URL-encoded in the link.
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Paper Manager for Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a paper from arXiv")
    index_parser.add_argument("--arxiv-id", required=True, help="arXiv paper ID")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check if paper exists")
    check_parser.add_argument("--arxiv-id", required=True, help="arXiv paper ID")

    # Link command
    link_parser = subparsers.add_parser("link", help="Link paper to repository")
    link_parser.add_argument("--repo-id", required=True, help="Repository ID")
    link_parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    link_parser.add_argument("--arxiv-id", help="Single arXiv ID")
    link_parser.add_argument("--arxiv-ids", help="Comma-separated arXiv IDs")
    link_parser.add_argument("--citation", help="Full citation text")
    link_parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct commit")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create research article")
    create_parser.add_argument("--template", required=True, help="Template name")
    create_parser.add_argument("--title", required=True, help="Paper title")
    create_parser.add_argument("--output", required=True, help="Output filename")
    create_parser.add_argument("--authors", help="Comma-separated authors")
    create_parser.add_argument("--abstract", help="Abstract text")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get paper information")
    info_parser.add_argument("--arxiv-id", required=True, help="arXiv paper ID")
    info_parser.add_argument("--format", default="json", choices=["json", "text"])

    # Citation command
    citation_parser = subparsers.add_parser("citation", help="Generate citation")
    citation_parser.add_argument("--arxiv-id", required=True, help="arXiv paper ID")
    citation_parser.add_argument("--format", default="bibtex", choices=["bibtex", "apa", "mla"])

    # Search command
    search_parser = subparsers.add_parser("search", help="Search papers")
    search_parser.add_argument("--query", required=True, help="Search query")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize manager
    manager = PaperManager()

    # Execute command
    if args.command == "index":
        result = manager.index_paper(args.arxiv_id)
        print(json.dumps(result, indent=2))

    elif args.command == "check":
        result = manager.check_paper(args.arxiv_id)
        print(json.dumps(result, indent=2))

    elif args.command == "link":
        arxiv_ids = []
        if args.arxiv_id:
            arxiv_ids.append(args.arxiv_id)
        if args.arxiv_ids:
            arxiv_ids.extend([id.strip() for id in args.arxiv_ids.split(",")])

        if not arxiv_ids:
            print("Error: Must provide --arxiv-id or --arxiv-ids")
            sys.exit(1)

        for arxiv_id in arxiv_ids:
            result = manager.link_paper_to_repo(
                repo_id=args.repo_id,
                arxiv_id=arxiv_id,
                repo_type=args.repo_type,
                citation=args.citation,
                create_pr=args.create_pr
            )
            print(json.dumps(result, indent=2))

    elif args.command == "create":
        result = manager.create_research_article(
            template=args.template,
            title=args.title,
            output=args.output,
            authors=args.authors,
            abstract=args.abstract
        )
        print(json.dumps(result, indent=2))

    elif args.command == "info":
        result = manager.get_arxiv_info(args.arxiv_id)
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Title: {result.get('title')}")
                print(f"Authors: {', '.join(result.get('authors', []))}")
                print(f"arXiv URL: {result.get('arxiv_url')}")
                print(f"\nAbstract:\n{result.get('abstract')}")

    elif args.command == "citation":
        citation = manager.generate_citation(args.arxiv_id, args.format)
        print(citation)

    elif args.command == "search":
        print(f"Searching for: {args.query}")
        print("Search functionality coming soon!")
        print(f"Visit: https://huggingface.co/papers?search={args.query}")


if __name__ == "__main__":
    main()
