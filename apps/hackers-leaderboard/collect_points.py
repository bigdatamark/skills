#!/usr/bin/env python3
# [MODULE DOC] CLI script that computes engagement points for the hf-skills organization hackathon.
# Behavior: fetches the org's members (seeding them with zero stats), lists all org models,
# datasets and spaces, credits non-org repo authors with a repo point, and scans each repo's
# discussions: 1 point per discussion opened, per PR opened and per comment. Optionally scans
# trending Hub repos for PRs/discussions by org members. Prints a top-20 table, saves JSON
# locally and optionally uploads JSONL plus metadata to a dataset.
# Constants: API_BASE (Hub API root), ORG_NAME (hf-skills), USER_AGENT, DISCUSSION_LIMIT (100,
# discussions requested per repo), TRENDING_LIMIT (50 trending repos per type).
# CLI arguments: --push-to-hub (flag), --output (JSON path, default leaderboard.json), --repo-id
# (dataset, default hf-skills/hackers-leaderboard), --scan-external (flag, scan trending
# repos), --repo-type (one or more of models/datasets/spaces for the external scan; default all).
# Env vars: HF_TOKEN (optional; used for API session and member listing).
# Top-level executed code: main runs when executed as a script.
# Side effects: many network requests, console output, file write, optional dataset upload.
"""
Collect engagement points from the hf-skills organization.

Tracks user activity across all repos (models, datasets, spaces) and counts:
- 1 point per discussion opened
- 1 point per comment made
- 1 point per PR opened
- 1 point per repo owned/created

Results are saved to a dataset for the hackers leaderboard.

Usage:
    HF_TOKEN=$HF_TOKEN python collect_points.py [--push-to-hub]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

API_BASE = "https://huggingface.co/api"
ORG_NAME = "hf-skills"
USER_AGENT = "hf-skills-leaderboard/1.0"
DISCUSSION_LIMIT = 100  # Max discussions to fetch per repo
TRENDING_LIMIT = 50  # Number of trending repos to scan for external PRs


@dataclass
class UserStats:
    # [CLASS DOC] Mutable per-user engagement counters (dataclass).
    # Fields: username; is_org_member (default true); discussions_opened, comments_made,
    # prs_opened, repos_owned (integers, default 0); activities (list of activity mappings,
    # default empty).
    # Side effects: none.
    """Tracks engagement stats for a single user."""

    username: str
    is_org_member: bool = True
    discussions_opened: int = 0
    comments_made: int = 0
    prs_opened: int = 0
    repos_owned: int = 0
    activities: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        # [FUNCTION DOC] Total points for the user (read-only property).
        # Behavior: sum of discussions_opened, comments_made, prs_opened and repos_owned.
        # Inputs: none.
        # Outputs: integer.
        # Exceptions: none.
        # Side effects: none.
        return self.discussions_opened + self.comments_made + self.prs_opened + self.repos_owned

    def to_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Serialize stats to a mapping for export.
        # Behavior: returns username, is_org_member, total_points and the four counters; activities
        # are not included.
        # Inputs: none.
        # Outputs: mapping.
        # Exceptions: none.
        # Side effects: none.
        return {
            "username": self.username,
            "is_org_member": self.is_org_member,
            "total_points": self.total_points,
            "discussions_opened": self.discussions_opened,
            "comments_made": self.comments_made,
            "prs_opened": self.prs_opened,
            "repos_owned": self.repos_owned,
        }


class PointsCollector:
    # [CLASS DOC] Collects engagement points across hf-skills repos (and optionally trending repos).
    # Behavior: holds the token, an HTTP session with User-Agent and optional bearer header, a
    # mapping from username to UserStats and a log list; methods scan repos and export results.
    # Side effects: network access, console output, file writes (see methods).
    """Collects engagement points from the hf-skills organization."""

    def __init__(self, token: str | None = None) -> None:
        # [FUNCTION DOC] Initialize the collector.
        # Behavior: stores the token; creates a requests session with User-Agent and, if a token is
        # given, an Authorization bearer header; initializes empty user_stats and logs.
        # Inputs: token (optional string, default none).
        # Outputs: none.
        # Exceptions: none.
        # Side effects: none beyond creating the session.
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.user_stats: dict[str, UserStats] = {}
        self.logs: list[str] = []

    def log(self, message: str) -> None:
        # [FUNCTION DOC] HELPER: Print a message and remember it.
        # Inputs: message (string).
        # Outputs: prints to stdout.
        # Exceptions: none.
        # Side effects: appends to the logs list.
        """Add a log message."""
        print(message)
        self.logs.append(message)

    def _fetch_org_members(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Get the usernames of all members of ORG_NAME.
        # Behavior: tries huggingface_hub HfApi list_organization_members with the token, keeping
        # non-empty usernames; on any failure logs and falls back to GET
        # /organizations/<org>/members, taking user, username or name from each entry and dropping
        # empty ones; if that fails too, logs and returns empty.
        # Inputs: none.
        # Outputs: list of username strings.
        # Exceptions: catches and swallows all exceptions from both attempts.
        # Side effects: network requests; logging.
        """Fetch all members of the organization."""
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            members = list(api.list_organization_members(ORG_NAME))
            usernames = [m.username for m in members if m.username]
            self.log(f"👥 Found {len(usernames)} organization members")
            return usernames
        except Exception as e:
            self.log(f"⚠️ Failed to fetch org members: {e}")
            # Fallback: try the API directly
            try:
                url = f"{API_BASE}/organizations/{ORG_NAME}/members"
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                members = response.json()
                usernames = [m.get("user") or m.get("username") or m.get("name") for m in members]
                usernames = [u for u in usernames if u]
                self.log(f"👥 Found {len(usernames)} organization members (via API)")
                return usernames
            except Exception as e2:
                self.log(f"⚠️ Fallback also failed: {e2}")
                return []

    def collect_all(self) -> dict[str, UserStats]:
        # [FUNCTION DOC] Scan every org repo and accumulate points.
        # Behavior: seeds user_stats with zero stats for each org member; lists org models, datasets
        # and spaces; for each repo (id or modelId, skipping missing) credits the author (or id
        # prefix) with repos_owned unless it is the org itself; then scans the repo's discussions.
        # Inputs: none.
        # Outputs: a shallow copy of the username-to-UserStats mapping.
        # Exceptions: none raised deliberately; sub-steps swallow request errors.
        # Side effects: network requests; logging; mutates user_stats (re-seeding resets members).
        # KNOWN ISSUE: repos listed with author=hf-skills always have the org as author, so
        # repos_owned is effectively never awarded despite the documented 1 point per repo owned.
        """Collect points from all repos in the organization."""
        self.log(f"🔍 Scanning organization: {ORG_NAME}")

        # First, get all org members and initialize their stats
        members = self._fetch_org_members()
        for username in members:
            self.user_stats[username] = UserStats(username=username)

        # Collect from all repo types
        models = self._list_repos("models")
        datasets = self._list_repos("datasets")
        spaces = self._list_repos("spaces")

        all_repos = [
            *[(r, "model") for r in models],
            *[(r, "dataset") for r in datasets],
            *[(r, "space") for r in spaces],
        ]

        self.log(f"📦 Found {len(models)} models, {len(datasets)} datasets, {len(spaces)} spaces")

        for repo_info, repo_type in all_repos:
            repo_id = repo_info.get("id") or repo_info.get("modelId")
            if not repo_id:
                continue

            # Credit repo owner
            owner = repo_info.get("author") or repo_id.split("/")[0]
            if owner and owner != ORG_NAME:
                self._add_point(owner, "repos_owned", repo_id, "repo_created")

            # Scan discussions
            self._scan_discussions(repo_id, repo_type)

        return dict(self.user_stats)

    def scan_external_repos(self, repo_types: list[str] | None = None) -> None:
        # [FUNCTION DOC] Scan trending Hub repos for PRs and discussions authored by org members.
        # Behavior: uses current user_stats keys as members (logs and returns if empty, so
        # collect_all must run first); defaults to all three repo types; for each type fetches
        # trending repos, skips ones without id and those inside the org, and scans each for members.
        # Inputs: repo_types (optional list of 'models', 'datasets', 'spaces'; default all).
        # Outputs: none.
        # Exceptions: none raised; sub-steps swallow request errors.
        # Side effects: very many network requests (two per member per repo); logging; mutates stats.
        # KNOWN ISSUE: the member set includes external contributors already added by collect_all,
        # not only org members, since it is taken from all user_stats keys.
        """Scan trending repos across the Hub for PRs by org members.

        Args:
            repo_types: List of repo types to scan. Options: "models", "datasets", "spaces".
                       If None, scans all types.
        """
        org_members = set(self.user_stats.keys())
        if not org_members:
            self.log("⚠️ No org members loaded. Run collect_all() first.")
            return

        if repo_types is None:
            repo_types = ["models", "datasets", "spaces"]

        self.log(f"🌐 Scanning trending repos for PRs by {len(org_members)} org members...")
        self.log(f"📂 Repo types: {', '.join(repo_types)}")

        for repo_type in repo_types:
            trending = self._fetch_trending(repo_type)
            self.log(f"📈 Scanning {len(trending)} trending {repo_type}...")

            for repo_info in trending:
                repo_id = repo_info.get("id") or repo_info.get("modelId")
                if not repo_id:
                    continue

                # Skip org repos (already scanned)
                if repo_id.startswith(f"{ORG_NAME}/"):
                    continue

                # Scan for PRs/discussions by each org member using author filter
                self._scan_repo_for_members(repo_id, repo_type, org_members)

    def _fetch_trending(self, repo_type: str) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Fetch trending repos of one type.
        # Behavior: GETs /<repo_type> sorted by trendingScore with limit TRENDING_LIMIT (30 s timeout).
        # Inputs: repo_type (string: models, datasets or spaces).
        # Outputs: parsed JSON (expected list of repo mappings), or empty list on request failure.
        # Exceptions: swallows request errors with a log; JSON decode errors propagate.
        # Side effects: one network request.
        """Fetch trending repos of a given type."""
        endpoint = f"{API_BASE}/{repo_type}"
        params = {"sort": "trendingScore", "limit": TRENDING_LIMIT}

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.log(f"⚠️ Failed to fetch trending {repo_type}: {e}")
            return []

    def _scan_repo_for_members(self, repo_id: str, repo_type: str, org_members: set[str]) -> None:
        # [FUNCTION DOC] HELPER: Look up each member's PRs and discussions on a single repo.
        # Behavior: splits repo id into namespace and name (returns if not exactly two parts); for
        # every member queries discussions filtered by author with type pull_request, then again
        # with type discussion.
        # Inputs: repo_id (string); repo_type (API path segment); org_members (set of usernames).
        # Outputs: none.
        # Exceptions: none.
        # Side effects: two network requests per member (plus comment fetches); mutates stats.
        """Scan a repo's discussions for activity by org members using author filter."""
        # Parse namespace and repo from repo_id
        parts = repo_id.split("/")
        if len(parts) != 2:
            return
        namespace, repo = parts

        for member in org_members:
            # Use author filter for efficient querying
            self._fetch_member_discussions(
                repo_type=repo_type,
                namespace=namespace,
                repo=repo,
                author=member,
                discussion_type="pull_request",
            )
            self._fetch_member_discussions(
                repo_type=repo_type,
                namespace=namespace,
                repo=repo,
                author=member,
                discussion_type="discussion",
            )

    def _fetch_member_discussions(
        self,
        repo_type: str,
        namespace: str,
        repo: str,
        author: str,
        discussion_type: str = "all",
    ) -> None:
        # [FUNCTION DOC] HELPER: Credit one author's discussions/PRs on a repo and their comments there.
        # Behavior: GETs /<type>/<ns>/<repo>/discussions with author, type and status=all; on request
        # failure returns silently; for each returned discussion awards prs_opened or
        # discussions_opened (activity external_pr / external_discussion) and logs it; if it has
        # comments, fetches it to count the author's comments.
        # Inputs: repo_type, namespace, repo, author (strings); discussion_type (string, default all).
        # Outputs: none.
        # Exceptions: swallows request errors silently.
        # Side effects: network requests; logging; mutates stats.
        # KNOWN ISSUE: if the API ignores the author or type filter, discussions by others are
        # credited to this author and items may be counted twice across the two type queries.
        # KNOWN ISSUE: only comments on discussions the author opened are counted externally.
        """Fetch discussions for a specific author on a repo.

        Uses: GET /api/{repoType}/{namespace}/{repo}/discussions?author={author}&type={type}
        """
        url = f"{API_BASE}/{repo_type}/{namespace}/{repo}/discussions"
        params = {
            "author": author,
            "type": discussion_type,
            "status": "all",
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return

        discussions = data.get("discussions", [])
        repo_id = f"{namespace}/{repo}"

        for discussion in discussions:
            is_pr = discussion.get("isPullRequest", False)
            disc_num = discussion.get("num")

            if is_pr:
                self._add_point(author, "prs_opened", repo_id, "external_pr", disc_num)
                self.log(f"🔀 Found PR by {author} on {repo_id}")
            else:
                self._add_point(author, "discussions_opened", repo_id, "external_discussion", disc_num)
                self.log(f"💬 Found discussion by {author} on {repo_id}")

            # Count comments on the discussion
            num_comments = discussion.get("numComments", 0)
            if num_comments > 0:
                self._fetch_discussion_comments(repo_type, namespace, repo, disc_num, author)

    def _fetch_discussion_comments(
        self,
        repo_type: str,
        namespace: str,
        repo: str,
        disc_num: int,
        target_author: str,
    ) -> None:
        # [FUNCTION DOC] HELPER: Count a target author's comments on one external discussion.
        # Behavior: GETs the discussion details; on request failure returns silently; for each
        # comment event whose author name (or fullname) equals target_author, awards comments_made
        # with activity external_comment.
        # Inputs: repo_type, namespace, repo (strings); disc_num (integer); target_author (string).
        # Outputs: none.
        # Exceptions: swallows request errors silently.
        # Side effects: one network request; mutates stats.
        # KNOWN ISSUE: the opening post is usually itself a comment event, so the author also gets a
        # comment point for the discussion they opened.
        """Fetch comments on a discussion and count those by target author."""
        url = f"{API_BASE}/{repo_type}/{namespace}/{repo}/discussions/{disc_num}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return

        repo_id = f"{namespace}/{repo}"
        events = data.get("events", [])
        for event in events:
            if event.get("type") == "comment":
                author_info = event.get("author", {}) or {}
                author = author_info.get("name") or author_info.get("fullname")
                if author == target_author:
                    self._add_point(author, "comments_made", repo_id, "external_comment", disc_num)

    def _list_repos(self, repo_type: str) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: List all org repos of one type.
        # Behavior: GETs /<repo_type> with author=ORG_NAME and limit 1000 (30 s timeout).
        # Inputs: repo_type (models, datasets or spaces).
        # Outputs: parsed JSON list of repo mappings, or empty list on request failure.
        # Exceptions: swallows request errors with a log.
        # Side effects: one network request.
        """List all repos of a given type in the organization."""
        endpoint = f"{API_BASE}/{repo_type}"
        params = {"author": ORG_NAME, "limit": 1000}

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.log(f"⚠️ Failed to list {repo_type}: {e}")
            return []

    def _scan_discussions(self, repo_id: str, repo_type: str) -> None:
        # [FUNCTION DOC] HELPER: Fetch a repo's discussions and process each one.
        # Behavior: maps singular repo type to API path (default models); GETs the discussions list
        # with limit DISCUSSION_LIMIT; on failure logs and returns; logs the count and processes each
        # discussion.
        # Inputs: repo_id (string); repo_type (model, dataset or space).
        # Outputs: none.
        # Exceptions: swallows request errors with a log.
        # Side effects: network requests; logging; mutates stats.
        """Scan all discussions for a repo and count engagement."""
        # Map repo type to API path
        type_map = {"model": "models", "dataset": "datasets", "space": "spaces"}
        api_type = type_map.get(repo_type, "models")

        url = f"{API_BASE}/{api_type}/{repo_id}/discussions"

        try:
            response = self.session.get(url, params={"limit": DISCUSSION_LIMIT}, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            self.log(f"⚠️ Failed to get discussions for {repo_id}: {e}")
            return

        discussions = data.get("discussions", [])
        if not discussions:
            return

        self.log(f"💬 {repo_id}: found {len(discussions)} discussions")

        for discussion in discussions:
            self._process_discussion(repo_id, api_type, discussion)

    def _process_discussion(self, repo_id: str, api_type: str, discussion: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Award points for one discussion's opener and its comments.
        # Behavior: reads author name (or fullname), PR flag and number; if author exists and is not
        # the org, awards prs_opened or discussions_opened; if a number exists, fetches comments.
        # Inputs: repo_id (string); api_type (API path segment); discussion (mapping).
        # Outputs: none.
        # Exceptions: none.
        # Side effects: network request for comments; mutates stats.
        # KNOWN ISSUE: a discussion numbered 0 would be skipped by the truthiness check (numbers
        # normally start at 1).
        """Process a single discussion and its comments."""
        author_info = discussion.get("author", {}) or {}
        author = author_info.get("name") or author_info.get("fullname")
        is_pr = discussion.get("isPullRequest", False)
        disc_num = discussion.get("num")

        if author and author != ORG_NAME:
            activity_type = "pr_opened" if is_pr else "discussion_opened"
            point_type = "prs_opened" if is_pr else "discussions_opened"
            self._add_point(author, point_type, repo_id, activity_type, disc_num)

        # Fetch discussion details to get comments
        if disc_num:
            self._fetch_comments(repo_id, api_type, disc_num)

    def _fetch_comments(self, repo_id: str, api_type: str, disc_num: int) -> None:
        # [FUNCTION DOC] HELPER: Award a comment point for every comment on an org discussion.
        # Behavior: GETs the discussion details; silently returns on request failure; for each event
        # of type comment with an author other than the org, awards comments_made (activity comment).
        # Inputs: repo_id (string); api_type (string); disc_num (integer).
        # Outputs: none.
        # Exceptions: swallows request errors silently.
        # Side effects: one network request; mutates stats.
        # KNOWN ISSUE: the opening post is typically a comment event, so openers are double-counted
        # (discussion/PR point plus a comment point).
        """Fetch and count comments on a discussion."""
        url = f"{API_BASE}/{api_type}/{repo_id}/discussions/{disc_num}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            # Silently skip failed comment fetches
            return

        events = data.get("events", [])
        for event in events:
            event_type = event.get("type")
            # Count comments (not the initial post, status changes, etc.)
            if event_type == "comment":
                author_info = event.get("author", {}) or {}
                author = author_info.get("name") or author_info.get("fullname")
                if author and author != ORG_NAME:
                    self._add_point(author, "comments_made", repo_id, "comment", disc_num)

    def _add_point(
        self,
        username: str,
        point_type: str,
        repo_id: str,
        activity_type: str,
        disc_num: int | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Increment a counter for a user and record the activity.
        # Behavior: ignores empty usernames; creates non-member stats for unknown users; adds 1 to the
        # named counter attribute (treating missing as 0); appends an activity with type, repo_id,
        # discussion_num and current UTC timestamp.
        # Inputs: username, point_type (counter attribute name), repo_id, activity_type (strings);
        # disc_num (optional integer, default none).
        # Outputs: none.
        # Exceptions: none.
        # Side effects: mutates user_stats.
        # KNOWN ISSUE: an unknown point_type would silently create a new attribute instead of failing.
        """Add a point to a user's stats."""
        if not username:
            return

        # Initialize stats for users not in the org (external contributors)
        if username not in self.user_stats:
            self.user_stats[username] = UserStats(username=username, is_org_member=False)

        stats = self.user_stats[username]
        current = getattr(stats, point_type, 0)
        setattr(stats, point_type, current + 1)

        stats.activities.append(
            {
                "type": activity_type,
                "repo_id": repo_id,
                "discussion_num": disc_num,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_leaderboard(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC] Return all users' stats as mappings sorted by total points, highest first.
        # Inputs: none.
        # Outputs: list of mappings.
        # Exceptions: none.
        # Side effects: none.
        """Get the leaderboard sorted by total points."""
        leaderboard = [stats.to_dict() for stats in self.user_stats.values()]
        leaderboard.sort(key=lambda x: x["total_points"], reverse=True)
        return leaderboard

    def save_json(self, filepath: str) -> None:
        # [FUNCTION DOC] Write the leaderboard to a local JSON file.
        # Behavior: writes generated_at, organization, total_participants and leaderboard, indented
        # by 2; logs the path.
        # Inputs: filepath (string).
        # Outputs: file written (overwritten).
        # Exceptions: propagates file system errors.
        # Side effects: file write; logging.
        """Save the leaderboard to a JSON file."""
        leaderboard = self.get_leaderboard()
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "organization": ORG_NAME,
            "total_participants": len(leaderboard),
            "leaderboard": leaderboard,
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        self.log(f"💾 Saved leaderboard to {filepath}")

    def push_to_hub(self, repo_id: str = "hf-skills/hackers-leaderboard") -> None:
        # [FUNCTION DOC] Upload the leaderboard and metadata to a Hugging Face dataset repo.
        # Behavior: imports HfApi (logs and returns if missing); builds JSONL rows and metadata
        # (generated_at, organization, total_participants, total_points); creates the dataset repo
        # if needed and uploads data/leaderboard.jsonl and data/metadata.json; logs success.
        # Inputs: repo_id (string, default hf-skills/hackers-leaderboard).
        # Outputs: none.
        # Exceptions: catches ImportError and any upload exception, logging them only.
        # Side effects: network repo creation and commits; logging.
        # KNOWN ISSUE: HfApi is created without the collector's token, so it relies on a cached
        # login or HF_TOKEN in the environment.
        """Push the leaderboard data to a HF dataset."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            self.log("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
            return

        api = HfApi()
        leaderboard = self.get_leaderboard()

        # Create dataset as JSONL
        jsonl_content = "\n".join(json.dumps(row) for row in leaderboard)

        # Also create a metadata file
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "organization": ORG_NAME,
            "total_participants": len(leaderboard),
            "total_points": sum(row["total_points"] for row in leaderboard),
        }

        try:
            # Create repo if it doesn't exist
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            self.log(f"📁 Ensured dataset repo exists: {repo_id}")

            # Upload leaderboard data
            api.upload_file(
                path_or_fileobj=jsonl_content.encode(),
                path_in_repo="data/leaderboard.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update leaderboard - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            )

            # Upload metadata
            api.upload_file(
                path_or_fileobj=json.dumps(metadata, indent=2).encode(),
                path_in_repo="data/metadata.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update metadata - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            )

            self.log(f"🚀 Pushed leaderboard to {repo_id}")
        except Exception as e:
            self.log(f"❌ Failed to push to hub: {e}")


def main() -> None:
    # [FUNCTION DOC] CLI entry point: collect, print, save and optionally push the hackers leaderboard.
    # Behavior: parses arguments; reads HF_TOKEN (warns if absent); runs collect_all, then the
    # external scan if requested; prints a banner and up to 20 ranked users with points and
    # counter breakdown, a remaining count, totals; saves JSON; pushes if requested.
    # Inputs: command-line arguments (see module doc).
    # Outputs: console output, JSON file, optional upload. Exit code 0 unless an uncaught
    # exception occurs (argparse exits 2 on bad arguments).
    # Exceptions: none caught here.
    # Side effects: env var read, network, file write, stdout.
    parser = argparse.ArgumentParser(description="Collect engagement points from hf-skills organization")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push results to HF dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="leaderboard.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="hf-skills/hackers-leaderboard",
        help="HF dataset repo ID for pushing",
    )
    parser.add_argument(
        "--scan-external",
        action="store_true",
        help="Also scan trending repos for PRs/discussions by org members",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        nargs="+",
        choices=["models", "datasets", "spaces"],
        default=None,
        help="Repo types to scan (for --scan-external). Default: all types",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️ No HF_TOKEN found. Some requests may be rate-limited.")

    collector = PointsCollector(token=token)
    collector.collect_all()

    # Optionally scan external repos for member activity
    if args.scan_external:
        collector.scan_external_repos(repo_types=args.repo_type)

    # Print leaderboard
    print("\n" + "=" * 50)
    print("🏆 HACKERS LEADERBOARD")
    print("=" * 50)

    leaderboard = collector.get_leaderboard()
    for i, entry in enumerate(leaderboard[:20], 1):
        print(
            f"{i:2}. {entry['username']:20} - {entry['total_points']:4} points "
            f"(💬{entry['discussions_opened']} 📝{entry['comments_made']} "
            f"🔀{entry['prs_opened']} 📦{entry['repos_owned']})"
        )

    if len(leaderboard) > 20:
        print(f"   ... and {len(leaderboard) - 20} more participants")

    print("=" * 50)
    print(f"Total participants: {len(leaderboard)}")
    print(f"Total points awarded: {sum(e['total_points'] for e in leaderboard)}")

    # Save locally
    collector.save_json(args.output)

    # Push to hub if requested
    if args.push_to_hub:
        collector.push_to_hub(args.repo_id)


if __name__ == "__main__":
    main()
