#!/usr/bin/env python3
# [MODULE DOC] CLI script that discovers or validates a SageMaker execution IAM role via the aws CLI.
# Behavior: verifies AWS credentials with "sts get-caller-identity"; with an argument, validates
#   that one role (name or ARN); without, lists all account roles, filters by name patterns,
#   ranks by last-used date then creation date (newest first) and picks the first whose trust
#   policy mentions sagemaker.amazonaws.com.
# CLI arguments: optional role name or full IAM role ARN (the name after the last slash is used).
# Env vars: none read directly; the aws subprocess inherits the shell's AWS profile, region,
#   SSO session and proxy settings.
# Constants: ROLE_NAME_PATTERNS - case-sensitive shell-style patterns for candidate role names.
# Top-level executed code: the main-guard that calls main and exits with its return code.
# Outputs: the chosen role ARN on stdout; diagnostics on stderr. Exit codes: 0 usable role found,
#   1 none usable, 2 aws CLI missing or credentials failure.
"""Discover and validate a SageMaker execution role. Cross-platform (Windows/macOS/Linux).

Without args: searches the account for usable roles, ranks by last-used date
(newest creation date when last-used is unrecorded), returns the first one
with a valid trust policy.
With arg: validates the named role.

Does NOT create roles — see create_role.py.

Why Python (not a shell script): this runs `aws` from the *same* shell the user
launched it in, inheriting their AWS profile, region, SSO session, proxy, and
credential chain. A bundled Bash helper on Windows (Git Bash/WSL/MSYS) often
does NOT share that context, so `aws sts get-caller-identity` fails there even
when the native CLI works fine. Run this from the shell where `aws sts
get-caller-identity` already succeeds.

Usage:
    python check_role.py                          # discover
    python check_role.py <role-name-or-arn>       # validate a specific role

(On Windows the launcher is usually `python`; on macOS/Linux it's `python3`.)

Exit: 0 = usable role found (ARN printed to stdout)
      1 = no usable role
      2 = AWS CLI / credentials error
"""

from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
import sys

ROLE_NAME_PATTERNS = [
    "AmazonSageMaker-ExecutionRole-*",
    "SageMakerExecutionRole*",
    "*SageMaker*Execution*",
    "*sagemaker*execution*",
]


def log(msg: str) -> None:
    # [FUNCTION DOC] HELPER: Write a "[check_role]"-prefixed message to stderr.
    # Behavior: prints the prefixed message to stderr and flushes.
    # Inputs: msg - text to log.
    # Outputs: none returned; one stderr line.
    # Exceptions: none.
    # Side effects: writes to stderr.
    print(f"[check_role] {msg}", file=sys.stderr, flush=True)


def aws_bin() -> str:
    # [FUNCTION DOC] HELPER: Resolve the aws CLI executable path or exit.
    # Behavior: searches PATH (honoring Windows executable extensions) for "aws"; if missing, logs
    #   install guidance and exits the process with code 2.
    # Inputs: none.
    # Outputs: the absolute path of the aws executable.
    # Exceptions: terminates via system exit (code 2) when not found.
    # Side effects: PATH lookup; stderr logging; may exit the process.
    """Resolve the `aws` executable, honoring PATHEXT (finds aws.exe/aws.cmd on Windows)."""
    exe = shutil.which("aws")
    if not exe:
        log("ERROR: the 'aws' CLI was not found on PATH.")
        log("Install AWS CLI v2 and confirm `aws sts get-caller-identity` works in this shell.")
        sys.exit(2)
    return exe


def run_aws(args: list[str]) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Run an aws CLI command and return the completed process.
    # Behavior: resolves the aws executable and runs it with the given arguments, capturing stdout
    #   and stderr as text; does not check the exit status.
    # Inputs: args - list of CLI argument strings following "aws".
    # Outputs: the completed-process result (return code, stdout, stderr).
    # Exceptions: none caught; aws_bin may exit the process.
    # Side effects: spawns an aws subprocess (network calls to AWS).
    """Run an aws CLI command, capturing output. Never raises on nonzero exit."""
    return subprocess.run(
        [aws_bin(), *args],
        capture_output=True,
        text=True,
    )


def get_role(role_name: str) -> dict | None:
    # [FUNCTION DOC] HELPER: Fetch an IAM role description via "aws iam get-role".
    # Behavior: runs get-role with JSON output; on nonzero exit returns nothing; otherwise parses
    #   the JSON and returns its "Role" object, or nothing if parsing fails or the key is absent.
    # Inputs: role_name - IAM role name.
    # Outputs: a mapping describing the role, or none.
    # Exceptions: catches JSON decode errors and missing-key errors (returns none).
    # Side effects: aws subprocess / AWS API call.
    """Return the Role dict from `aws iam get-role`, or None if it can't be read."""
    proc = run_aws(["iam", "get-role", "--role-name", role_name, "--output", "json"])
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)["Role"]
    except (json.JSONDecodeError, KeyError):
        return None


def trust_allows_sagemaker(role: dict) -> bool:
    # [FUNCTION DOC] HELPER: Check whether a role's trust policy mentions the SageMaker service principal.
    # Behavior: takes the role's AssumeRolePolicyDocument (empty mapping if absent), serializes it to
    #   JSON text and returns true if "sagemaker.amazonaws.com" appears anywhere in it.
    # Inputs: role - mapping describing an IAM role.
    # Outputs: boolean.
    # Exceptions: none.
    # Side effects: none.
    # KNOWN ISSUE: a plain substring check also matches Deny statements, conditions, or non-Principal
    #   fields containing that text, so a role whose trust policy denies SageMaker would pass.
    """True if the trust policy lets sagemaker.amazonaws.com assume the role."""
    trust = role.get("AssumeRolePolicyDocument", {})
    # Substring check over the serialized doc — matches the original shell helper
    # and is robust to single/list Principal.Service shapes.
    return "sagemaker.amazonaws.com" in json.dumps(trust)


def validate_role(role_name: str, quiet: bool = False) -> str | None:
    # [FUNCTION DOC] HELPER: Return a role's ARN if the role exists and trusts SageMaker.
    # Behavior: fetches the role; if absent logs (unless quiet) and returns nothing; if its trust
    #   policy does not mention SageMaker logs (unless quiet) and returns nothing; otherwise returns
    #   the role's "Arn" field.
    # Inputs: role_name - IAM role name; quiet - suppress log messages when true (default false).
    # Outputs: the role ARN string, or none.
    # Exceptions: none caught.
    # Side effects: aws subprocess; stderr logging.
    """Return the role ARN if it exists and trusts SageMaker, else None."""
    role = get_role(role_name)
    if role is None:
        if not quiet:
            log(f"Role '{role_name}' does not exist or you cannot describe it.")
        return None
    if not trust_allows_sagemaker(role):
        if not quiet:
            log(f"Role '{role_name}': trust policy does not allow sagemaker.amazonaws.com")
        return None
    return role.get("Arn")


def main() -> int:
    # [FUNCTION DOC] Discover or validate a SageMaker execution role; the script entry point.
    # Behavior: 1) run "sts get-caller-identity", returning 2 on failure; 2) if an argument is given,
    #   strip an IAM ARN down to the role name, validate it, print the ARN and return 0 or return 1;
    #   3) otherwise list all role names (return 1 if listing fails, empty list on bad JSON), keep
    #   those matching any ROLE_NAME_PATTERNS, return 1 if none; 4) fetch each candidate once
    #   (skipping unreadable ones), build (last-used date, create date, name) triples with empty
    #   strings for missing dates, sort descending and log the ranking; 5) return the first whose
    #   trust policy mentions SageMaker (print ARN, return 0), else log advice and return 1.
    # Inputs: none (reads the process command-line arguments).
    # Outputs: ARN on stdout on success; return codes 0, 1 or 2 as described in the module block.
    # Exceptions: catches JSON decode errors from list-roles; aws_bin may exit with code 2.
    # Side effects: multiple aws subprocesses (IAM/STS API calls); stderr logging.
    # KNOWN ISSUE: the "quiet" option of validate_role is never used, and the discovery path does
    #   not call validate_role at all (it duplicates the trust check inline).
    if run_aws(["sts", "get-caller-identity"]).returncode != 0:
        log("ERROR: 'aws sts get-caller-identity' failed. Run hf-cloud-aws-context-discovery first.")
        log("Run this helper from the shell where the AWS CLI is configured (e.g. PowerShell on Windows).")
        return 2

    # Path 1: validate a user-supplied role
    if len(sys.argv) >= 2:
        supplied = sys.argv[1]
        role_name = supplied.rsplit("/", 1)[-1] if supplied.startswith("arn:aws:iam::") else supplied
        log(f"Validating: {role_name}")
        arn = validate_role(role_name)
        if arn:
            log(f"OK: {arn}")
            print(arn)
            return 0
        return 1

    # Path 2: discover candidates
    log("Searching for SageMaker execution roles in the account...")
    proc = run_aws(["iam", "list-roles", "--query", "Roles[*].RoleName", "--output", "json"])
    if proc.returncode != 0:
        log("Could not list roles (caller likely lacks iam:ListRoles).")
        log("Ask the user for an existing role ARN, or have someone with IAM access run this.")
        return 1
    try:
        all_roles = json.loads(proc.stdout) or []
    except json.JSONDecodeError:
        all_roles = []

    candidates = [
        name
        for name in all_roles
        if any(fnmatch.fnmatchcase(name, pat) for pat in ROLE_NAME_PATTERNS)
    ]

    if not candidates:
        log("No matching roles. Options:")
        log("  1. Ask the user for an ARN (role might have an unusual name)")
        log("  2. Create one (see create_role.py, requires iam:CreateRole)")
        return 1

    log(f"Found {len(candidates)} candidate(s): {' '.join(candidates)}")

    # Rank by RoleLastUsed (most recent first) — alphabetical order rarely picks
    # the actively-maintained role in accounts with multiple SageMaker roles.
    # Fetch each role once and reuse the result for both ranking and validation.
    log("Ranking by last-used date (fallback: creation date)...")
    roles_by_name: dict[str, dict] = {}
    ranked: list[tuple[str, str, str]] = []  # (last_used, create_date, role_name)
    for name in candidates:
        role = get_role(name)
        if role is None:
            continue
        roles_by_name[name] = role
        last_used = (role.get("RoleLastUsed") or {}).get("LastUsedDate") or ""
        # IAM often reports no RoleLastUsed at all (tracking only covers recent
        # activity); when every candidate ties at "", newest CreateDate wins.
        create_date = str(role.get("CreateDate") or "")
        # ISO-8601 timestamps sort chronologically as strings; "" (never used)
        # sorts before any timestamp, so descending order puts it last.
        ranked.append((last_used, create_date, name))

    ranked.sort(reverse=True)

    log("Ranking (most recent first):")
    for last_used, create_date, name in ranked:
        when = f"last used {last_used}" if last_used else f"never used, created {create_date or '?'}"
        log(f"  - {name} ({when})")

    for _, _, name in ranked:
        if trust_allows_sagemaker(roles_by_name[name]):
            arn = roles_by_name[name].get("Arn")
            log(f"Using: {arn}")
            print(arn)
            return 0

    log("No candidate passed validation (they exist but lack correct trust policy).")
    log("Fix the trust policy or create a new role (see create_role.py).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
