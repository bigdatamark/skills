#!/usr/bin/env python3
# [MODULE DOC] CLI script that creates a SageMaker execution IAM role with a trust policy and an
# inline minimum-permissions policy, using the aws CLI from the current shell.
# Behavior: refuses if the role already exists, warns for SSO callers, creates the role from
#   references/trust-policy.json, attaches references/minimum-permissions.json as the inline
#   policy "SageMakerDeploymentMinimum" (substituting the model bucket placeholder when a bucket
#   is given), then prints the new role's ARN.
# CLI arguments: required role name; optional S3 bucket name for model artifacts.
# Env vars: none read directly; aws subprocesses inherit the shell's AWS configuration.
# Constants: REFERENCES - the skill's references directory; TRUST_POLICY and
#   PERMISSIONS_TEMPLATE - JSON policy files inside it.
# Top-level executed code: path constants are computed at import; the main-guard runs main and
#   exits with its return code.
# Outputs: role ARN on stdout; logs on stderr. Exit codes: 0 created, 1 role exists or an aws
#   command failed, 2 aws CLI missing, 64 usage error.
"""Create a SageMaker execution role. Cross-platform (Windows/macOS/Linux).

Run only after check_role.py confirms no usable role exists.
Requires iam:CreateRole and iam:PutRolePolicy.

SSO principals typically lack these — this script will fail with AccessDenied
in that case, and the right answer is to ask an AWS admin for a role.

Like check_role.py, this calls `aws` from the current shell so it inherits the
same AWS context (profile, region, SSO session, proxy). Policy documents are
passed inline rather than via `file://` paths, which sidesteps Windows path
translation entirely.

Usage:
    python create_role.py <role-name> [<model-s3-bucket>]

Without bucket, the inline policy keeps a placeholder for later editing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REFERENCES = Path(__file__).resolve().parent.parent / "references"
TRUST_POLICY = REFERENCES / "trust-policy.json"
PERMISSIONS_TEMPLATE = REFERENCES / "minimum-permissions.json"



def log(msg: str) -> None:
    # [FUNCTION DOC] HELPER: Write a "[create_role]"-prefixed message to stderr.
    # Behavior: prints the prefixed message to stderr and flushes.
    # Inputs: msg - text to log.
    # Outputs: none returned; one stderr line.
    # Exceptions: none.
    # Side effects: writes to stderr.
    print(f"[create_role] {msg}", file=sys.stderr, flush=True)


def aws_bin() -> str:
    # [FUNCTION DOC] HELPER: Resolve the aws CLI executable path or exit.
    # Behavior: looks up "aws" on PATH; if absent logs an error and exits the process with code 2.
    # Inputs: none.
    # Outputs: the aws executable path.
    # Exceptions: terminates via system exit (code 2) when not found.
    # Side effects: PATH lookup; stderr logging; may exit the process.
    exe = shutil.which("aws")
    if not exe:
        log("ERROR: the 'aws' CLI was not found on PATH. Install AWS CLI v2.")
        sys.exit(2)
    return exe


def run_aws(args: list[str], check: bool = False) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Run an aws CLI command, optionally exiting on failure.
    # Behavior: runs aws with the arguments, capturing text output; if check is true and the exit
    #   code is nonzero, logs the command and its stderr and exits the process with code 1.
    # Inputs: args - CLI argument strings after "aws"; check - exit on failure when true
    #   (default false).
    # Outputs: the completed-process result.
    # Exceptions: terminates via system exit (code 1) on checked failure; aws_bin may exit (code 2).
    # Side effects: spawns an aws subprocess (AWS API calls); stderr logging.
    proc = subprocess.run([aws_bin(), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        log(f"AWS command failed: aws {' '.join(args)}")
        if proc.stderr.strip():
            log(proc.stderr.strip())
        sys.exit(1)
    return proc


def main() -> int:
    # [FUNCTION DOC] Create the role and attach its inline policy; the script entry point.
    # Behavior: 1) require a role name (return 64 with usage otherwise); optional bucket argument;
    #   2) if "iam get-role" succeeds, report it exists and return 1; 3) get the caller ARN (checked)
    #   and warn if it is an AWS SSO reserved role; 4) read the trust policy file and run
    #   "iam create-role" with it inline (checked); 5) read the permissions template, replace
    #   REPLACE_WITH_MODEL_BUCKET with the bucket if given (else warn about the placeholder);
    #   6) run "iam put-role-policy" (checked); 7) fetch the role ARN (checked), log and print it.
    # Inputs: none (reads the process command-line arguments).
    # Outputs: returns 0, 1 or 64; ARN on stdout; logs on stderr.
    # Exceptions: none caught; file read errors propagate; checked aws failures exit with code 1.
    # Side effects: reads policy files; creates an IAM role and inline policy in the AWS account.
    # KNOWN ISSUE: any get-role failure (including AccessDenied or network errors) is treated as
    #   "role does not exist", and if put-role-policy fails after create-role the role is left
    #   half-configured with no rollback.
    if len(sys.argv) < 2:
        log(f"Usage: {Path(sys.argv[0]).name} <role-name> [<model-s3-bucket>]")
        return 64

    role_name = sys.argv[1]
    model_bucket = sys.argv[2] if len(sys.argv) > 2 else ""

    if run_aws(["iam", "get-role", "--role-name", role_name]).returncode == 0:
        log(f"Role '{role_name}' already exists. Use check_role.py to validate it.")
        return 1

    ident = run_aws(["sts", "get-caller-identity", "--query", "Arn", "--output", "text"], check=True)
    caller_arn = ident.stdout.strip()
    if ":assumed-role/AWSReservedSSO_" in caller_arn:
        log(f"WARNING: SSO caller ({caller_arn}) — IAM creation likely to fail with AccessDenied.")
        log("If it does, ask an AWS admin to create the role.")

    trust_doc = TRUST_POLICY.read_text(encoding="utf-8")

    log(f"Creating role: {role_name}")
    run_aws(
        [
            "iam", "create-role",
            "--role-name", role_name,
            "--assume-role-policy-document", trust_doc,
            "--description", "SageMaker execution role (hf-cloud-sagemaker-iam-preflight skill)",
        ],
        check=True,
    )


    inline_policy = PERMISSIONS_TEMPLATE.read_text(encoding="utf-8")
    if model_bucket:
        inline_policy = inline_policy.replace("REPLACE_WITH_MODEL_BUCKET", model_bucket)
        log(f"Inline policy will grant S3 access to: {model_bucket}")
    else:
        log("WARNING: No model bucket specified — policy contains placeholder.")
        log("Update before deployment, or pass the bucket name as the 2nd argument.")

    log("Attaching inline policy: SageMakerDeploymentMinimum")
    run_aws(
        [
            "iam", "put-role-policy",
            "--role-name", role_name,
            "--policy-name", "SageMakerDeploymentMinimum",
            "--policy-document", inline_policy,
        ],
        check=True,
    )

    arn = run_aws(
        ["iam", "get-role", "--role-name", role_name, "--query", "Role.Arn", "--output", "text"],
        check=True,
    ).stdout.strip()
    log(f"Created: {arn}")
    print(arn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
