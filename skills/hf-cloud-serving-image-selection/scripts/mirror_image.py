#!/usr/bin/env python3
# [MODULE DOC] CLI script that mirrors an ECR Public container image into a private ECR repository using
# the aws and docker CLIs (for SageMaker in a VPC without NAT).
# Behavior: checks tools and the docker daemon, determines the tag, account id and region,
#   creates the private repo if missing (scan on push), skips if the tag already exists there,
#   otherwise logs docker into public.ecr.aws (always via us-east-1) and the private registry,
#   pulls, re-tags and pushes the image, and prints the private URI.
# CLI arguments: required public image URI and private repository name; optional tag override.
# Env vars: AWS_REGION, then AWS_DEFAULT_REGION, for the target region; subprocesses inherit the
#   shell's AWS and docker configuration.
# Constants: none.
# Top-level executed code: the main-guard that runs main and exits with its return code.
# Outputs: private image URI on stdout; logs on stderr. Exit codes: 0 success or already
#   present, 1 any failure (missing tool, daemon down, no tag, auth/pull/push failure), 64 usage.
"""Mirror an ECR Public image to a private ECR repo. Cross-platform.

Needed when SageMaker runs in a VPC without NAT gateway: SageMaker can't pull
from public.ecr.aws in that case, so the image must be in private ECR.

Requires: docker daemon, aws CLI with ecr/ecr-public permissions.
Idempotent: skips push if the tag already exists in the private repo.

Runs `aws`/`docker` from the current shell, inheriting its AWS context — no
Bash/WSL context-sharing problem on Windows.

Usage:
    python mirror_image.py <public-image-uri> <private-repo-name> [<tag-override>]

Prints the resulting private URI to stdout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def log(msg: str) -> None:
    # [FUNCTION DOC] HELPER: Write a "[mirror_image]"-prefixed message to stderr.
    # Behavior: prints the prefixed message to stderr and flushes.
    # Inputs: msg - text.
    # Outputs: none returned; one stderr line.
    # Exceptions: none.
    # Side effects: writes to stderr.
    print(f"[mirror_image] {msg}", file=sys.stderr, flush=True)


def require(cmd: str) -> str:
    # [FUNCTION DOC] HELPER: Resolve an executable on PATH or exit.
    # Behavior: looks up the command; if missing logs an error and exits the process with code 1.
    # Inputs: cmd - executable name.
    # Outputs: the executable path.
    # Exceptions: terminates via system exit (code 1) when not found.
    # Side effects: PATH lookup; stderr logging; may exit the process.
    exe = shutil.which(cmd)
    if not exe:
        log(f"ERROR: {cmd} not installed or not on PATH")
        sys.exit(1)
    return exe


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Run a command capturing text output, without checking its exit status.
    # Behavior: runs the command with stdout/stderr captured as text, forwarding extra keyword
    #   options to the subprocess call.
    # Inputs: args - command and arguments; extra keyword options for the subprocess call.
    # Outputs: the completed-process result.
    # Exceptions: none caught.
    # Side effects: spawns a subprocess.
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def resolve_region() -> str:
    # [FUNCTION DOC] HELPER: Determine the AWS region from env vars or the aws CLI profile.
    # Behavior: returns the first non-empty of AWS_REGION and AWS_DEFAULT_REGION; otherwise runs
    #   "aws configure get region" and returns its output if it succeeds, else "".
    # Inputs: none.
    # Outputs: region string, possibly empty.
    # Exceptions: none caught.
    # Side effects: reads environment variables; may spawn an aws subprocess.
    # KNOWN ISSUE: it re-looks up aws on PATH instead of using the already resolved path; if aws
    #   were missing this would pass a null program name (in practice main checks aws first).
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        if os.environ.get(var):
            return os.environ[var]
    proc = run([shutil.which("aws"), "configure", "get", "region"])
    return proc.stdout.strip() if proc.returncode == 0 else ""


def main() -> int:
    # [FUNCTION DOC] Mirror the image and print the private URI; the script entry point.
    # Behavior: 1) require two arguments (return 64 otherwise); 2) resolve aws and docker, fail if
    #   "docker info" fails; 3) tag = override, else the text after the last ":" if the final path
    #   segment contains one, else fail (no implicit latest); 4) get the account id via STS;
    #   5) resolve the region; build "<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>";
    #   6) create the repo if describe fails; 7) if the tag exists in the repo, print the URI and
    #   return 0; 8) log in to public.ecr.aws and the private registry with passwords from aws;
    #   9) pull, tag, push (tag result unchecked); print the URI and return 0.
    # Inputs: none (reads the process command-line arguments).
    # Outputs: returns 0, 1 or 64; private URI on stdout; logs on stderr.
    # Exceptions: none caught.
    # Side effects: aws subprocesses (may create an ECR repository), docker login/pull/tag/push
    #   (network, local image store, stores registry credentials in docker config).
    # KNOWN ISSUE: a digest reference (name@sha256:...) is treated as having a tag, yielding the hash
    #   as the tag; the create-repository and docker tag results are never checked.
    if len(sys.argv) < 3:
        log(f"Usage: {os.path.basename(sys.argv[0])} <public-image-uri> <private-repo-name> [<tag-override>]")
        return 64

    public_uri = sys.argv[1]
    private_repo = sys.argv[2]
    tag_override = sys.argv[3] if len(sys.argv) > 3 else ""

    aws = require("aws")
    docker = require("docker")
    if run([docker, "info"]).returncode != 0:
        log("ERROR: docker daemon not running")
        return 1

    # Extract tag
    if tag_override:
        tag = tag_override
    elif ":" in public_uri.rsplit("/", 1)[-1]:
        tag = public_uri.rsplit(":", 1)[-1]
    else:
        log("ERROR: public URI has no tag — refusing implicit ':latest'")
        return 1

    acct = run([aws, "sts", "get-caller-identity", "--query", "Account", "--output", "text"])
    if acct.returncode != 0:
        log("ERROR: 'aws sts get-caller-identity' failed. Configure AWS credentials.")
        return 1
    account_id = acct.stdout.strip()

    region = resolve_region()
    if not region:
        log("ERROR: no AWS region. Set AWS_REGION or configure profile.")
        return 1

    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    private_uri = f"{registry}/{private_repo}:{tag}"

    log(f"Public : {public_uri}")
    log(f"Private: {private_uri}")

    reg = ["--region", region]

    # Create private repo if missing
    if run([aws, "ecr", "describe-repositories", "--repository-names", private_repo, *reg]).returncode != 0:
        log(f"Creating private ECR repo: {private_repo}")
        run(
            [aws, "ecr", "create-repository", "--repository-name", private_repo,
             "--image-scanning-configuration", "scanOnPush=true", *reg]
        )

    # Skip if tag already exists in private
    if run(
        [aws, "ecr", "describe-images", "--repository-name", private_repo,
         "--image-ids", f"imageTag={tag}", *reg]
    ).returncode == 0:
        log(f"Tag '{tag}' already in private ECR — skipping pull/push")
        print(private_uri)
        return 0

    # Auth. ECR Public auth always uses us-east-1.
    def docker_login(password: str, registry_host: str) -> bool:
        # [FUNCTION DOC] HELPER: Log docker into a registry using a password fed on stdin.
        # Behavior: runs "docker login --username AWS --password-stdin <registry>" with the password as
        #   input, capturing output.
        # Inputs: password - registry password text; registry_host - registry hostname.
        # Outputs: true if login succeeded, else false.
        # Exceptions: none caught.
        # Side effects: spawns docker; stores credentials in the docker client config.
        login = subprocess.run(
            [docker, "login", "--username", "AWS", "--password-stdin", registry_host],
            input=password, capture_output=True, text=True,
        )
        return login.returncode == 0

    pub_pw = run([aws, "ecr-public", "get-login-password", "--region", "us-east-1"])
    if pub_pw.returncode != 0 or not docker_login(pub_pw.stdout, "public.ecr.aws"):
        log("ERROR: failed to authenticate to public.ecr.aws")
        return 1

    priv_pw = run([aws, "ecr", "get-login-password", *reg])
    if priv_pw.returncode != 0 or not docker_login(priv_pw.stdout, registry):
        log(f"ERROR: failed to authenticate to {registry}")
        return 1

    log(f"Pulling {public_uri} (may take several minutes)...")
    if subprocess.run([docker, "pull", public_uri]).returncode != 0:
        log("ERROR: docker pull failed")
        return 1
    subprocess.run([docker, "tag", public_uri, private_uri])
    log("Pushing to private ECR...")
    if subprocess.run([docker, "push", private_uri]).returncode != 0:
        log("ERROR: docker push failed")
        return 1

    log("Done.")
    print(private_uri)
    return 0


if __name__ == "__main__":
    sys.exit(main())
