# [MODULE DOC] HELPER: Shared library for the SageMaker deployment scripts (deploy.py, deploy_async.py,
# deploy_ic.py): logging, env-var flag parsing, endpoint naming, the common tag schema, idempotent
# model creation, and polling an endpoint until it is in service.
# Behavior: import-only; defines functions and one constant; no top-level work besides imports.
# CLI arguments: none. Env vars: none.
# Constants: CREATED_BY_TAG_VALUE - value of the "CreatedBy" tag put on every created resource,
#   used to find all resources these scripts made.
# Side effects: none at import (requires botocore to be installed).
"""Shared helpers for the deployment scripts (deploy.py, deploy_async.py).

Anything that should look identical between real-time and async deployments
lives here — tag schema, name format, env-var parsing, model creation,
endpoint waiting. Mode-specific logic (endpoint config shape, autoscaling
policies, alarm metrics) stays in the calling script.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Any

from botocore.exceptions import ClientError


# Shared tagging convention. CreatedBy lets us find every resource these
# scripts have ever created via:
#   aws resourcegroupstaggingapi get-resources \
#     --tag-filters Key=CreatedBy,Values=agentic-deploy-skills
CREATED_BY_TAG_VALUE = "agentic-deploy-skills"


def log(prefix: str, msg: str) -> None:
    # [FUNCTION DOC] HELPER: Write a log line with a per-script prefix to stderr.
    # Behavior: prints "[prefix] msg" to stderr and flushes.
    # Inputs: prefix - script label; msg - text.
    # Outputs: none returned; one stderr line.
    # Exceptions: none.
    # Side effects: writes to stderr.
    """Stream a log line to stderr with a per-script prefix."""
    print(f"[{prefix}] {msg}", file=sys.stderr, flush=True)


def parse_env(env_args: list[str]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Turn a list of KEY=VALUE strings into a name-to-value mapping.
    # Behavior: for each item, requires an "=" and splits on the first one; later duplicate keys
    #   overwrite earlier ones.
    # Inputs: env_args - list of "KEY=VALUE" strings (from repeated --env flags).
    # Outputs: mapping from key to value (values may contain "=" and may be empty).
    # Exceptions: raises a system exit with an explanatory message if an item lacks "=".
    # Side effects: none.
    """Parse --env KEY=VALUE flags from argparse into a dict."""
    env: dict[str, str] = {}
    for item in env_args:
        if "=" not in item:
            raise SystemExit(f"--env must be KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        env[k] = v
    return env


def make_endpoint_name(model_name: str, override: str | None) -> str:
    # [FUNCTION DOC] HELPER: Produce an endpoint name, either the override or a timestamped default.
    # Behavior: returns the override unchanged if given and non-empty; otherwise lowercases the model
    #   name, replaces underscores with hyphens, appends "-" plus the current UTC time formatted
    #   year-month-day-hour-minute, and truncates the result to 63 characters.
    # Inputs: model_name - base name; override - explicit name or none.
    # Outputs: endpoint name string.
    # Exceptions: none.
    # Side effects: reads the current clock.
    # KNOWN ISSUE: the override is not validated or truncated, and truncation to 63 characters cuts
    #   the timestamp (not the base name) and may leave a trailing hyphen; other illegal characters
    #   (dots, spaces, etc.) in the model name are not sanitized despite the docstring's claim.
    """Generate a SageMaker-legal endpoint name.

    Default format: <model-name>-<YYYYMMDD-HHMM>. SageMaker names are limited
    to 63 chars and must be DNS-friendly (lowercase, hyphens, no underscores).
    """
    if override:
        return override
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    base = model_name.replace("_", "-").lower()
    return f"{base}-{stamp}"[:63]


def build_tags(
    *,
    project: str,
    caller_arn: str,
    environment: str,
    model_s3_uri: str | None = None,
    extra: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    # [FUNCTION DOC] HELPER: Build the standard list of AWS tags applied to created resources.
    # Behavior: derives Owner as the text after the last "/" of the caller ARN (or the whole ARN);
    #   emits Project, Owner, Environment and CreatedBy tags, then ModelArtifact if a model S3 URI
    #   is given, then every extra key/value pair in order.
    # Inputs (keyword-only): project; caller_arn; environment; model_s3_uri (optional, default
    #   none); extra - optional mapping of additional tag names to values (default none).
    # Outputs: list of Key/Value tag mappings.
    # Exceptions: none.
    # Side effects: none.
    """Build the shared tag set applied to every resource we create.

    `extra` lets the caller add deployment-mode-specific tags (e.g.
    `{"InferenceMode": "async"}`) without forking the schema.
    """
    owner = caller_arn.split("/")[-1] if "/" in caller_arn else caller_arn
    tags = [
        {"Key": "Project", "Value": project},
        {"Key": "Owner", "Value": owner},
        {"Key": "Environment", "Value": environment},
        {"Key": "CreatedBy", "Value": CREATED_BY_TAG_VALUE},
    ]
    if model_s3_uri:
        tags.append({"Key": "ModelArtifact", "Value": model_s3_uri})
    if extra:
        for k, v in extra.items():
            tags.append({"Key": k, "Value": v})
    return tags


def create_model(
    sm: Any,
    *,
    model_name: str,
    image_uri: str,
    role_arn: str,
    model_s3_uri: str | None,
    env: dict[str, str],
    tags: list[dict[str, str]],
    log_prefix: str = "deploy",
) -> str:
    # [FUNCTION DOC] HELPER: Create a SageMaker Model, reusing it if one with the same name exists.
    # Behavior: logs, builds the primary container with the image, adding Environment when env is
    #   non-empty and ModelDataUrl when an S3 URI is given, then calls create_model with the name,
    #   container, execution role and tags. If the client error message contains "Cannot create
    #   already existing model" it logs reuse; any other client error is re-raised.
    # Inputs: sm - a boto3 SageMaker client; model_name; image_uri; role_arn; model_s3_uri (may be
    #   none); env - environment mapping; tags - tag list; log_prefix - label (default "deploy").
    # Outputs: the model name.
    # Exceptions: swallows the already-exists client error; re-raises other client errors.
    # Side effects: AWS API call creating a SageMaker model; stderr logging.
    # KNOWN ISSUE: an existing model is reused without checking that its image, role, data or env
    #   match the requested ones, so stale definitions silently persist.
    """Create a SageMaker Model. Idempotent — reuses on AlreadyExists.

    Returns the model name on success. The model definition itself is mode-
    independent: same call, same parameters, whether the eventual endpoint
    is real-time or async.
    """
    log(log_prefix, f"Creating model: {model_name}")
    primary_container: dict[str, Any] = {"Image": image_uri}
    if env:
        primary_container["Environment"] = env
    if model_s3_uri:
        primary_container["ModelDataUrl"] = model_s3_uri

    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer=primary_container,
            ExecutionRoleArn=role_arn,
            Tags=tags,
        )
    except ClientError as e:
        if "Cannot create already existing model" in str(e):
            log(log_prefix, f"Model {model_name} already exists — reusing")
        else:
            raise
    return model_name


def wait_for_endpoint(
    sm: Any,
    endpoint_name: str,
    timeout_minutes: int = 30,
    log_prefix: str = "deploy",
) -> None:
    # [FUNCTION DOC] HELPER: Poll an endpoint until it is InService, failing on Failed or timeout.
    # Behavior: computes a deadline from now; loop: describe the endpoint, return when status is
    #   InService, raise if Failed (with FailureReason or a placeholder), else log status and
    #   elapsed seconds and sleep 30 seconds. After the deadline passes, raise a timeout error.
    # Inputs: sm - boto3 SageMaker client; endpoint_name; timeout_minutes (default 30);
    #   log_prefix (default "deploy").
    # Outputs: none.
    # Exceptions: raises a runtime error on Failed status and a timeout error on deadline; client
    #   errors from describe propagate.
    # Side effects: repeated AWS API calls; sleeps; stderr logging.
    """Poll DescribeEndpoint until InService, or raise on Failed/timeout."""
    log(log_prefix, f"Waiting for {endpoint_name} to reach InService (up to {timeout_minutes} min)...")
    start = time.time()
    deadline = start + (timeout_minutes * 60)

    while time.time() < deadline:
        resp = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        elapsed = int(time.time() - start)

        if status == "InService":
            log(log_prefix, f"InService after {elapsed}s")
            return
        if status == "Failed":
            reason = resp.get("FailureReason", "(no reason given)")
            raise RuntimeError(f"Endpoint creation failed after {elapsed}s: {reason}")

        log(log_prefix, f"  status={status} elapsed={elapsed}s")
        time.sleep(30)

    raise TimeoutError(f"Endpoint did not reach InService within {timeout_minutes} minutes")
