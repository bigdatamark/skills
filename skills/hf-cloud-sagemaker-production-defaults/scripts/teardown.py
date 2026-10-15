#!/usr/bin/env python3
# [MODULE DOC] CLI script that deletes a SageMaker endpoint and its related resources via the aws CLI.
# Behavior: discovers the endpoint's config, models (from production variants and from inference
#   components) and inference components, then deletes in order: CloudWatch alarms (by name
#   prefix), autoscaling policies and scalable targets, inference components (with retries),
#   endpoint, endpoint config, models. Missing resources are skipped. Does not delete the IAM
#   role, data-capture S3 objects, SNS topics or model artifacts.
# CLI arguments: required endpoint name; optional region.
# Env vars: AWS_REGION, then AWS_DEFAULT_REGION, used if no region argument; aws subprocesses
#   inherit the shell's AWS configuration.
# Constants: VARIANT_DIM - scalable dimension for variant instance count; IC_DIM - scalable
#   dimension for inference-component copy count.
# Top-level executed code: the main-guard that runs main and exits with its return code.
# Outputs: logs on stderr. Exit codes: 0 done, 1 no region, 2 aws CLI missing, 64 usage error.
"""Tear down a SageMaker endpoint and its associated resources. Cross-platform.

Deletes in safe order: alarms -> autoscaling -> inference components ->
endpoint (stops billing) -> endpoint config -> model.

Handles both deployment shapes:
  - model-based endpoints (deploy.py, deploy_async.py): the variant carries the
    model, autoscaling targets endpoint/<name>/variant/AllTraffic
  - inference-component endpoints (deploy_ic.py, scale-to-zero): the component
    carries the model, autoscaling targets inference-component/<name>

Does NOT delete: IAM role, data capture S3 objects, SNS topic, model artifacts.
Idempotent — missing resources are skipped, not errors.

Calls the `aws` CLI from the current shell, so it inherits the same AWS context
(profile, region, SSO session) — no Bash/WSL context-sharing problem on Windows.

Usage:
    python teardown.py <endpoint-name> [<region>]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

VARIANT_DIM = "sagemaker:variant:DesiredInstanceCount"
IC_DIM = "sagemaker:inference-component:DesiredCopyCount"


def log(msg: str) -> None:
    # [FUNCTION DOC] HELPER: Write a "[teardown]"-prefixed message to stderr.
    # Behavior: prints the prefixed message to stderr and flushes.
    # Inputs: msg - text.
    # Outputs: none returned; one stderr line.
    # Exceptions: none.
    # Side effects: writes to stderr.
    print(f"[teardown] {msg}", file=sys.stderr, flush=True)


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


def run_aws(args: list[str]) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Run an aws CLI command, capturing text output, without checking the exit code.
    # Behavior: runs the aws executable with the given arguments.
    # Inputs: args - CLI argument strings after "aws".
    # Outputs: the completed-process result.
    # Exceptions: none caught; aws_bin may exit the process.
    # Side effects: spawns an aws subprocess (AWS API calls).
    return subprocess.run([aws_bin(), *args], capture_output=True, text=True)


def json_out(proc: subprocess.CompletedProcess, default):
    # [FUNCTION DOC] HELPER: Parse a completed aws process's stdout as JSON, falling back to a default.
    # Behavior: returns the default if the process failed, if stdout is not valid JSON, or if the
    #   parsed value is empty/falsy; otherwise returns the parsed value.
    # Inputs: proc - completed-process result; default - value to return on failure.
    # Outputs: parsed JSON value or the default.
    # Exceptions: catches JSON decode errors.
    # Side effects: none.
    if proc.returncode != 0:
        return default
    try:
        return json.loads(proc.stdout) or default
    except json.JSONDecodeError:
        return default


def resolve_region(arg_region: str | None) -> str:
    # [FUNCTION DOC] HELPER: Determine the AWS region to use.
    # Behavior: returns the argument if non-empty; else the first non-empty of AWS_REGION and
    #   AWS_DEFAULT_REGION; else the output of "aws configure get region" if that succeeds; else an
    #   empty string.
    # Inputs: arg_region - region from the command line, or none.
    # Outputs: region string, possibly empty.
    # Exceptions: none caught.
    # Side effects: reads environment variables; may spawn an aws subprocess.
    """Region from arg, then env, then the active profile's config."""
    if arg_region:
        return arg_region
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        if os.environ.get(var):
            return os.environ[var]
    proc = run_aws(["configure", "get", "region"])
    return proc.stdout.strip() if proc.returncode == 0 else ""


def delete_alarms(prefix: str, reg: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Delete all CloudWatch alarms whose names start with a prefix.
    # Behavior: lists alarm names with the prefix; if any, deletes them in one call (result not
    #   checked) and logs them.
    # Inputs: prefix - alarm name prefix; reg - region arguments appended to each aws call.
    # Outputs: none.
    # Exceptions: none caught.
    # Side effects: CloudWatch API calls via aws subprocesses; stderr logging.
    # KNOWN ISSUE: describe-alarms results and delete-alarms (max 100 names per call) are not
    #   handled for very large alarm sets, and prefix matching can delete alarms of another endpoint
    #   whose name merely starts with this endpoint's name plus a hyphen.
    alarms = json_out(
        run_aws(["cloudwatch", "describe-alarms", "--alarm-name-prefix", prefix,
                 "--query", "MetricAlarms[*].AlarmName", "--output", "json", *reg]),
        [],
    )
    if alarms:
        run_aws(["cloudwatch", "delete-alarms", "--alarm-names", *alarms, *reg])
        log(f"Deleted alarms: {' '.join(alarms)}")


def delete_scaling_stack(resource_id: str, dimension: str, reg: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Remove all autoscaling policies for a resource, then deregister its target.
    # Behavior: lists scaling policy names for the resource in the sagemaker namespace and deletes
    #   each (results unchecked, each logged); then lists scalable targets for the resource and, if
    #   it is registered, deregisters it with the given dimension and logs.
    # Inputs: resource_id - Application Auto Scaling resource id; dimension - scalable dimension;
    #   reg - region arguments.
    # Outputs: none.
    # Exceptions: none caught.
    # Side effects: Application Auto Scaling API calls via aws subprocesses; stderr logging.
    # KNOWN ISSUE: success is logged even if a delete/deregister command failed.
    """Delete every scaling policy on a target, then deregister the target."""
    policies = json_out(
        run_aws(["application-autoscaling", "describe-scaling-policies",
                 "--service-namespace", "sagemaker", "--resource-id", resource_id,
                 "--query", "ScalingPolicies[*].PolicyName", "--output", "json", *reg]),
        [],
    )
    for policy in policies:
        run_aws(["application-autoscaling", "delete-scaling-policy",
                 "--service-namespace", "sagemaker", "--resource-id", resource_id,
                 "--scalable-dimension", dimension, "--policy-name", policy, *reg])
        log(f"Deleted autoscaling policy: {policy}")

    targets = json_out(
        run_aws(["application-autoscaling", "describe-scalable-targets",
                 "--service-namespace", "sagemaker", "--resource-ids", resource_id,
                 "--query", "ScalableTargets[*].ResourceId", "--output", "json", *reg]),
        [],
    )
    if resource_id in targets:
        run_aws(["application-autoscaling", "deregister-scalable-target",
                 "--service-namespace", "sagemaker", "--resource-id", resource_id,
                 "--scalable-dimension", dimension, *reg])
        log(f"Deregistered scalable target: {resource_id}")


def list_inference_components(endpoint_name: str, reg: list[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: List inference component names attached to an endpoint.
    # Behavior: runs "sagemaker list-inference-components" filtered by endpoint name and returns the
    #   names, or an empty list on failure.
    # Inputs: endpoint_name; reg - region arguments.
    # Outputs: list of component names.
    # Exceptions: none caught.
    # Side effects: aws subprocess / SageMaker API call.
    return json_out(
        run_aws(["sagemaker", "list-inference-components",
                 "--endpoint-name-equals", endpoint_name,
                 "--query", "InferenceComponents[*].InferenceComponentName",
                 "--output", "json", *reg]),
        [],
    )


def component_model_name(ic_name: str, reg: list[str]) -> str:
    # [FUNCTION DOC] HELPER: Return the model name backing an inference component.
    # Behavior: describes the component and returns Specification.ModelName, or an empty string if
    #   unavailable.
    # Inputs: ic_name - component name; reg - region arguments.
    # Outputs: model name or empty string.
    # Exceptions: none caught.
    # Side effects: aws subprocess / SageMaker API call.
    desc = json_out(
        run_aws(["sagemaker", "describe-inference-component",
                 "--inference-component-name", ic_name, *reg]),
        {},
    )
    return desc.get("Specification", {}).get("ModelName", "")


def delete_inference_components(ic_names: list[str], reg: list[str],
                                timeout_seconds: int = 900) -> None:
    # [FUNCTION DOC] HELPER: Delete inference components, retrying until they are gone or a timeout.
    # Behavior: while some remain and before the deadline: for each pending component log once,
    #   issue a delete (result ignored), then keep only those that still describe successfully; if
    #   any remain, sleep 15 seconds. Retrying handles transient states and adapter-before-base
    #   ordering. Finally logs a warning listing leftovers, or success.
    # Inputs: ic_names - component names; reg - region arguments; timeout_seconds (default 900).
    # Outputs: none.
    # Exceptions: none caught.
    # Side effects: repeated SageMaker API calls via aws subprocesses; sleeps; stderr logging.
    # KNOWN ISSUE: any describe failure (e.g. throttling or permissions) is treated as "deleted".
    """Delete components and wait until they are gone.

    Retries the delete on every poll, because SageMaker refuses it in two
    transient states seen in practice:
      - CREATE_IN_PROGRESS (container still starting)
      - UPDATE_RC_IN_PROGRESS (a scaling action is changing the copy count)
    An adapter component must also go before its base component, and the
    ordering falls out of the same retry loop.
    """
    pending = list(ic_names)
    deadline = time.time() + timeout_seconds
    announced: set[str] = set()
    while pending and time.time() < deadline:
        for ic_name in pending:
            if ic_name not in announced:
                log(f"Deleting inference component: {ic_name}")
                announced.add(ic_name)
            run_aws(["sagemaker", "delete-inference-component",
                     "--inference-component-name", ic_name, *reg])
        still_there = [
            ic_name for ic_name in pending
            if run_aws(["sagemaker", "describe-inference-component",
                        "--inference-component-name", ic_name, *reg]).returncode == 0
        ]
        pending = still_there
        if pending:
            time.sleep(15)

    if pending:
        log(f"WARNING: components still present after {timeout_seconds}s: {' '.join(pending)}")
    else:
        log("All inference components deleted")


def main() -> int:
    # [FUNCTION DOC] Tear down the named endpoint and related resources; the script entry point.
    # Behavior: 1) require an endpoint name (return 64 otherwise); resolve the region, return 1 if
    #   none; 2) describe the endpoint; if found note its config name, else log and continue looking
    #   for orphans; 3) collect model names from the config's production variants; 4) list inference
    #   components and add their models; 5) delete alarms prefixed "<endpoint>-" and
    #   "<component>-"; 6) delete scaling stacks for the AllTraffic variant and each component;
    #   7) delete components; 8) delete the endpoint if it existed; 9) delete the config if it still
    #   describes; 10) delete each model that still describes; 11) log completion.
    # Inputs: none (reads the process command-line arguments).
    # Outputs: returns 0, 1 or 64; logs on stderr.
    # Exceptions: none caught.
    # Side effects: many AWS API calls deleting resources; stderr logging.
    # KNOWN ISSUE: delete-endpoint is asynchronous and not awaited, so the immediate config/model
    #   deletes can race the endpoint deletion; deletion results are never checked yet logged as
    #   done, and the exit code is 0 even when deletes fail.
    if len(sys.argv) < 2:
        log(f"Usage: {os.path.basename(sys.argv[0])} <endpoint-name> [<region>]")
        return 64

    endpoint_name = sys.argv[1]
    region = resolve_region(sys.argv[2] if len(sys.argv) > 2 else None)
    if not region:
        log("ERROR: no AWS region. Pass region as 2nd arg or set AWS_REGION.")
        return 1

    reg = ["--region", region]
    log(f"Tearing down endpoint: {endpoint_name} in {region}")

    # Discover what's attached to this endpoint.
    config_name = ""
    model_names: list[str] = []
    desc = run_aws(["sagemaker", "describe-endpoint", "--endpoint-name", endpoint_name, *reg])
    endpoint_exists = desc.returncode == 0
    if endpoint_exists:
        config_name = json_out(desc, {}).get("EndpointConfigName", "")
    else:
        log("Endpoint not found — checking for orphan resources anyway")

    if config_name:
        cfg = json_out(
            run_aws(["sagemaker", "describe-endpoint-config",
                     "--endpoint-config-name", config_name, *reg]),
            {},
        )
        # Model-based endpoints name the model on the variant. IC-based ones do not.
        for variant in cfg.get("ProductionVariants", []):
            if variant.get("ModelName"):
                model_names.append(variant["ModelName"])

    ic_names = list_inference_components(endpoint_name, reg)
    if ic_names:
        log(f"Inference components on this endpoint: {' '.join(ic_names)}")
        for ic_name in ic_names:
            model = component_model_name(ic_name, reg)
            if model and model not in model_names:
                model_names.append(model)

    # Alarms — discovered by name prefix. deploy.py / deploy_async.py name them
    # "<endpoint>-*"; deploy_ic.py names the wake alarm "<component>-*".
    delete_alarms(f"{endpoint_name}-", reg)
    for ic_name in ic_names:
        delete_alarms(f"{ic_name}-", reg)

    # Autoscaling — variant target (model-based) and component targets (IC-based).
    delete_scaling_stack(f"endpoint/{endpoint_name}/variant/AllTraffic", VARIANT_DIM, reg)
    for ic_name in ic_names:
        delete_scaling_stack(f"inference-component/{ic_name}", IC_DIM, reg)

    # Inference components must go before the endpoint.
    if ic_names:
        delete_inference_components(ic_names, reg)

    # Endpoint (stops billing)
    if endpoint_exists:
        run_aws(["sagemaker", "delete-endpoint", "--endpoint-name", endpoint_name, *reg])
        log(f"Deleted endpoint: {endpoint_name} (billing stopped)")

    # Endpoint config
    if config_name and run_aws(
        ["sagemaker", "describe-endpoint-config", "--endpoint-config-name", config_name, *reg]
    ).returncode == 0:
        run_aws(["sagemaker", "delete-endpoint-config", "--endpoint-config-name", config_name, *reg])
        log(f"Deleted endpoint config: {config_name}")

    # Models
    for model_name in model_names:
        if run_aws(["sagemaker", "describe-model", "--model-name", model_name, *reg]).returncode == 0:
            run_aws(["sagemaker", "delete-model", "--model-name", model_name, *reg])
            log(f"Deleted model: {model_name}")

    log("Teardown complete. Data capture S3 objects (if any) NOT deleted — manage separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
