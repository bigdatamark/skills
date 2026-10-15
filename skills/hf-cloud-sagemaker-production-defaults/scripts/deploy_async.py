#!/usr/bin/env python
# [MODULE DOC] CLI script that deploys a SageMaker asynchronous-inference endpoint with production
# defaults using boto3, including scale-to-zero and a wake-from-zero step-scaling path.
# Behavior: parses flags, creates (or reuses) the model and an async endpoint config (S3 output,
#   optional failure path and SNS notifications, per-instance concurrency), creates the endpoint
#   and waits for InService, registers backlog-based target tracking plus a step-scaling
#   wake-from-zero policy, looks up that policy's ARN, creates backlog, failure and
#   HasBacklogWithoutCapacity alarms (the last one triggering the step policy), logs a summary
#   and prints a JSON summary.
# CLI arguments: required --model-name, --image-uri, --role-arn, --instance-type, --region,
#   --output-s3-uri; optional --model-s3-uri, repeatable --env KEY=VALUE,
#   --inference-ami-version, --failure-s3-uri, --success-sns-topic, --error-sns-topic,
#   --endpoint-name, --project, --environment (default "dev"), --initial-instance-count,
#   --min-capacity (default 0), --max-capacity, --backlog-per-instance-target,
#   --max-concurrent-invocations-per-instance, --no-autoscaling, --sns-alarm-topic, --no-alarms.
# Env vars: none read directly; boto3 uses the standard AWS credential chain.
# Constants: DEFAULTS - default capacities, backlog target, cooldowns, wake step size,
#   concurrency, alarm thresholds/periods and environment tag.
# Top-level executed code: imports helpers from the sibling _common module; main-guard runs main.
# Outputs: progress on stderr; one JSON line on stdout (adds inference mode and output URI);
#   exit 0 on success, otherwise a system exit message or uncaught exception.
"""Create a SageMaker async-inference endpoint with production defaults.

Minimal usage:
    python deploy_async.py --model-name <name> --image-uri <uri> \
        --role-arn <arn> --instance-type ml.g5.xlarge --region <region> \
        --output-s3-uri s3://bucket/path/

Async differs from real-time in three structural ways:
  1. AsyncInferenceConfig on the endpoint config (S3 output path, optional
     SNS notification topics, optional failure path)
  2. Autoscaling MinCapacity=0 is supported — endpoints can genuinely scale
     to zero between batches. Target-tracking on ApproximateBacklogSizePerInstance.
  3. Scale-from-zero (0→1) requires a SEPARATE step-scaling policy bound to
     a HasBacklogWithoutCapacity alarm. Target-tracking alone can't transition
     from zero, so without the step policy the endpoint won't wake up.

See SKILL.md for the recommended chained-with-resolver pattern.
"""

import argparse
import json
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError

from _common import (
    build_tags,
    create_model,
    log as _log,
    make_endpoint_name,
    parse_env,
    wait_for_endpoint,
)


# Defaults — change here, not at call sites.
DEFAULTS = {
    "initial_instance_count": 1,
    "min_capacity": 0,                          # async genuinely supports scale-to-zero
    "max_capacity": 4,
    "backlog_per_instance_target": 5,           # target queue depth per instance
    "scale_in_cooldown_seconds": 600,           # slower scale-in for async — batches are bursty
    "scale_out_cooldown_seconds": 60,
    "wake_from_zero_step_size": 1,              # 0→1 instance when backlog appears
    "max_concurrent_invocations_per_instance": 4,
    # Alarms
    "alarm_backlog_size_threshold": 50,         # queue too deep
    "alarm_failed_invocations_threshold": 5,    # repeated failures
    "alarm_evaluation_periods": 1,
    "alarm_period_seconds": 60,                 # async needs faster eval for wake-from-zero
    "environment_tag": "dev",
}


def log(msg: str) -> None:
    # [FUNCTION DOC] HELPER: Log a message to stderr with the "deploy_async" prefix.
    # Behavior: delegates to the shared logger.
    # Inputs: msg - text.
    # Outputs: none returned; one stderr line.
    # Exceptions: none.
    # Side effects: writes to stderr.
    _log("deploy_async", msg)


def create_async_endpoint_config(
    sm: Any, *, config_name: str, model_name: str, instance_type: str,
    initial_instance_count: int, inference_ami_version: str | None,
    output_s3_uri: str, failure_s3_uri: str | None,
    success_topic_arn: str | None, error_topic_arn: str | None,
    max_concurrent_invocations: int,
    tags: list[dict],
) -> str:
    # [FUNCTION DOC] Create an async-inference endpoint config, reusing it if it already exists.
    # Behavior: builds an "AllTraffic" variant (model, instance type, initial count, weight 1.0,
    #   optional InferenceAmiVersion); builds an output config with the S3 output path, optional
    #   failure path and optional SNS success/error topics; wraps it with a client config holding
    #   max concurrent invocations per instance; calls create_endpoint_config with the variant,
    #   async config and tags. An "already existing" client error is logged as reuse; others raise.
    # Inputs (keyword-only after sm): sm - boto3 SageMaker client; config_name; model_name;
    #   instance_type; initial_instance_count; inference_ami_version (or none); output_s3_uri;
    #   failure_s3_uri, success_topic_arn, error_topic_arn (each optional); max_concurrent_invocations;
    #   tags.
    # Outputs: the config name.
    # Exceptions: swallows the already-exists client error; re-raises others.
    # Side effects: AWS API call; stderr logging.
    # KNOWN ISSUE: a reused config is not compared to the requested settings.
    log(f"Creating async endpoint config: {config_name}")

    production_variant: dict[str, Any] = {
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InstanceType": instance_type,
        "InitialInstanceCount": initial_instance_count,
        "InitialVariantWeight": 1.0,
    }
    if inference_ami_version:
        production_variant["InferenceAmiVersion"] = inference_ami_version
        log(f"  InferenceAmiVersion set to: {inference_ami_version}")

    output_config: dict[str, Any] = {"S3OutputPath": output_s3_uri}
    if failure_s3_uri:
        output_config["S3FailurePath"] = failure_s3_uri
    if success_topic_arn or error_topic_arn:
        notification: dict[str, str] = {}
        if success_topic_arn:
            notification["SuccessTopic"] = success_topic_arn
        if error_topic_arn:
            notification["ErrorTopic"] = error_topic_arn
        output_config["NotificationConfig"] = notification

    async_config: dict[str, Any] = {
        "OutputConfig": output_config,
        "ClientConfig": {
            "MaxConcurrentInvocationsPerInstance": max_concurrent_invocations,
        },
    }

    kwargs: dict[str, Any] = {
        "EndpointConfigName": config_name,
        "ProductionVariants": [production_variant],
        "AsyncInferenceConfig": async_config,
        "Tags": tags,
    }

    try:
        sm.create_endpoint_config(**kwargs)
    except ClientError as e:
        if "Cannot create already existing endpoint configuration" in str(e):
            log(f"Endpoint config {config_name} already exists — reusing")
        else:
            raise
    return config_name


def create_endpoint(sm: Any, *, endpoint_name: str, config_name: str, tags: list[dict]) -> None:
    # [FUNCTION DOC] Create the SageMaker endpoint from a config.
    # Behavior: logs and calls create_endpoint with name, config and tags; does not wait or reuse.
    # Inputs (keyword-only after sm): sm - boto3 SageMaker client; endpoint_name; config_name; tags.
    # Outputs: none.
    # Exceptions: client errors propagate.
    # Side effects: AWS API call creating a billable endpoint; stderr logging.
    log(f"Creating endpoint: {endpoint_name}")
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
        Tags=tags,
    )


def register_async_autoscaling(
    *, endpoint_name: str, variant_name: str, min_capacity: int, max_capacity: int,
    backlog_per_instance_target: int, scale_in_cooldown: int, scale_out_cooldown: int,
    wake_step_size: int, region: str,
) -> None:
    # [FUNCTION DOC] Register backlog target tracking and a wake-from-zero step policy on a variant.
    # Behavior: registers the variant's desired instance count as a scalable target with min/max;
    #   puts "<endpoint>-backlog-target-tracking" tracking the average of
    #   ApproximateBacklogSizePerInstance (endpoint dimension) at the target with the cooldowns;
    #   puts "<endpoint>-step-wake-from-zero", a step policy adding wake_step_size instances for any
    #   metric value at or above the alarm threshold, with the scale-out cooldown.
    # Inputs (keyword-only): endpoint_name; variant_name; min_capacity; max_capacity;
    #   backlog_per_instance_target; scale_in_cooldown; scale_out_cooldown; wake_step_size; region.
    # Outputs: none.
    # Exceptions: client errors propagate.
    # Side effects: Application Auto Scaling API calls; stderr logging.
    """Register two autoscaling policies on the variant:

    1. Target-tracking on ApproximateBacklogSizePerInstance — handles
       ongoing scaling between MinCapacity and MaxCapacity.
    2. Step-scaling that increments capacity by `wake_step_size` when
       triggered — needed for 0→1 wake-from-zero, because target-tracking
       can't transition from zero capacity.

    The step-scaling policy is invoked by a HasBacklogWithoutCapacity alarm
    (created separately in create_async_alarms).
    """
    log(f"Registering async autoscaling: min={min_capacity} max={max_capacity} backlog-target={backlog_per_instance_target}")
    appscaling = boto3.client("application-autoscaling", region_name=region)
    resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

    appscaling.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity,
    )

    # (1) Target-tracking for ongoing scaling between min and max
    appscaling.put_scaling_policy(
        PolicyName=f"{endpoint_name}-backlog-target-tracking",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": float(backlog_per_instance_target),
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": scale_in_cooldown,
            "ScaleOutCooldown": scale_out_cooldown,
        },
    )

    # (2) Step-scaling for wake-from-zero. The CloudWatch alarm bound to
    # this policy is created in create_async_alarms.
    appscaling.put_scaling_policy(
        PolicyName=f"{endpoint_name}-step-wake-from-zero",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            "MetricAggregationType": "Maximum",
            "Cooldown": scale_out_cooldown,
            "StepAdjustments": [
                {
                    "MetricIntervalLowerBound": 0,
                    "ScalingAdjustment": wake_step_size,
                },
            ],
        },
    )


def create_async_alarms(
    *, endpoint_name: str, variant_name: str, sns_topic_arn: str | None,
    region: str, wake_alarm_arns_for_step_policy: list[str],
) -> None:
    # [FUNCTION DOC] Create the async endpoint's CloudWatch alarms, including the wake-from-zero trigger.
    # Behavior: puts "<endpoint>-ApproximateBacklogSize" (average above the backlog threshold) and
    #   "<endpoint>-InvocationsFailed" (sum above the failure threshold), both notifying the SNS
    #   topic if given; then "<endpoint>-HasBacklogWithoutCapacity" (average at or above 1 over one
    #   60-second period) whose actions are the supplied step policy ARNs. Warns if no SNS topic.
    # Inputs (keyword-only): endpoint_name; variant_name (unused); sns_topic_arn (or none); region;
    #   wake_alarm_arns_for_step_policy - list of policy ARNs for the wake alarm (may be empty).
    # Outputs: none.
    # Exceptions: client errors propagate.
    # Side effects: CloudWatch API calls; stderr logging.
    # KNOWN ISSUE: variant_name is accepted but never used.
    """Create CloudWatch alarms for the async endpoint, including the
    HasBacklogWithoutCapacity alarm that drives the wake-from-zero policy.
    """
    log(f"Creating CloudWatch alarms for {endpoint_name}")
    cw = boto3.client("cloudwatch", region_name=region)
    actions = [sns_topic_arn] if sns_topic_arn else []
    endpoint_dim = [{"Name": "EndpointName", "Value": endpoint_name}]

    # 1. Backlog too deep — capacity is keeping up poorly with incoming requests
    cw.put_metric_alarm(
        AlarmName=f"{endpoint_name}-ApproximateBacklogSize",
        AlarmDescription="Async queue backlog too deep",
        MetricName="ApproximateBacklogSize",
        Namespace="AWS/SageMaker",
        Dimensions=endpoint_dim,
        Statistic="Average",
        Period=DEFAULTS["alarm_period_seconds"],
        EvaluationPeriods=DEFAULTS["alarm_evaluation_periods"],
        Threshold=DEFAULTS["alarm_backlog_size_threshold"],
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=actions,
    )

    # 2. InvocationsFailed — repeated client/server errors in async processing
    cw.put_metric_alarm(
        AlarmName=f"{endpoint_name}-InvocationsFailed",
        AlarmDescription="Async invocations failing repeatedly",
        MetricName="InvocationsFailed",
        Namespace="AWS/SageMaker",
        Dimensions=endpoint_dim,
        Statistic="Sum",
        Period=DEFAULTS["alarm_period_seconds"],
        EvaluationPeriods=DEFAULTS["alarm_evaluation_periods"],
        Threshold=DEFAULTS["alarm_failed_invocations_threshold"],
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=actions,
    )

    # 3. HasBacklogWithoutCapacity — drives the step-scaling wake-from-zero
    # policy. The Alarm action is the step policy ARN, not the SNS topic.
    cw.put_metric_alarm(
        AlarmName=f"{endpoint_name}-HasBacklogWithoutCapacity",
        AlarmDescription="Async backlog exists but no instances running — triggers wake-from-zero",
        MetricName="HasBacklogWithoutCapacity",
        Namespace="AWS/SageMaker",
        Dimensions=endpoint_dim,
        Statistic="Average",
        Period=60,                              # tight evaluation: we want fast wake-up
        EvaluationPeriods=1,
        Threshold=1,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=wake_alarm_arns_for_step_policy,   # step policy ARNs
    )

    if not sns_topic_arn:
        log("WARNING: no --sns-alarm-topic — backlog/failure alarms exist but won't notify anyone.")


def get_step_policy_arn(*, endpoint_name: str, variant_name: str, region: str) -> str:
    # [FUNCTION DOC] HELPER: Look up the ARN of the wake-from-zero step-scaling policy.
    # Behavior: describes scaling policies for the variant resource filtered to
    #   "<endpoint>-step-wake-from-zero" and returns the first policy's ARN.
    # Inputs (keyword-only): endpoint_name; variant_name; region.
    # Outputs: the policy ARN string.
    # Exceptions: raises a runtime error if no policy is found; client errors propagate.
    # Side effects: Application Auto Scaling API call.
    """Look up the ARN of the step-scaling policy we just created.

    `put_scaling_policy` returns the ARN on creation but we don't capture it
    above to keep the function signature small. Re-fetch via describe.
    """
    appscaling = boto3.client("application-autoscaling", region_name=region)
    resp = appscaling.describe_scaling_policies(
        ServiceNamespace="sagemaker",
        ResourceId=f"endpoint/{endpoint_name}/variant/{variant_name}",
        PolicyNames=[f"{endpoint_name}-step-wake-from-zero"],
    )
    policies = resp.get("ScalingPolicies", [])
    if not policies:
        raise RuntimeError(f"Step-scaling policy not found for {endpoint_name} — was register_async_autoscaling called?")
    return policies[0]["PolicyARN"]


def main() -> int:
    # [FUNCTION DOC] Parse arguments and run the full async deployment; the script entry point.
    # Behavior: 1) define and parse flags; 2) parse env pairs, compute endpoint and config names;
    #   3) create clients and read the caller ARN; 4) exit with a message if min capacity is 0 and
    #   autoscaling is disabled; 5) build tags with InferenceMode=async; 6) create model, async
    #   config and endpoint, then wait for InService; 7) unless autoscaling is disabled, register
    #   both policies and fetch the step policy ARN; 8) unless alarms are disabled, create alarms,
    #   passing the step ARN (if any) to the wake alarm; 9) log a summary with an example async
    #   invoke command and teardown command, and print the JSON summary.
    # Inputs: none (reads the process command-line arguments).
    # Outputs: returns 0; JSON on stdout; logs on stderr.
    # Exceptions: raises a system exit for the invalid capacity combination; others propagate.
    # Side effects: AWS API calls creating model, config, endpoint, scaling policies and alarms.
    # KNOWN ISSUE: with min capacity 0 and --no-alarms, the HasBacklogWithoutCapacity alarm is never
    #   created, so the endpoint can scale to zero and never wake up; this combination is not
    #   rejected. The capacity check also runs only after AWS clients are created and STS is called.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required
    p.add_argument("--model-name", required=True)
    p.add_argument("--image-uri", required=True, help="From hf-cloud-serving-image-selection")
    p.add_argument("--role-arn", required=True, help="From hf-cloud-sagemaker-iam-preflight")
    p.add_argument("--instance-type", required=True, help="e.g. ml.g5.xlarge")
    p.add_argument("--region", required=True, help="From hf-cloud-aws-context-discovery")
    p.add_argument("--output-s3-uri", required=True,
                   help="S3 path where async results are written. e.g. s3://amzn-s3-demo-async-output/async-output/")

    # Conditional
    p.add_argument("--model-s3-uri", default=None,
                   help="S3 model artifact extracted to /opt/ml/model (preferred); omit only for Hub-at-runtime")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE; repeatable")
    p.add_argument("--inference-ami-version", default=None,
                   help="REQUIRED for vLLM DLC with CUDA 13+ (e.g. al2-ami-sagemaker-inference-gpu-3-1)")
    p.add_argument("--failure-s3-uri", default=None,
                   help="S3 path for failed async invocations (optional; default: errors written to OutputConfig.S3OutputPath)")
    p.add_argument("--success-sns-topic", default=None,
                   help="SNS topic ARN notified when async invocation succeeds")
    p.add_argument("--error-sns-topic", default=None,
                   help="SNS topic ARN notified when async invocation fails")

    # Naming
    p.add_argument("--endpoint-name", default=None, help="Default: <model-name>-<timestamp>")
    p.add_argument("--project", default=None, help="Tag value (default: model name)")
    p.add_argument("--environment", default=DEFAULTS["environment_tag"])

    # Capacity / scaling
    p.add_argument("--initial-instance-count", type=int, default=DEFAULTS["initial_instance_count"])
    p.add_argument("--min-capacity", type=int, default=DEFAULTS["min_capacity"],
                   help="Async supports 0 (scale-to-zero between batches). Default 0.")
    p.add_argument("--max-capacity", type=int, default=DEFAULTS["max_capacity"])
    p.add_argument("--backlog-per-instance-target", type=int,
                   default=DEFAULTS["backlog_per_instance_target"],
                   help="Target queue depth per instance for autoscaling")
    p.add_argument("--max-concurrent-invocations-per-instance", type=int,
                   default=DEFAULTS["max_concurrent_invocations_per_instance"])
    p.add_argument("--no-autoscaling", action="store_true",
                   help="NOT RECOMMENDED for async — endpoint won't scale to zero")

    # Alarms
    p.add_argument("--sns-alarm-topic", default=None, help="SNS topic ARN for backlog/failure alarms")
    p.add_argument("--no-alarms", action="store_true")

    args = p.parse_args()

    env_dict = parse_env(args.env)
    endpoint_name = make_endpoint_name(args.model_name, args.endpoint_name)
    config_name = f"{endpoint_name}-config"

    sts = boto3.client("sts", region_name=args.region)
    sm = boto3.client("sagemaker", region_name=args.region)
    caller_arn = sts.get_caller_identity()["Arn"]

    if args.min_capacity == 0 and args.no_autoscaling:
        raise SystemExit(
            "--min-capacity 0 requires autoscaling to be enabled (the wake-from-zero "
            "policy is what brings instances up). Either remove --no-autoscaling or "
            "set --min-capacity 1."
        )

    tags = build_tags(
        project=args.project or args.model_name,
        caller_arn=caller_arn,
        environment=args.environment,
        model_s3_uri=args.model_s3_uri,
        extra={"InferenceMode": "async"},
    )

    create_model(
        sm, model_name=args.model_name, image_uri=args.image_uri,
        role_arn=args.role_arn, model_s3_uri=args.model_s3_uri,
        env=env_dict, tags=tags, log_prefix="deploy_async",
    )
    create_async_endpoint_config(
        sm, config_name=config_name, model_name=args.model_name,
        instance_type=args.instance_type,
        initial_instance_count=args.initial_instance_count,
        inference_ami_version=args.inference_ami_version,
        output_s3_uri=args.output_s3_uri,
        failure_s3_uri=args.failure_s3_uri,
        success_topic_arn=args.success_sns_topic,
        error_topic_arn=args.error_sns_topic,
        max_concurrent_invocations=args.max_concurrent_invocations_per_instance,
        tags=tags,
    )
    create_endpoint(sm, endpoint_name=endpoint_name, config_name=config_name, tags=tags)
    wait_for_endpoint(sm, endpoint_name, log_prefix="deploy_async")

    if not args.no_autoscaling:
        register_async_autoscaling(
            endpoint_name=endpoint_name, variant_name="AllTraffic",
            min_capacity=args.min_capacity, max_capacity=args.max_capacity,
            backlog_per_instance_target=args.backlog_per_instance_target,
            scale_in_cooldown=DEFAULTS["scale_in_cooldown_seconds"],
            scale_out_cooldown=DEFAULTS["scale_out_cooldown_seconds"],
            wake_step_size=DEFAULTS["wake_from_zero_step_size"],
            region=args.region,
        )
        # The wake-from-zero alarm needs the step policy's ARN as its action.
        step_policy_arn = get_step_policy_arn(
            endpoint_name=endpoint_name, variant_name="AllTraffic", region=args.region,
        )
    else:
        log("WARNING: autoscaling skipped. Endpoint will NOT scale (in either direction).")
        step_policy_arn = None

    if not args.no_alarms:
        create_async_alarms(
            endpoint_name=endpoint_name, variant_name="AllTraffic",
            sns_topic_arn=args.sns_alarm_topic, region=args.region,
            wake_alarm_arns_for_step_policy=[step_policy_arn] if step_policy_arn else [],
        )

    # Summary
    log("")
    log(f"Async deployment complete: {endpoint_name}")
    log(f"  Instance:           {args.instance_type}")
    if args.no_autoscaling:
        autoscaling_summary = "OFF"
    else:
        zero_label = "enabled" if args.min_capacity == 0 else "disabled"
        autoscaling_summary = f"{args.min_capacity}-{args.max_capacity} instances (scale-to-zero {zero_label})"
    log(f"  Autoscaling:        {autoscaling_summary}")
    log(f"  Output S3 path:     {args.output_s3_uri}")
    log(f"  Notifications:      success={args.success_sns_topic or 'none'} error={args.error_sns_topic or 'none'}")
    log("")
    log("Invoke (async, via S3 input location):")
    log(f"  aws sagemaker-runtime invoke-endpoint-async \\")
    log(f"    --endpoint-name {endpoint_name} \\")
    log(f"    --input-location s3://YOUR-INPUT-BUCKET/path/to/input.json \\")
    log(f"    --content-type application/json --region {args.region}")
    log(f"  # Result will land at: {args.output_s3_uri}")
    log("  # Write input.json as BOM-free UTF-8 (on Windows, NOT 'Set-Content -Encoding UTF8')")
    log("")
    log(f"Teardown: python3 teardown.py {endpoint_name} {args.region}")

    print(json.dumps({
        "endpoint_name": endpoint_name,
        "endpoint_config_name": config_name,
        "model_name": args.model_name,
        "region": args.region,
        "instance_type": args.instance_type,
        "inference_mode": "async",
        "output_s3_uri": args.output_s3_uri,
    }))

    return 0


if __name__ == "__main__":
    sys.exit(main())
