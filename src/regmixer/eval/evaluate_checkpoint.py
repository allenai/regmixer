import click
import json
import os
import subprocess
import sys
from typing import Optional
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from regmixer.eval.constants import (
    AUS_CLUSTERS,
    CLUSTERS,
    GOOG_CLUSTERS,
    ALL_FORMAT_OPTIONS,
    ALL_TASK_GROUPS_OPTIONS,
    ALL_MMLU_TASKS,
    ALL_CORE_TASKS,
    ALL_TULUISH_TASKS,
    ALL_GEN_TASKS,
)


def merge_json_dicts(json_dicts):
    merged_dict = {}
    for d in json_dicts:
        merged_dict.update(d)
    return merged_dict


def verify_json(json_string):
    try:
        json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input. {str(e)}", file=sys.stderr)
        sys.exit(1)


def get_checkpoint_name(path):
    split_path = path.split("checkpoints")[-1]
    modified_path = split_path.replace("/", "_").strip("_")
    return modified_path


def check_s3_path(path):
    s3 = boto3.client("s3")

    try:
        s3.head_object(Bucket=path.split("/")[2], Key="/".join(path.split("/")[3:]))
        return True
    except (NoCredentialsError, ClientError):
        return False


@click.command()
@click.option("-C", "--checkpoint", required=True, type=str, help="Path to checkpoint")
@click.option("-a", "--add-bos", is_flag=True, help="Add BOS token")
@click.option("-c", "--cluster", type=click.Choice(["aus", "goog", "*"]), help="Set cluster")
@click.option("-d", "--dashboard", help="Set dashboard name")
@click.option("-b", "--budget", help="Set budget")
@click.option("-w", "--workspace", help="Set workspace")
@click.option("-f", "--format", help="Set format (mc, rc, or both separated by comma)")
@click.option(
    "-g", "--eval-groups", help="Set task groups (mmlu, core, or both separated by comma)"
)
@click.option("-t", "--eval-tasks", help="Set specific tasks (comma-separated)")
@click.option(
    "-p", "--partition-size", type=int, help="Set partition size (must be a positive integer)"
)
@click.option("-u", "--wandb-url", help="Url of run in Weights & Biases to push results to")
@click.option(
    "-y", "--priority", help="Set priority (low, normal, high, or urgent)", default="normal"
)
@click.option("-n", "--num-gpus", help="Set number of GPUs", type=int)
@click.option("-x", "--extra-eval-args", help="Extra arguments to pass to oe-eval")
@click.option("-r", "--dry-run", is_flag=True, help="Dry run (do not launch jobs)")
@click.option("-s", "--beaker-secret", help="Beaker secret to use for Hugging Face access")
@click.option("-l", "--extra-gantry-args", help="Extra arguments to pass to Gantry")
@click.option("-z", "--inference-batch-size", help="Set batch size for inference", type=int)
@click.option("-v", "--use-vllm", is_flag=True, help="Use VLLM model type")
@click.option("-i", "--beaker-image", help="Use a specific Beaker image")
def main(
    checkpoint: str,
    add_bos: bool,
    cluster: str,
    dashboard: Optional[str],
    budget: str,
    workspace: str,
    priority: str,
    formats: Optional[str],
    eval_groups: Optional[str],
    eval_tasks: Optional[str],
    partition_size: Optional[int],
    wandb_url: Optional[str],
    num_gpus: Optional[int],
    extra_eval_args: Optional[str],
    dry_run: bool,
    beaker_secret: Optional[str],
    extra_gantry_args: Optional[str],
    inference_batch_size: Optional[int],
    use_vllm: bool,
    beaker_image: Optional[str],
):
    suffix = "-bos" if add_bos else ""
    model_type = "vllm" if use_vllm else "hf"
    gantry_args = "{}"
    use_gantry_flag = ""
    hf_secret = ""

    if extra_gantry_args:
        verify_json(extra_gantry_args)
        gantry_args = json.dumps(
            merge_json_dicts([json.loads(gantry_args), json.loads(extra_gantry_args)])
        )
        use_gantry_flag = "--use-gantry"

    if beaker_secret:
        hf_secret = beaker_secret
        use_hf_token = {"hf_token": True}
        gantry_args = json.dumps(merge_json_dicts([json.loads(gantry_args), use_hf_token]))

    clusters = CLUSTERS

    if cluster:
        if cluster == "aus":
            clusters = AUS_CLUSTERS
        elif cluster == "goog":
            clusters = GOOG_CLUSTERS
        else:
            clusters = cluster

    format_array = formats.split(",") if formats else ["rc"]
    for fmt in format_array:
        if fmt not in ALL_FORMAT_OPTIONS:
            print(f"Error: Invalid format '{fmt}'. Valid options are: {ALL_FORMAT_OPTIONS}")
            sys.exit(1)

    task_groups = eval_groups.split(",") if eval_groups else ["mmlu", "core"]
    for task_group in task_groups:
        if task_group not in ALL_TASK_GROUPS_OPTIONS:
            print(
                f"Error: Invalid task group '{task_group}'. Valid options are: {ALL_TASK_GROUPS_OPTIONS}"
            )
            sys.exit(1)

    task_list = eval_tasks.split(",") if eval_tasks else []

    if not task_list:
        if "mmlu" in task_groups:
            for fmt in format_array:
                for task in ALL_MMLU_TASKS:
                    task_list.append(f"{task}:{fmt}::olmes")
        if "core" in task_groups:
            for fmt in format_array:
                for task in ALL_CORE_TASKS:
                    task_list.append(f"{task}:{fmt}::olmes")
        if "tuluish" in task_groups:
            task_list.extend(ALL_TULUISH_TASKS)
        if "gen" in task_groups:
            task_list.extend(ALL_GEN_TASKS)

    budget = budget if budget else "ai2/oe-data"
    workspace = workspace if workspace else budget
    checkpoint_name = get_checkpoint_name(checkpoint)
    output_prefix = f"s3://ai2-llm/evaluation/{dashboard}/{checkpoint_name}"

    if (
        subprocess.run(
            ["command", "-v", "oe-eval"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).returncode
        != 0
    ):
        print(
            "Error: `oe-eval` is not installed; run `pip install git+https://github.com/allenai/oe-eval-internal.git`"
        )
        sys.exit(1)

    num_tasks = len(task_list)
    partition_size = partition_size if partition_size else 1
    num_jobs = (num_tasks + partition_size - 1) // partition_size

    print("----------------------------------------")
    print(f"Dashboard: {dashboard}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output prefix: {output_prefix}")
    print(f"Tasks: {task_list}")
    print(f"Clusters: {clusters}")
    print(f"Partition size: {partition_size}")
    print(f"Number of tasks: {num_tasks}")
    print(f"Number of jobs: {num_jobs}")
    print(f"Priority: {priority}")
    print(f"Model type: {model_type}")

    if beaker_image:
        print(f"Beaker image: {beaker_image}")
    if wandb_url:
        print(f"Weights & Biases URL: {wandb_url}")
    if num_gpus:
        print(f"GPUs: {num_gpus}")
    if hf_secret:
        print(f"HF secret: {hf_secret}")
    if gantry_args:
        print(f"Gantry args: {gantry_args}")
    if inference_batch_size:
        print(f"Batch size: {inference_batch_size}")
    print("----------------------------------------")

    partitioned_tasks = [
        task_list[i : i + partition_size] for i in range(0, num_tasks, partition_size)
    ]

    for partition in partitioned_tasks:
        task_args = [
            item for sublist in [["--task", task] for task in partition] for item in sublist
        ]
        partition_name = "_".join(partition).replace(":", "_").replace(" ", "_").replace("__", "_")
        output_path = f"{output_prefix}{suffix}/{partition_name}"

        if check_s3_path(output_path):
            if not dry_run:
                print(f"Test already exists: {output_path}")
                continue

        command = [
            "oe-eval",
            "--model",
            f"{checkpoint_name}{suffix}",
            "--model-args",
            f"model_path={checkpoint},add_bos_token={add_bos}",
            "--task",
            f"{partition}",
            "--remote-output-dir",
            f"{output_path}",
            "--beaker-workspace",
            f"{workspace}",
            "--beaker-budget",
            f"{budget}",
            "--beaker-priority",
            f"{priority}",
            "--cluster",
            f"{clusters}",
            "--datalake-tags",
            f"dashboard={dashboard},checkpoint={checkpoint_name}",
            "--gantry-secret-aws-access-key-id",
            "AWS_ACCESS_KEY_ID",
            "--gantry-secret-aws-secret-access",
            "AWS_SECRET_ACCESS_KEY",
            "--model-type",
            model_type,
            *task_args,
        ]

        if beaker_image:
            command.extend(["--beaker-image", beaker_image])
        if inference_batch_size:
            command.extend(["--batch-size", str(inference_batch_size)])
        if wandb_url:
            command.extend(["--wandb-run", wandb_url])
        if num_gpus:
            command.extend(["--gpus", str(num_gpus)])
        if hf_secret:
            command.extend(["--gantry-secret-hf-read-only", hf_secret])
        if gantry_args != "{}":
            command.extend(["--gantry-args", gantry_args])
        if use_gantry_flag:
            command.append(use_gantry_flag)
        if extra_eval_args:
            command.extend(extra_eval_args.split())
        if dry_run:
            command.append("--dry-run")

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"Error: Command '{e.cmd}' returned non-zero exit status {e.returncode}.",
                file=sys.stderr,
            )
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()
