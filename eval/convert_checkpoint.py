#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import textwrap
from typing import List, Optional
from pathlib import Path

class CheckpointError(Exception):
    """Custom exception for checkpoint-related errors."""
    pass

def check_s3_path(path: str) -> bool:
    """Check if an S3 path exists using the AWS CLI."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            error_details = f"AWS CLI Output:\n{result.stderr}"
            raise CheckpointError(f"Failed to check S3 path: {path}", error_details)
        return True
    except subprocess.CalledProcessError as e:
        error_details = f"Command failed with exit code {e.returncode}\n{e.stderr}"
        raise CheckpointError(f"AWS CLI command failed while checking path: {path}", error_details)
    except FileNotFoundError:
        raise CheckpointError(
            "AWS CLI not found",
            "Please ensure the AWS CLI is installed:\n  pip install awscli"
        )

def validate_checkpoint_path(checkpoint_path: str):
    """Validate the checkpoint path format and existence."""
    if not checkpoint_path:
        raise CheckpointError(
            "No checkpoint path provided",
            "Usage: script.py <checkpoint_path> [options]"
        )

    if not (checkpoint_path.startswith("s3://") or checkpoint_path.startswith("/")):
        raise CheckpointError(
            "Invalid checkpoint path format",
            "The checkpoint path must either:\n"
            "  - Start with 's3://' for S3 paths\n"
            "  - Start with '/' for absolute paths\n"
            f"Received: {checkpoint_path}"
        )

def build_commands(
    checkpoint_path: str,
    olmo_branch: str = "main",
    custom_tokenizer: Optional[str] = None,
    suffix: str = "hf"
) -> List[str]:
    """Build the list of commands to be executed."""
    commands = [
        "pip install awscli",
        "git clone https://github.com/allenai/OLMo.git",
        "cd OLMo"
    ]

    try:
        # Validate git branch name
        if not all(c.isalnum() or c in "-_/." for c in olmo_branch):
            raise CheckpointError(
                "Invalid git branch name",
                f"Branch name contains invalid characters: {olmo_branch}\n"
                "Only alphanumeric characters, hyphens, underscores, dots, and forward slashes are allowed."
            )
        commands.append(f"git checkout {olmo_branch}")
    except Exception as e:
        raise CheckpointError("Failed to process git branch", str(e))

    commands.append("pip install -e '.[all]'")
    
    conversion_cmd = (
        f"if [ ! -d '{checkpoint_path}-{suffix}' ]; then "
        f"python hf_olmo/convert_olmo_to_hf.py "
        f"--checkpoint-dir '{checkpoint_path}' "
        f"--destination-dir '{checkpoint_path}-{suffix}' "
        f"--keep-olmo-artifacts"
    )
    
    if custom_tokenizer:
        conversion_cmd += f" --tokenizer {custom_tokenizer}"
    
    conversion_cmd += "; else echo 'Destination directory already exists. Skipping conversion.'; fi"
    commands.append(conversion_cmd)
    
    return commands

def convert(
    checkpoint_path: str,
    suffix: str = "hf",
    workspace: str = "ai2/oe-data",
    budget: Optional[str] = None,
    priority: str = "normal",
    olmo_branch: str = "main",
    tokenizer: Optional[str] = None,
    gpus: int = 0,
    capture_output: bool = False
) -> subprocess.CompletedProcess:
    """
    Convert a checkpoint using specified parameters.
    
    Args:
        checkpoint_path: Path to the checkpoint (s3:// or absolute path)
        suffix: Suffix for output (default: "hf")
        workspace: Workspace (default: "ai2/oe-data")
        budget: Budget (defaults to workspace value)
        priority: Priority level (default: "normal")
        olmo_branch: OLMo branch (default: "main")
        tokenizer: Path to custom tokenizer
        gpus: Number of GPUs to use (default: 0)
        capture_output: Whether to capture command output (default: False)
    
    Returns:
        subprocess.CompletedProcess: Result of the gantry command execution
    
    Raises:
        CheckpointError: If any validation or execution error occurs
    """

    if not budget:
        budget = workspace
        print(f"Using default budget: {budget}")
    
    # Validate checkpoint path
    validate_checkpoint_path(checkpoint_path)
    
    # Initialize cluster and weka configurations
    clusters = ["--cluster","ai2/*"]
    weka_mountpoints = ""
    use_nfs = ""
    
    # Check for specific directories
    specific_dirs = [
        "climate-default", "mosaic-default", "nora-default",
        "oe-adapt-default", "oe-data-default", "oe-eval-default",
        "oe-training-default", "prior-default", "reviz-default",
        "skylight-default"
    ]
    
    for dir_name in specific_dirs:
        if checkpoint_path.startswith(f"/{dir_name}"):
            weka_mountpoints = ["--weka",f"{dir_name}:/{dir_name}"]
            clusters = [
                "--cluster","ai2/jupiter-cirrascale-2",
                "--cluster","ai2/saturn-cirrascale",
                "--cluster","ai2/neptune-cirrascale"
            ]
            use_nfs = "--no-nfs"
            break
    
    # Check S3 path existence
    if checkpoint_path.startswith("s3://"):
        if check_s3_path(checkpoint_path):
            print(f"Verified S3 path exists: {checkpoint_path}")
            use_nfs = "--no-nfs"
    else:
        print(f"Skipping existence check for non-S3 path: {checkpoint_path}")
    
    # Build and join commands
    commands = build_commands(
        checkpoint_path,
        olmo_branch,
        tokenizer,
        suffix
    )
    joined_commands = " && ".join(commands)
    
    # Build gantry command
    gantry_cmd = [
        "gantry", "run",
        "--description", f"Converting {checkpoint_path}",
        "--allow-dirty",
        "--no-python",
        "--workspace", workspace,
        "--priority", priority,
        "--gpus", str(gpus),
        "--preemptible"
    ]
    
    gantry_cmd.extend(clusters)
    gantry_cmd.extend([
        "--budget", budget,
        "--env-secret", "AWS_ACCESS_KEY_ID=S2_AWS_ACCESS_KEY_ID",
        "--env-secret", "AWS_SECRET_ACCESS_KEY=S2_AWS_SECRET_ACCESS_KEY",
        "--shared-memory", "10GiB"
    ])
    
    if weka_mountpoints:
        gantry_cmd.extend(weka_mountpoints)
    if use_nfs:
        gantry_cmd.append(use_nfs)
    
    gantry_cmd.extend(["--yes", "--", "/bin/bash", "-c", joined_commands])
    
    print("Executing gantry command...")
    
    # Execute gantry command
    try:
        result = subprocess.run(
            gantry_cmd,
            capture_output=capture_output,
            text=True,
            check=True
        )
        print(f"{result.stdout}")
        return result
        
    except subprocess.CalledProcessError as e:
        raise CheckpointError(
            "Gantry execution failed",
            f"Command failed with exit code {e.returncode}\n\n"
            f"stdout:\n{e.stdout}\n\n"
            f"stderr:\n{e.stderr}"
        )
    except FileNotFoundError:
        raise CheckpointError(
            "Gantry command not found",
            "Please ensure gantry is installed and available in your PATH"
        )

def main():
    parser = argparse.ArgumentParser(
        description="Convert checkpoints with various options",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint_path", help="Path to the checkpoint (s3:// or absolute path)")
    parser.add_argument("-s", "--suffix", default="hf", help="Suffix for output (default: hf)")
    parser.add_argument("-w", "--workspace", default="ai2/oe-data", help="Workspace (default: ai2/oe-data)")
    parser.add_argument("-b", "--budget", help="Budget (defaults to workspace value)")
    parser.add_argument("-p", "--priority", default="normal", 
                       choices=["low", "normal", "high", "urgent"],
                       help="Priority level (default: normal)")
    parser.add_argument("-c", "--olmo-branch", default="main", help="OLMo branch (default: main)")
    parser.add_argument("-t", "--tokenizer", help="Path to custom tokenizer")
    parser.add_argument("-g", "--gpus", type=int, default=0,
                       help="Number of GPUs to use (default: 0)")

    try:
        args = parser.parse_args()
        
        convert(
            checkpoint_path=args.checkpoint_path,
            suffix=args.suffix,
            workspace=args.workspace,
            budget=args.budget,
            priority=args.priority,
            olmo_branch=args.olmo_branch,
            tokenizer=args.tokenizer,
            gpus=args.gpus,
            capture_output=True
        )
            
    except CheckpointError as e:
        print(str(e), getattr(e, 'args')[1] if len(e.args) > 1 else None)
        sys.exit(1)
    except Exception as e:
        print(
            "An unexpected error occurred",
            f"Type: {type(e).__name__}\n"
            f"Details: {str(e)}"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()