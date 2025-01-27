"""

It's helpful to get a view of all the runs in a Wandb workspace, especially
know how they group together. This script helps you do this.

You can run the script like:

    python organize_runs_in_workspace.py --workspace "ai2-llm/regmixer" --output-dir "./regmixer_runs/" --batch-size 100 --finished-only

@kylel

"""

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import wandb
from tqdm import tqdm


def setup_logging(log_file: Path):
    """Set up logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def fetch_wandb_runs(
    workspace: str, batch_size: int = 50, finished_only: bool = False
) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    """
    Fetch all runs from a W&B workspace and organize them by group with streaming and progress updates.

    Args:
        workspace (str): The W&B workspace in format "entity/project"
        batch_size (int): Number of runs to process in each batch
        finished_only (bool): If True, only include finished runs

    Returns:
        Tuple[Dict[str, List[Any]], Dict[str, str]]:
            - Dictionary mapping group names to lists of runs
            - Dictionary mapping group names to their regex patterns
    """
    # Initialize the API
    api = wandb.Api()

    try:
        # Get total number of runs for progress tracking
        logging.info(f"Fetching run count from {workspace}...")
        filters = {"state": "finished"} if finished_only else {}
        runs = api.runs(path=workspace, filters=filters)
        total_runs = len(runs)
        logging.info(f"Found {total_runs} total runs")

        # Create a defaultdict to store runs by group
        runs_by_group = defaultdict(list)

        # Create progress bar
        pbar = tqdm(total=total_runs, desc="Processing runs")

        start_time = time.time()
        last_update = start_time
        processed_runs = 0

        # Process runs in batches
        current_batch = []
        for run in runs:
            current_batch.append(run)
            processed_runs += 1

            # Process batch when it reaches batch_size or on last run
            if len(current_batch) >= batch_size or processed_runs == total_runs:
                for batch_run in current_batch:
                    # Get group name, use 'ungrouped' if no group is specified
                    group = batch_run.group if batch_run.group else "ungrouped"

                    # Create a dictionary with relevant run information
                    run_info = {
                        "id": batch_run.id,
                        "author": batch_run.user.username,
                        "name": batch_run.name,
                        "state": batch_run.state,
                        "config": batch_run.config,
                        "summary": batch_run.summary._json_dict,
                        "created_at": batch_run.created_at,
                        "heartbeat_at": batch_run.heartbeat_at,
                        "tags": batch_run.tags,
                    }

                    # Add run to appropriate group
                    runs_by_group[group].append(run_info)

                # Update progress bar
                pbar.update(len(current_batch))

                # Log progress every 30 seconds
                current_time = time.time()
                if current_time - last_update >= 30:
                    elapsed_time = current_time - start_time
                    runs_per_second = processed_runs / elapsed_time
                    remaining_runs = total_runs - processed_runs
                    estimated_remaining_time = (
                        remaining_runs / runs_per_second if runs_per_second > 0 else 0
                    )

                    logging.info(
                        f"\nProgress update:\n"
                        f"- Processed {processed_runs}/{total_runs} runs ({processed_runs / total_runs * 100:.1f}%)\n"
                        f"- Processing speed: {runs_per_second:.1f} runs/second\n"
                        f"- Estimated time remaining: {estimated_remaining_time / 60:.1f} minutes\n"
                        f"- Found {len(runs_by_group)} distinct groups so far"
                    )

                    last_update = current_time

                # Clear batch
                current_batch = []

        pbar.close()
        logging.info(f"Processing completed in {(time.time() - start_time) / 60:.1f} minutes")

        return dict(runs_by_group)

    except Exception as e:
        logging.error(f"Error fetching runs: {str(e)}")
        return {}, {}


@click.command()
@click.option("--workspace", required=True, help='W&B workspace in format "entity/project"')
@click.option(
    "--output-dir", required=True, type=click.Path(), help="Directory to save output files"
)
@click.option("--batch-size", default=50, help="Number of runs to process in each batch")
@click.option("--finished-only", is_flag=True, help="Only include finished runs")
def main(workspace: str, output_dir: str, batch_size: int, finished_only: bool):
    """Fetch all runs from a W&B workspace and save them to files."""
    # Convert output_dir to Path and create if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir / "fetch_runs.log")

    logging.info(
        f"Starting run collection from {workspace} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logging.info(f"Fetching {'only finished' if finished_only else 'all'} runs")

    # Fetch and organize runs
    runs_by_group = fetch_wandb_runs(workspace, batch_size, finished_only)

    # Save the complete data as JSON
    data_path = output_dir / "runs_by_group.json"
    with open(data_path, "w") as f:
        json.dump(runs_by_group, f, indent=2)
    logging.info(f"Saved complete run data to {data_path}")


if __name__ == "__main__":
    main()
