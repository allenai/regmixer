import click
import wandb

DEFAULT_WORKSPACE = "ai2-llm/regmixer"
DEFAULT_KEYS = ["train/CE loss"]


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--group",
    "-g",
    type=str,
    help="The group ID to export metrics from",
    required=True,
)
def export_group(group: str):
    api = wandb.Api()
    all_runs = api.runs(path="ai2-llm/regmixer")
    filtered = []
    for run in all_runs:
        try:
            group_id = run.config["trainer"]["callbacks"]["wandb"]["group"]
            run_id = run.id
            if group_id == group:
                filtered.append(
                    (run_id, run.history(samples=100, pandas=(True), keys=["train/CE loss"]))
                )
        except KeyError:
            raise KeyError("'{group}' experiment group not found!")

    print(filtered)
