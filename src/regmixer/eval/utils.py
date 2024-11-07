import click
import wandb
import pandas as pd

DEFAULT_WORKSPACE = "ai2-llm/regmixer"
METRICS = {"training_loss": "train/CE loss"}


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
                    (
                        run_id,
                        run.history(
                            samples=1000, pandas=(True), keys=[value for value in METRICS.values()]
                        ),
                    )
                )
        except KeyError:
            raise KeyError("'{group}' experiment group not found!")

    averages = [(run, _get_averages_for_run(history)) for run, history in filtered]
    print(averages)


def _get_averages_for_run(history) -> list[float]:
    df = pd.DataFrame(history)
    results = []
    for name, key in METRICS.items():
        results.append((name, df.loc[:, key].tail(10).mean()))

    return results
