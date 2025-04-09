import click
from beaker import Beaker

beaker = Beaker.from_env()


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--source",
    "-s",
    type=str,
    help="The source workspace to copy secrets from",
    required=True,
)
@click.option(
    "--dest",
    "-d",
    type=str,
    help="The destination workspace to copy secrets to",
    default="ai2/dolma2",
)
def prepare_from(source: str, dest: str):
    if source == dest:
        raise ValueError("Dest workspace cannot be source workspace")

    if not source.startswith("ai2/"):
        raise ValueError("source workspace must be in the ai2 organization")

    user = beaker.account.whoami().name.upper()
    source_workspace = beaker.workspace.get(source)
    target_workspace = beaker.workspace.get(dest)

    required = (
        f"{user}_BEAKER_TOKEN",
        f"{user}_WANDB_API_KEY",
        f"{user}_AWS_CONFIG",
        f"{user}_AWS_CREDENTIALS",
        f"{user}_COMET_API_KEY",
        "SLACK_WEBHOOK_URL"
        #"R2_ENDPOINT_URL",
        #"WEKA_ENDPOINT_URL",
    )

    for secret_name in required:
        secret_value = beaker.secret.read(secret_name, workspace=source_workspace)
        beaker.secret.write(secret_name, secret_value, workspace=target_workspace)

        print(f"copied '{secret_name}' to {target_workspace.full_name}")


if __name__ == "__main__":
    cli()
