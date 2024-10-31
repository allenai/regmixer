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
def prepare_from(source: str):
    if source == "ai2/regmixer":
        raise ValueError("source workspace cannot be ai2/regmixer")

    if not source.startswith("ai2/"):
        raise ValueError("source workspace must be in the ai2 organization")

    user = beaker.account.whoami().name.upper()
    source_workspace = beaker.workspace.get(source)
    target_workspace = beaker.workspace.get("ai2/regmixer")

    required = (
        f"{user}_BEAKER_TOKEN",
        f"{user}_WANDB_API_KEY",
        f"{user}_AWS_CONFIG",
        f"{user}_AWS_CREDENTIALS",
        "R2_ENDPOINT_URL",
        "WEKA_ENDPOINT_URL",
    )

    for secret_name in required:
        secret_value = beaker.secret.read(secret_name, workspace=source_workspace)
        beaker.secret.write(secret_name, secret_value, workspace=target_workspace)

        print(f"copied '{secret_name}' to {target_workspace.full_name}")


@cli.command()
@click.option(
    "--user",
    "-u",
    type=str,
    help="Github user",
    required=True,
)
@click.option(
    "--token",
    "-t",
    type=str,
    help="Github token",
    required=True,
)
def set_gh(user: str, token: str):
    beaker_user = beaker.account.whoami().name.upper()
    target_workspace = beaker.workspace.get("ai2/regmixer")

    beaker.secret.write(f"{beaker_user}_GH_TOKEN", token, workspace=target_workspace)
    beaker.secret.write(f"{beaker_user}_GH_USER", user, workspace=target_workspace)

    print(f"copied github user/token to {beaker.workspace.get('ai2/regmixer').full_name}")


if __name__ == "__main__":
    cli()
