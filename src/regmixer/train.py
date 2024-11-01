from typing import List, Tuple, cast

import click
from olmo_core.distributed.utils import init_hybrid_shard_mesh
from olmo_core.train import (
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.data import TokenizerConfig
from olmo_core.train.callbacks import (
    CometCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.utils import get_default_device, seed_all
from torch.distributed.elastic.multiprocessing.errors import record

from regmixer.aliases import SourceInstance
from regmixer.model.transformer import TransformerConfigBuilder


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--max-tokens",
    "-t",
    type=int,
    help="Max tokens for the mixture dataset",
    required=True,
)
@click.option(
    "--source",
    "-s",
    multiple=True,
    type=(str, str, str),
    help="Source datasets in the form `name path,path,... ratio`",
)
@click.option(
    "--run-name",
    "-n",
    type=str,
    help="Name of the run",
    required=True,
)
@click.option(
    "--sequence-length",
    "-l",
    type=int,
    help="Sequence length for the transformer",
)
@click.option(
    "--seed",
    "-S",
    type=int,
    help="Seed for the experiment",
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    type=str,
    help="Overrides for the transformer config",
)
@record
def train(
    run_name: str,
    max_tokens: int,
    source: List[Tuple[str, str, str]],
    override: List[str],
    sequence_length: int,
    seed: int,
):
    sources: List[SourceInstance] = []
    for item in source:
        name, paths, ratio = item
        paths = paths.split(",")
        sources.append(SourceInstance(name=name, paths=paths, ratio=float(ratio)))

    tokenizer = TokenizerConfig.dolma2()

    config = TransformerConfigBuilder(
        run_name=run_name,
        max_tokens=max_tokens,
        sources=sources,
        overrides=override,
        sequence_length=sequence_length,
        seed=seed,
        tokenizer_config=tokenizer,
    ).build()

    seed_all(config.init_seed)
    model = config.model.build(
        init_device="meta",
        device=get_default_device(),
        dp_mesh=init_hybrid_shard_mesh(num_replicas=2),
    )
    optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset)
    trainer = config.trainer.build(model, optim, data_loader)
    config_dict = config.as_config_dict()
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


if __name__ == "__main__":
    try:
        prepare_training_environment()
        cli()
    finally:
        teardown_training_environment()
