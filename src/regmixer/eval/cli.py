import json
import logging
import pathlib
import warnings
from typing import Optional

import click
import numpy as np
import pandas as pd
import wandb
from olmo_core.utils import prepare_cli_environment

from regmixer.eval.constants import GroupedWandbMetrics
from regmixer.synthesize_mixture import calculate_priors
from regmixer.utils import config_from_path
from regmixer.eval.utils import (
    build_regression,
    get_output_dir,
    get_runs_from_api,
    mk_run_from_json,
    mk_run_metrics,
    mk_weights_from_config,
    mk_output_prefix,
    plot_correlation,
    plot_distributions,
    simulate,
)

logger = logging.getLogger(__name__)

BASE_CACHE_DIR = "cache/"
DEFAULT_WORKSPACE = "ai2-llm/regmixer"


@click.group()
def cli():
    prepare_cli_environment()


@cli.command()
@click.option(
    "--experiment-groups",
    "-g",
    type=str,
    multiple=True,
    help="The group ID(s) to fit the regression model against",
    required=True,
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Relative path to the experiment configuration file.",
    required=True,
)
@click.option(
    "-a",
    "--alpha",
    type=float,
    default=1.0,
    help="Alpha to apply to simulated distributions",
    required=False,
)
@click.option(
    "-A",
    "--group-average",
    type=str,
    help="The metric group to average for the regression task (regression will be fit against the average)",
    required=False,
    default=None,
)
@click.option(
    "-G",
    "--group-metrics",
    type=str,
    help="The metric group to fit regressions against (each metric will be fit separately)",
    default=None,
)
@click.option(
    "-s",
    "--num-samples",
    type=int,
    default=10,
    help="The number of evaluation samples per metric to collect from the run history",
    required=False,
)
@click.option(
    "-w",
    "--workspace",
    type=str,
    default=DEFAULT_WORKSPACE,
    help="The Wandb workspace to query for the runs",
    required=False,
)
@click.option(
    "-N",
    "--no-cache",
    is_flag=True,
    help="Do not use the cache for the runs",
    required=False,
    default=False,
)
@click.option(
    "-e",
    "--use-entropy",
    is_flag=True,
    help="Select highest entropy samples for simulation.",
    required=False,
    default=False,
)
@click.option(
    "-S",
    "--simulation-samples",
    type=int,
    default=100_000,
    help="Number of simulation samples to generate for each metric",
    required=False,
)
def fit(
    experiment_groups: list[str],
    config: pathlib.Path,
    alpha: float,
    num_samples: int,
    simulation_samples: int,
    group_average: Optional[str],
    group_metrics: Optional[str],
    workspace: str,
    no_cache: bool,
    use_entropy: bool,
):
    output_dir = get_output_dir(experiment_groups)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(BASE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if group_average and group_metrics:
        raise ValueError("Cannot provide both group-average and group-metrics")

    api = wandb.Api()
    cache_path = pathlib.Path(BASE_CACHE_DIR) / f"{'_'.join(experiment_groups)}_runs_cache.json"

    if no_cache:
        logger.info(f"Cache disabled, will not use cache for run samples...")
        run_instances = get_runs_from_api(
            api, workspace, experiment_groups, cache_path, no_cache, num_samples
        )
    else:
        try:
            # TODO: Add partitioned cache per group maybe?
            with open(cache_path, "r") as f:
                run_dict = json.load(f)
                run_instances = [mk_run_from_json(run) for run in run_dict]

            logger.info(f"Loaded cached runs from {cache_path}")

        except FileNotFoundError:
            logger.warning(f"Failed to load cache from {cache_path}, fetching runs from API...")
            run_instances = get_runs_from_api(
                api, workspace, experiment_groups, cache_path, no_cache, num_samples
            )

    # Filter out failed runs or runs without evals
    run_instances = [run for run in run_instances if run.samples.shape[0] > 0]

    logger.info(
        f"Found {len(run_instances)} runs in {workspace} that match group id filter, gathering samples..."
    )
    eval_metric_group = GroupedWandbMetrics.all_metrics
    eval_metric_group_name = eval_metric_group.name

    if group_average:
        eval_metric_group = GroupedWandbMetrics[group_average]
        eval_metric_group_name = f"avg_{group_average}"

    if group_metrics:
        eval_metric_group = GroupedWandbMetrics[group_metrics]
        eval_metric_group_name = group_metrics

    logger.info(f"Building correlation for metric group: {eval_metric_group_name}")
    logger.info(
        f"Found {len(run_instances)} valid run instances for group(s) {experiment_groups} to fit regression..."
    )

    launch_config = config_from_path(config)

    logger.info(f"Calculating source weights...")
    priors = calculate_priors(
        source_configs=launch_config.sources,
        dtype=launch_config.dtype,
        use_cache=(no_cache == False),
    )
    logger.info(f"Source weights:")
    logger.info(priors)

    run_ratios = [
        {"run": run.id, "index": idx, **mk_weights_from_config(run.config, priors)}
        for idx, run in enumerate(run_instances)
    ]

    run_metrics = [
        {
            "run": run.id,
            "index": idx,
            **mk_run_metrics(
                history=run.samples,
                samples=num_samples,
                metrics=(eval_metric_group_name, eval_metric_group.value),
                average=group_average != None,
            ),
        }
        for idx, run in enumerate(run_instances)
    ]

    ratios = pd.DataFrame(run_ratios)
    metrics = pd.DataFrame(run_metrics)

    # Domain weights
    X_train = ratios[ratios.columns[2:]].values
    X_test = ratios[ratios.columns[2:]].values

    # Metric values
    Y_train = metrics[metrics.columns[2:]].values
    Y_test = metrics[metrics.columns[2:]].values

    np.random.seed(42)
    predictors = []
    metrics_to_index = eval_metric_group.value

    if group_average:
        metrics_to_index = [eval_metric_group_name]

    indexed_metrics = list(enumerate(metrics_to_index))
    logger.info(f"Fitting regression for metrics:")
    logger.info(indexed_metrics)

    for idx, metric in indexed_metrics:
        predictors.append(build_regression(idx, Y_train, Y_test, X_train, X_test))

    results = []
    cached_samples = np.array([])

    for idx, metric in indexed_metrics:
        plot_correlation(
            Y_test,
            X_test,
            idx,
            predictors=predictors,
            metric_name=metric,
            alpha=alpha,
            output_dir=output_dir,
        )
        weights, samples = simulate(
            index=idx,
            predictor=predictors,
            df_config=ratios,
            prior_distributions=np.array(list(priors[0].values())),
            metric_name=metric,
            alpha=alpha,
            output_dir=output_dir,
            use_entropy=use_entropy,
            cached_samples=cached_samples,
            n_samples=simulation_samples,
        )

        cached_samples = samples
        results.append((metric, weights))

    if not group_average:
        # If we're not optimizing for the average of the metric group, then we average the reweighted distributions after fitting
        avg_name = f"avg_{eval_metric_group_name}"
        average = np.mean([result[1] for result in results], axis=0)
        columns = ratios.columns[2:].to_list()
        plot_distributions(
            prior=np.array(list(priors[0].values())),
            prediction=average,
            metric_name=avg_name,
            alpha=alpha,
            columns=columns,
            output_dir=output_dir,
        )

        with open(
            f"{mk_output_prefix(output_dir, avg_name, alpha=alpha)}_optimal.json",
            "w",
        ) as f:
            out = [{"domain": columns[idx], "weight": weight} for idx, weight in enumerate(average)]
            logger.info("Average of optimized weights:")
            logger.info(out)
            f.write(json.dumps(out))


if __name__ == "main":
    cli(obj={})
