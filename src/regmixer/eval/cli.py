import json
import logging
import pathlib
import warnings
from typing import Optional
from copy import deepcopy 
from sklearn.model_selection import train_test_split

import click
import numpy as np
import pandas as pd
import wandb
import torch 
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
    plot_correlation,
    plot_and_log_weights,
    simulate2,
    save_eval_config,
    solve_log_linear,
    PROPOSER_TYPES
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
@click.option(
    "-r",
    "--regression-type",
    type=str,
    default="lightgbm",
    help="Whether to use LightGBM or linear regression for fitting",
    required=False,
)
@click.option(
    "-t",
    "--train-split",
    type=float,
    default=1.0,
    help="Fraction of dataset used for training. Default = 1.0 means that train equals test.",
    required=False,
)
@click.option(
    "--n-test",
    type=int,
    default=0,
    help="Number of test samples we evaluate regression model on, primarily used for analysis.",
    required=False,
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="Random state for train-test split",
    required=False,
)
@click.option(
    "--opt-avg-metric",
    is_flag=True,
    help="If set, each metric is fit separately, and then a mixture is selected to minimize the average of all metrics",
    required=False,
    default=False,


)
@click.option(
    "--proposer-type",
    type=str,
    help="Proposer type: either simulation or search",
    required=False,
    default="simulation",
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
    regression_type: str,
    train_split: float,
    n_test: int,
    seed: int,
    opt_avg_metric: bool,
    proposer_type: str,
):

    output_dir = get_output_dir(experiment_groups)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(BASE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if group_average and group_metrics:
        raise ValueError("Cannot provide both group-average and group-metrics")
    
    if proposer_type == "search" and regression_type != "search":
        raise ValueError("Proposer type search only works with regression type search")
    
    eval_config = {
        "config": config,
        "alpha": alpha,
        "num_samples": num_samples,
        "simulation_samples": simulation_samples,
        "group_average": group_average,
        "group_metrics": group_metrics,
        "workspace": workspace,
        "regression_type": regression_type,
        "train_split": train_split,
        "n_test": n_test,
        "seed": seed,
        "opt_avg_metric": opt_avg_metric,
    }
    if proposer_type != "simulation":
        eval_config["proposer_type"] = proposer_type

    output_dir = save_eval_config(eval_config, output_dir)



    api = wandb.Api()


    eval_metric_group = GroupedWandbMetrics.all_metrics
    eval_metric_group_name = eval_metric_group.name

    if group_average:
        eval_metric_group = GroupedWandbMetrics[group_average]
        eval_metric_group_name = f"avg_{group_average}"

    if group_metrics:
        eval_metric_group = GroupedWandbMetrics[group_metrics]
        eval_metric_group_name = group_metrics

    cache_path = pathlib.Path(BASE_CACHE_DIR) / f"{'_'.join(experiment_groups)}_{eval_metric_group}_runs_cache.json"


    if no_cache:
        logger.info(f"Cache disabled, will not use cache for run samples...")
        run_instances = get_runs_from_api(
            api, workspace, experiment_groups, cache_path, no_cache, num_samples, eval_metric_group
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
                api, workspace, experiment_groups, cache_path, no_cache, num_samples, eval_metric_group
            )

    # Filter out failed runs or runs without evals
    run_instances = [run for run in run_instances if run.samples.shape[0] > 0]

    logger.info(
        f"Found {len(run_instances)} runs in {workspace} that match group id filter, gathering samples..."
    )

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
        for idx, run in enumerate(run_instances) if len(run.samples) > 0
    ]
    ratios = pd.DataFrame(run_ratios)
    metrics = pd.DataFrame(run_metrics)

    # X = Domain weights
    X_train = ratios[ratios.columns[2:]].values
    # Y = Metric values 
    Y_train = metrics[metrics.columns[2:]].values

    if n_test == 0:
        # if n_test is 0
        X_test = deepcopy(X_train)
        Y_test = deepcopy(Y_train)
    else:
        # If we want to evaluate on a fixed number of test samples
        logger.info(f"Using {n_test} samples for test data")
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=n_test / len(Y_train), random_state=seed)

        if train_split != 1.0:
            # If we also want to subsample the training_data to study the effect of number of proxy runs
            logger.info(f"Subsampling training data to {train_split} of original size")
            X_train, _, Y_train, _ = train_test_split(X_train, Y_train, train_size=train_split, random_state=seed)


    logger.info(f"Number of train samples: {len(Y_train)}. Number of test samples: {len(Y_test)}.")

    predictors = []
    metrics_to_index = eval_metric_group.value

    if group_average:
        metrics_to_index = [eval_metric_group_name]

    indexed_metrics = list(enumerate(metrics_to_index))
    logger.info(f"Fitting {regression_type} regression for metrics:")
    logger.info(indexed_metrics)


    for idx, metric in indexed_metrics:
        predictors.append(build_regression(idx, Y_train, X_train, regression_type))

    results = []

    for idx, metric in indexed_metrics:
        plot_correlation(
            Y_test,
            X_test,
            Y_train,
            X_train,
            idx,
            predictors=predictors,
            train_split=train_split,
            metric_name=metric,
            regression_type=regression_type,
            n_test=n_test,
            split_seed=seed,
            n_samples=num_samples,
            alpha=alpha,
            output_dir=output_dir,
        )

        if not opt_avg_metric and n_test == 0:

            weights = PROPOSER_TYPES[proposer_type]().propose(
                index=idx,
                predictor=predictors,
                prior_distributions=np.array(list(priors[0].values())),
                num_samples=simulation_samples,
                opt_avg_metric=opt_avg_metric
            )

            plot_and_log_weights(
                prior=np.array(list(priors[0].values())),
                prediction=weights,
                metric_name=metric,
                regression_type=regression_type,
                train_split=train_split,
                n_test=n_test,
                split_seed=seed,
                n_samples=num_samples,
                alpha=alpha,
                df_config=ratios,
                output_dir=output_dir,
            )

            """weights = simulate2(
                index=idx,
                predictor=predictors,
                df_config=ratios,
                prior_distributions=np.array(list(priors[0].values())),
                metric_name=metric,
                regression_type=regression_type,
                train_split=train_split,
                n_test=n_test,
                split_seed=seed,
                n_samples=num_samples,
                alpha=alpha,
                output_dir=output_dir,
                num_samples=simulation_samples,
            )"""

            results.append((metric, weights))

    if opt_avg_metric and n_test == 0:
        assert group_metrics is not None and group_average is None # need to have this set
        weights = PROPOSER_TYPES[proposer_type]().propose(
            index=-1,
            predictor=predictors,
            prior_distributions=np.array(list(priors[0].values())),
            num_samples=simulation_samples,
            opt_avg_metric=opt_avg_metric
        )
        plot_and_log_weights(
            prior=np.array(list(priors[0].values())),
            prediction=weights,
            metric_name=group_metrics,
            regression_type=regression_type,
            train_split=train_split,
            n_test=n_test,
            split_seed=seed,
            n_samples=num_samples,
            alpha=alpha,
            df_config=ratios,
            output_dir=output_dir,
        )


        """        if regression_type in ["lightgbm", "linear", "log_linear"]:
                    weights = simulate2(
                            index=-1,
                            predictor=predictors,
                            df_config=ratios,
                            prior_distributions=np.array(list(priors[0].values())),
                            metric_name="opt_avg",
                            regression_type=regression_type,
                            train_split=train_split,
                            n_test=n_test,
                            split_seed=seed,
                            n_samples=num_samples,
                            alpha=alpha,
                            output_dir=output_dir,
                            num_samples=simulation_samples,
                    )
                elif regression_type == "log_linear":
                    weights = solve_log_linear(
                        predictor=predictors,
                        prior_distributions=np.array(list(priors[0].values())),
                        df_config=ratios,
                        metric_name="opt_avg",
                        regression_type=regression_type,
                        train_split=train_split,
                        n_test=n_test,
                        split_seed=seed,
                        n_samples=num_samples,
                        alpha=alpha,
                        output_dir=output_dir,
                    )
        """


    elif not group_average:
        # If we're not optimizing for the average of the metric group, then we average the reweighted distributions after fitting
        avg_name = f"avg_{eval_metric_group_name}"
        average = np.mean([result[1] for result in results], axis=0)
        plot_and_log_weights(
            prior=np.array(list(priors[0].values())),
            prediction=average,
            metric_name=avg_name,
            regression_type=regression_type,
            train_split=train_split,
            n_test=n_test,
            split_seed=seed,
            n_samples=num_samples,
            alpha=alpha,
            df_config=ratios,
            output_dir=output_dir,
        )


    for i, result in enumerate(results):
        metric, weights = result
        if n_test == 0:
            predicted_performance = predictors[i].predict(weights[None])
            logger.info(f"Metric: {metric}. Predicted performance using regression model: {predicted_performance}")

    logger.info(f"Results saved to {output_dir}")


if __name__ == "main":
    cli(obj={})
