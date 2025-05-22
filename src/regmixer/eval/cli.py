import json
import logging
import pathlib
import warnings
from typing import Optional
from copy import deepcopy 
from sklearn.model_selection import train_test_split

from pathlib import Path
import click
import numpy as np
import pandas as pd
import wandb
import torch 
import yaml
from olmo_core.utils import prepare_cli_environment

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


from regmixer.eval.constants import GroupedWandbMetrics, ObjectiveWeights
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
    plot_interaction_matrix,
    compute_mixture_neighborhood,
    filter_constrained_swarm,
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
    multiple=True,
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
    default=1,
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
@click.option(
    "--neighborhood",
    type=str,
    help="the training run display name that defines the neighborhood of subselected mixtures we regress on",
    required=False,
    default=None,
)
@click.option(
    "--final-cookbook-path",
    type=click.Path(exists=True),
    help="Path to cookbook config containing information about the full run for which the proposed mix is used (number of tokens, dataset being used)",
    required=False,
    default="/home/mayee/re/year6/ai2/olmo-cookbook/src/cookbook/recipes/train-1b-v2-5xC-dclm-larger-natural.yaml"
)
@click.option(
    "--constrain-swarm",
    is_flag=True,
    help="If set, we only use swarm runs that are unconstrained according to the final cookbook config.",
    required=False,
    default=False
)
@click.option(
    "--constrain-objective",
    is_flag=True,
    help="If set, we produce a proposed mix that is unconstrained according to the final cookbook config.",
    required=False,
    default=False
)
@click.option(
    "--obj-weights",
    type=str,
    help="The non-uniform weights used to average BPB over all tasks. If not set, uniform weights are used.",
    required=False,
    default=None,
)
@click.option(
    "--temperature",
    type=float,
    help="The temperature used to adjust the dirichlet prior in the simulation process. Closer to 0 = more uniform." ,
    required=False,
    default=None,
)
@click.option(
    "--keep-sources",
    type=str,
    multiple=True,
    help="If set, we only use swarm runs that have nonzero weight on keep_sources for the regression.",
    required=False,
    default=None,
)
def fit(
    experiment_groups: list[str],
    config: list[pathlib.Path],
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
    neighborhood: Optional[str],
    final_cookbook_path: Optional[Path],
    constrain_swarm: bool,
    constrain_objective: bool ,
    obj_weights: Optional[str],
    temperature: Optional[float],
    keep_sources: Optional[list[str]],
):

    output_dir = get_output_dir(experiment_groups)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(BASE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if group_average and group_metrics:
        raise ValueError("Cannot provide both group-average and group-metrics")
    
    if proposer_type == "search" and regression_type != "search":
        raise ValueError("Proposer type search only works with regression type search")
    
    eval_config = {
        "config": config[0] if len(config) == 1 else config,
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
    if neighborhood is not None:
        eval_config["neighborhood"] = neighborhood
    if constrain_swarm:
        eval_config["constrain_swarm"] = True 
        eval_config["final_cookbook_path"] = final_cookbook_path 
    if constrain_objective:
        eval_config["constrain_objective"] = True 
        eval_config["final_cookbook_path"] = final_cookbook_path 
    if obj_weights is not None:
        eval_config["obj_weights"] = obj_weights
    if temperature is not None:
        eval_config["temperature"] = temperature
    if len(keep_sources) != 0:
        eval_config["keep_sources"] = keep_sources

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


        

    cache_path = pathlib.Path(BASE_CACHE_DIR) / f"{'_'.join(experiment_groups)}_{eval_metric_group_name}_runs_cache.json"

    launch_configs = [config_from_path(c) for c in config]
    full_group_names = [f"{launch_config.name}-{group}" for group, launch_config in zip(experiment_groups, launch_configs)]
    if no_cache:
        logger.info(f"Cache disabled, will not use cache for run samples...")
        run_instances = get_runs_from_api(
            api, workspace, full_group_names, cache_path, no_cache, num_samples, eval_metric_group
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
                api, workspace, full_group_names, cache_path, no_cache, num_samples, eval_metric_group
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

    logger.info(f"Calculating source weights...")
    priors = calculate_priors(
        source_configs=launch_configs[0].sources,
        dtype=launch_configs[0].dtype,
        use_cache=(no_cache == False),
    )
    logger.info(f"Source weights:")
    logger.info(priors)

    run_ratios = [
        {"run": run.id, "name": run.display_name, "index": idx, **mk_weights_from_config(run.config, priors)}
        for idx, run in enumerate(run_instances)
    ]

    run_metrics = [
        {
            "run": run.id,
            "name": run.display_name, 
            "index": idx,
            **mk_run_metrics(
                history=run.samples,
                samples=num_samples,
                metrics=(eval_metric_group_name, eval_metric_group.value),
                display_name=run.display_name,
                average=group_average != None,
            ),
        }
        for idx, run in enumerate(run_instances) if len(run.samples) > 0
    ]

    if constrain_swarm:
        run_ratios, run_metrics = filter_constrained_swarm(final_cookbook_path, run_ratios, run_metrics)

    ratios = pd.DataFrame(run_ratios)
    metrics = pd.DataFrame(run_metrics)

    if len(ratios[ratios.columns[3:]]) > len(ratios):
        raise ValueError("The number of swarm runs is fewer than the number of mixing sources.")
    
    if len(keep_sources) != 0:
        old_len = len(ratios)
        other_columns = list(set(ratios.columns[3:]).difference(set(keep_sources)))
        ratios = ratios[ratios[list(keep_sources)].ne(0).all(axis=1) &  # all specified columns nonzero
            ratios[other_columns].eq(0).all(axis=1)  ]
        logger.info(f"Filtered out {old_len - len(ratios)} runs that were not only on {keep_sources}")
        metrics = metrics[metrics['name'].isin(ratios['name'])]
        ratios.drop(columns=other_columns, inplace=True)

    # X = Domain weights
    X_train = ratios[ratios.columns[3:]].values
    # Y = Metric values 
    Y_train = metrics[metrics.columns[3:]].values

    if n_test > 0:
        logger.info(f"Using {n_test} samples for test data")
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=n_test / len(Y_train), random_state=seed)


    if train_split != 1.0:
        # If we also want to subsample the training_data to study the effect of number of proxy runs
        logger.info(f"Subsampling training data to {train_split} of original size")

        if neighborhood is None:
            # we IID subselect training data
            X_train, _, Y_train, _ = train_test_split(X_train, Y_train, train_size=train_split, random_state=seed)
        else:
            X_train, Y_train = compute_mixture_neighborhood(X_train, Y_train, ratios, neighborhood, train_split)

    if n_test == 0:
        X_test = deepcopy(X_train)
        Y_test = deepcopy(Y_train)

    logger.info(f"Number of train samples: {len(Y_train)}. Number of test samples: {len(Y_test)}.")

    predictors = []
    metrics_to_index = eval_metric_group.value

    if group_average:
        metrics_to_index = [eval_metric_group_name]

    indexed_metrics = list(enumerate(metrics_to_index))
    logger.info(f"Fitting {regression_type} regression for metrics:")
    logger.info(indexed_metrics)

    if obj_weights:
        obj_weights = ObjectiveWeights[obj_weights]
        obj_weights = [obj_weights.value.get(metric, 1) for idx, metric in indexed_metrics]
        logger.info(f"Minimizing weighted average: {obj_weights}")

    for idx, metric in indexed_metrics:
        predictors.append(build_regression(idx, Y_train, X_train, regression_type))

    plot_interaction_matrix(output_dir, predictors, regression_type, ratios.columns[3:].tolist(), metrics.columns[3:].tolist())
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
                prior_distributions=priors[0],
                num_samples=simulation_samples,
                opt_avg_metric=opt_avg_metric,
                constrain_objective=constrain_objective,
                final_cookbook_path=final_cookbook_path,
                obj_weights=obj_weights,
                temperature=temperature
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

            results.append((metric, weights))

    if opt_avg_metric and n_test == 0:
        assert group_metrics is not None and group_average is None # need to have this set
        weights = PROPOSER_TYPES[proposer_type]().propose(
            index=-1,
            predictor=predictors,
            prior_distributions=priors[0],
            num_samples=simulation_samples,
            opt_avg_metric=opt_avg_metric,
            constrain_objective=constrain_objective,
            final_cookbook_path=final_cookbook_path,
            obj_weights=obj_weights,
            temperature=temperature
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

        results.append((group_metrics, weights))

    elif not group_average and n_test == 0:
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

        results.append((avg_name, average))

    if n_test == 0:
        metric, weights = results[-1]
        if obj_weights is not None:
            predictions = [p.predict(weights[None])[0] for p in predictors]
            predicted_performance = np.average(predictions, axis=0, weights=obj_weights)
        else:
            predicted_performance = np.array([p.predict(weights[None])[0] for p in predictors]).mean()
        logger.info(f"Metric: {metric}. Predicted performance using regression model: {predicted_performance}")

        with open(f"{output_dir}/predicted_performance.json", "w") as f:
            json.dump(float(predicted_performance), f)


    logger.info(f"Results saved to {output_dir}")


if __name__ == "main":
    cli(obj={})
