from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import json
import logging
import pathlib

from wandb.apis.public import Run
import click
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

from olmo_core.utils import prepare_cli_environment
from scipy.stats import spearmanr

from regmixer.eval.constants import GroupedWandbMetrics, WandbMetrics
from regmixer.synthesize_mixture import calculate_priors
from regmixer.utils import config_from_path
import warnings
import re

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

OUTPUT_DIR = "output/"
CACHE_DIR = "cache/"
DEFAULT_WORKSPACE = "ai2-llm/regmixer"
LGBM_HPS = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l1", "l2"],
    "seed": 42,
    "num_iterations": 10000,
    "learning_rate": 1e-2,
    "verbosity": -1,
}


@dataclass
class RunInstance:
    id: str
    display_name: str
    config: dict
    samples: pd.DataFrame

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "config": self.config,
            "samples": self.samples.to_dict(),
        }


@click.group()
def cli():
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    prepare_cli_environment()


@cli.command()
@click.option(
    "--group",
    "-g",
    type=str,
    help="The group ID to fit the regression model against",
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
    "-t",
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature to apply to the optimal weights",
    required=False,
)
@click.option(
    "-a",
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
    help="The number of evaluation samples per metric to average when fitting the regression",
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
def fit(
    group: str,
    config: pathlib.Path,
    temperature: float,
    num_samples: int,
    group_average: Optional[str],
    group_metrics: Optional[str],
    workspace: str,
    no_cache: bool,
):

    if group_average and group_metrics:
        raise ValueError("Cannot provide both group-average and group-metrics")

    api = wandb.Api()
    cache_path = pathlib.Path(CACHE_DIR) / f"{group}_runs_cache.json"

    if no_cache:
        logger.info(f"Cache disabled, will not use cache for run samples...")
        group_runs = get_runs_from_api(api, workspace, group, cache_path, no_cache, num_samples)
    else:
        try:
            with open(cache_path, "r") as f:
                run_dict = json.load(f)
                group_runs = [mk_run_from_json(run) for run in run_dict]

            logger.info(f"Loaded cached runs from {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}, error: {e}")
            group_runs = get_runs_from_api(api, workspace, group, cache_path, no_cache, num_samples)

    logger.info(
        f"Found {len(group_runs)} runs in {workspace} that match group id filter, gathering samples..."
    )
    eval_metric_group = GroupedWandbMetrics.all_metrics
    eval_metric_group_name = eval_metric_group.name

    if group_average:
        eval_metric_group = GroupedWandbMetrics[group_average]
        eval_metric_group_name = f"avg_{group_average}"

    if group_metrics:
        eval_metric_group = GroupedWandbMetrics[group_metrics]
        eval_metric_group_name = group_metrics

    logger.info(f"Building correlation with: {eval_metric_group_name}")
    logger.info(
        f"Found {len(group_runs)} valid run instances for group {group} to fit regression..."
    )

    launch_config = config_from_path(config)

    logger.info(f"Calculating source weights...")
    priors = calculate_priors(
        source_configs=launch_config.sources, dtype=launch_config.dtype, use_cache=True
    )
    logger.info(f"Source weights:")
    logger.info(priors)

    run_ratios = [
        {"run": run.id, "index": idx, **_mk_weights_from_config(run.config, priors)}
        for idx, run in enumerate(group_runs)
    ]

    run_metrics = [
        {
            "run": run.id,
            "index": idx,
            **_mk_run_metrics(
                history=run.samples,
                samples=num_samples,
                metrics=(eval_metric_group_name, eval_metric_group.value),
                average=group_average != None,
            ),
        }
        for idx, run in enumerate(group_runs)
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
        target = Y_train[:, idx]
        test_target = Y_test[:, idx]

        gbm = lgb.LGBMRegressor(**LGBM_HPS)

        regression = gbm.fit(
            X_train,
            target,
            eval_set=[(X_test, test_target)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(stopping_rounds=3, verbose=False),
            ],
        )
        r, _ = spearmanr(a=regression.predict(X_test), b=test_target)
        corr = np.round(r * 100, decimals=2)
        logger.info(f"{idx}: {metric} :: Correlation: {corr}")

        predictors.append(regression)

    for idx, metric in indexed_metrics:
        _plot_correlation(
            Y_test, X_test, idx, predictors=predictors, metric_name=metric, temperature=temperature
        )
        _simulate(
            index=idx,
            predictor=predictors,
            df_config=ratios,
            prior_distributions=list(priors[0].values()),
            metric_name=metric,
            temperature=temperature,
        )


def get_runs_from_api(
    api, workspace: str, group: str, cache_path: Path, no_cache: bool, num_samples: int
) -> list[RunInstance]:

    wandb_runs = api.runs(
        path=workspace,
        filters={"display_name": {"$regex": f".*{group}.*"}},
    )

    # NOTE: Last writer wins and should be sorted created_at desc
    memo = {}
    for run in wandb_runs:
        memo[run.display_name] = run

    all_runs: list[RunInstance] = sorted(
        [mk_run_history(run, num_samples) for run in memo.values()],
        key=lambda run: run.display_name.lower(),
    )

    if not no_cache:
        with open(cache_path, "w") as f:
            json.dump([run.as_dict() for run in all_runs], f)

    return all_runs


def mk_run_history(run: Run, samples: int) -> Any:
    return mk_run_instance(
        run,
        run.history(samples=samples, pandas=False, keys=[metric.value for metric in WandbMetrics]),
    )


def mk_run_from_json(run: dict) -> RunInstance:
    return RunInstance(
        id=run["id"],
        display_name=run["display_name"],
        config=run["config"],
        samples=pd.DataFrame(run["samples"]),
    )


def mk_run_instance(run: Run, history: list[Any]) -> RunInstance:
    samples = pd.DataFrame.from_records(history)
    logger.info(
        f"Collected RunInstance for {run.display_name}:{run.id} with samples: {samples.shape}"
    )
    return RunInstance(
        id=run.id,
        display_name=run.display_name,
        config=run.config,
        samples=samples,
    )


def _plot_correlation(
    Y_test: np.ndarray,
    X_test: np.ndarray,
    index: int,
    predictors: list[lgb.LGBMRegressor],
    metric_name: str,
    temperature: Optional[float] = None,
):
    plt.close()
    keys = {"pred": "Predicted", "true": "Actual"}
    data = {keys["true"]: Y_test[:, index], keys["pred"]: predictors[index].predict(X_test)}
    graph = sns.jointplot(
        data,
        x=keys["pred"],
        y=keys["true"],
        kind="reg",
        height=10,
        scatter_kws={"s": 128, "color": "#5969CB"},
        joint_kws={
            "line_kws": {
                "color": "#C3364A",
                "linewidth": 6,
                "linestyle": "dashed",
            }
        },
        marginal_kws={"line_kws": {"color": "#5969CB", "linewidth": 6}},
    )

    r, _ = spearmanr(data[keys["pred"]], data[keys["true"]])
    (phantom,) = graph.ax_joint.plot([], [], linestyle="", alpha=0)

    graph.ax_joint.legend(
        [phantom],
        [f"{metric_name} correlation: {np.round(r * 100, decimals=2)}"],  # noqa
        edgecolor="black",
        fancybox=False,
        prop={
            "size": 24,
        },
        handlelength=-0.5,
    )

    graph.ax_joint.set_ylabel(keys["true"], fontdict={"size": 32})
    graph.ax_joint.set_xlabel(keys["pred"], fontdict={"size": 32})
    graph.ax_marg_x.remove()
    graph.ax_marg_y.remove()
    graph.ax_joint.grid(True, ls="dashed")
    graph.ax_joint.spines[["right", "top"]].set_visible(True)

    graph.savefig(f"{_mk_plot_prefix(metric_name, temperature)}_fit.png", bbox_inches="tight")


def _mk_run_metrics(
    history,
    samples: int,
    metrics: Tuple[str, list[str]],
    average: bool = False,
) -> dict[str, float]:
    df = pd.DataFrame(history)
    results = {}
    group_name, group_metrics = metrics
    if average:
        result = np.mean(
            [df.loc[:, metric_name].tail(samples).mean() for metric_name in group_metrics]
        )

        results[group_name] = result
    else:
        for metric_name in group_metrics:
            results[metric_name] = df.loc[:, metric_name].tail(samples).mean()

    return results


def _mk_weights_from_config(config: dict, priors: tuple) -> dict[str, float]:
    source_configs = {
        source["source_name"]: source
        for source in config.get("dataset", {})
        .get("source_mixture_config", {})
        .get("source_configs", [])
    }
    weights = {}
    for source_name in priors[0].keys():
        weights[source_name] = source_configs.get(source_name, {}).get("target_ratio", 0.0)

    return weights


def _simulate(
    index: int,
    predictor: list[lgb.LGBMRegressor],
    prior_distributions: list[float],
    df_config: pd.DataFrame,
    metric_name: str,
    n_samples: int = 1_000_000,
    temperature: float = 1.0,
):
    np.random.seed(42)

    samples = np.random.dirichlet(prior_distributions * 1, n_samples)
    simulation = predictor[index].predict(samples)

    plt.close()
    plt.hist(simulation, bins=32)
    plt.savefig(f"{_mk_plot_prefix(metric_name, temperature)}_hist.png", bbox_inches="tight")
    plt.close()

    k = 128
    top_k_samples = samples[np.argsort(simulation)[0:k]]
    top_k_samples.shape

    predicted_domain_weights = np.mean(top_k_samples, axis=0)
    final_weights = (predicted_domain_weights + prior_distributions) / 2

    columns = df_config.columns[2:]

    df = pd.DataFrame(
        data=np.concatenate([np.array([prior_distributions]), top_k_samples], axis=0),
        columns=columns,
    )
    df = pd.melt(df)
    df["type"] = (["Original"] + ["Optimal"] * top_k_samples.shape[0]) * len(columns)

    logger.info(f":::::::::{metric_name}:::::::::")
    logger.info("Predicted optimal weights:")

    with open(f"{_mk_plot_prefix(metric_name, temperature=temperature)}_optimal.json", "w") as f:
        out = []
        for idx, weight in enumerate(final_weights):
            out.append({"domain": columns[idx], "weight": weight})

        logger.info(out)
        f.write(json.dumps(out))

    plt.rc("axes", unicode_minus=False)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 18,
        }
    )

    _, ax = plt.subplots(figsize=(12, 10), layout="compressed")
    ax.ticklabel_format(useMathText=True)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.tick_params(axis="x", labelrotation=90)

    pallette = {
        "Original": "#105257",
        "Optimal": "#F0529C",
    }
    sns.barplot(data=df, x="variable", y="value", hue="type", palette=pallette, ax=ax)

    ax.legend(
        edgecolor="black",
        fancybox=False,
        prop={
            "size": 18,
        },
        handlelength=0.5,
        ncol=2,
    )

    ax.grid(True)
    ax.set_ylim(0, 1.1)

    ax.set_xlabel(
        "Domain",
        fontdict={
            "size": 32,
        },
    )
    ax.set_ylabel(
        "Weight",
        fontdict={
            "size": 32,
        },
    )

    plt.savefig(
        f"{_mk_plot_prefix(metric_name, temperature=temperature)}_optimal.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )


def _mk_plot_prefix(metric: str, temperature: Optional[float] = None) -> str:
    def sanitize(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)

    return f"{OUTPUT_DIR}{sanitize(metric)}" + (
        f"_temp_{str(temperature).replace('.', '_')}" if temperature and temperature != 1.0 else ""
    )


if __name__ == "main":
    cli()
