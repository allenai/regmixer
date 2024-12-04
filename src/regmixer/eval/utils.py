import json
import logging
import pathlib
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import click
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from olmo_core.utils import prepare_cli_environment
from scipy.stats import spearmanr
from tqdm import tqdm
from wandb.apis.public import Run

from regmixer.eval.constants import GroupedWandbMetrics, WandbMetrics
from regmixer.synthesize_mixture import calculate_priors
from regmixer.utils import config_from_path

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

BASE_OUTPUT_DIR = "output/"
BASE_CACHE_DIR = "cache/"
DEFAULT_WORKSPACE = "ai2-llm/regmixer"
LGBM_HPS = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l1", "l2"],
    "seed": 42,
    "num_iterations": 10000,
    "learning_rate": 0.1,
    "verbosity": -1,
    "early_stopping_round": 3,
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


def get_output_dir(group: str) -> str:
    return f"{BASE_OUTPUT_DIR}{group}/"


@click.group()
def cli():
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
    type=bool,
    help="Select highest entropy samples for simulation.",
    required=False,
    default=True,
)
def fit(
    group: str,
    config: pathlib.Path,
    alpha: float,
    num_samples: int,
    group_average: Optional[str],
    group_metrics: Optional[str],
    workspace: str,
    no_cache: bool,
    use_entropy: bool,
):
    output_dir = get_output_dir(group)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(BASE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if group_average and group_metrics:
        raise ValueError("Cannot provide both group-average and group-metrics")

    api = wandb.Api()
    cache_path = pathlib.Path(BASE_CACHE_DIR) / f"{group}_runs_cache.json"

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
        source_configs=launch_config.sources,
        dtype=launch_config.dtype,
        use_cache=(no_cache == False),
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
        predictors.append(build_regression(idx, metric, Y_train, Y_test, X_train, X_test))

    results = []

    for idx, metric in indexed_metrics:
        _plot_correlation(
            Y_test,
            X_test,
            idx,
            predictors=predictors,
            metric_name=metric,
            alpha=alpha,
            output_dir=output_dir,
        )
        weights = simulate(
            index=idx,
            predictor=predictors,
            df_config=ratios,
            prior_distributions=np.array(list(priors[0].values())),
            metric_name=metric,
            alpha=alpha,
            output_dir=output_dir,
            use_entropy=use_entropy,
        )

        results.append((metric, weights))

    if not group_average:
        average = np.mean([result[1] for result in results], axis=0)
        _plot_distributions(
            prior=np.array(list(priors[0].values())),
            prediction=average,
            metric_name=f"avg_{eval_metric_group_name}",
            alpha=alpha,
            columns=ratios.columns[2:].to_list(),
            output_dir=output_dir,
        )


def build_regression(
    idx: int,
    metric: str,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> lgb.LGBMRegressor:
    target = Y_train[:, idx]
    test_target = Y_test[:, idx]

    gbm = lgb.LGBMRegressor(**LGBM_HPS)

    regression = gbm.fit(
        X_train,
        target,
        eval_set=[(X_test, test_target)],
        eval_metric="l2",
    )
    r, _ = spearmanr(a=regression.predict(X_test), b=test_target)
    corr = np.round(r * 100, decimals=2)

    logger.info(f"{idx}: {metric} :: Correlation: {corr}")

    return regression


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


def _plot_simulations(
    prior_distributions: np.ndarray,
    samples,
    columns: list[str],
    metric_name: str,
    alpha: float,
    output_dir: str = BASE_OUTPUT_DIR,
):
    plt.close()
    df = pd.DataFrame(
        data=np.concatenate([np.array([prior_distributions]), samples], axis=0),
        columns=columns,
    )
    df = df.sample(n=20, random_state=42)
    df["sample"] = df.index
    melted_df = df.melt(id_vars=["sample"], var_name="Domain", value_name="Weight")
    g = sns.FacetGrid(melted_df, col="sample", col_wrap=4, aspect=2)
    g.map_dataframe(sns.barplot, x="Domain", y="Weight", palette="viridis", hue="Domain")
    g.set(ylim=(0, 0.5))
    g.set_axis_labels("Domain", "Weight")

    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
        ax.yaxis.grid(True, linestyle="--", which="both", color="gray", alpha=0.7)

    plt.savefig(
        f"{_mk_plot_prefix(output_dir, metric_name, alpha)}_sim_grid.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


def _plot_correlation(
    Y_test: np.ndarray,
    X_test: np.ndarray,
    index: int,
    predictors: list[lgb.LGBMRegressor],
    metric_name: str,
    alpha: Optional[float] = None,
    output_dir: str = BASE_OUTPUT_DIR,
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
        scatter_kws={"s": 64, "color": "#105257"},
        joint_kws={
            "line_kws": {
                "color": "#F0529C",
                "linewidth": 4,
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
            "size": 18,
        },
        handlelength=-0.5,
    )

    graph.ax_joint.set_ylabel(keys["true"], fontdict={"size": 24})
    graph.ax_joint.set_xlabel(keys["pred"], fontdict={"size": 24})
    graph.ax_marg_x.remove()
    graph.ax_marg_y.remove()
    graph.ax_joint.grid(True, ls="dashed")
    graph.ax_joint.spines[["right", "top"]].set_visible(True)

    graph.savefig(f"{_mk_plot_prefix(output_dir, metric_name, alpha)}_fit.png", bbox_inches="tight")


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


def simulate(
    index: int,
    predictor: list[lgb.LGBMRegressor],
    prior_distributions: np.ndarray,
    df_config: pd.DataFrame,
    metric_name: str,
    use_entropy: bool,
    n_samples: int = 100_000,
    alpha: float = 1.0,
    normalization: bool = False,
    min_entropy: float = 1e-3,
    output_dir: str = BASE_OUTPUT_DIR,
) -> np.ndarray:
    np.random.seed(42)
    num_samples_per_strength = 25
    candidates = []

    logger.info(
        f"Generating {n_samples * num_samples_per_strength:,} sample candidates for {metric_name}..."
    )
    for idx in tqdm(range(n_samples), desc="Generating candidates"):
        min_strength_log = np.log10(min_entropy)
        max_strength_log = np.log10(alpha)

        for strength in np.logspace(min_strength_log, max_strength_log, num_samples_per_strength):
            weights = np.random.dirichlet(prior_distributions * strength, 1)

            if normalization:
                weights = (weights * 1.5 + prior_distributions) / 2.5

            candidates.append(weights)

    all_samples = np.array(candidates).reshape(-1, len(prior_distributions))
    all_samples = all_samples[~np.isnan(all_samples).any(axis=1)]
    logger.info(f"Generated {all_samples.shape} valid samples for {metric_name}...")

    if use_entropy:
        entropy = -np.sum(all_samples * np.log(all_samples + min_entropy), axis=1)
        high_entropy_indices = np.argsort(entropy)[-n_samples:]
        samples = all_samples[high_entropy_indices]
    else:
        samples = all_samples[np.random.choice(all_samples.shape[0], n_samples, replace=False)]

    logger.info(f"Simulating with {samples.shape[0]:,} samples for {metric_name}...")
    simulation = predictor[index].predict(samples)

    columns = df_config.columns[2:]
    _plot_simulations(
        prior_distributions=prior_distributions,
        samples=samples,
        columns=columns.to_list(),
        metric_name=metric_name,
        alpha=alpha,
        output_dir=output_dir,
    )

    plt.close()
    plt.hist(simulation, bins=32, color="#F0529C")
    plt.xlabel("Predicted")
    plt.ylabel("Frequency")
    plt.savefig(
        f"{_mk_plot_prefix(output_dir, metric_name, alpha)}_sim_dist.png",
        bbox_inches="tight",
    )
    plt.close()

    k = 128
    top_k_simulations = np.argsort(simulation)[0:k]
    # TODO: Make this conditional on the evaluation metric. ie: loss is min but downstream is max
    logger.info(f"Best prediction: {np.min(simulation)}")
    top_k_samples = samples[top_k_simulations]
    top_k_samples.shape

    top_k_mean_weights = np.mean(top_k_samples, axis=0)
    top_k_predicted_loss = predictor[index].predict([top_k_mean_weights])

    print("\n")
    logger.info(f"Predicted loss (top_k): {top_k_predicted_loss}")
    print("\n")
    logger.info(f":::::::::{metric_name}:::::::::")
    logger.info("Predicted optimal weights:")

    with open(f"{_mk_plot_prefix(output_dir, metric_name, alpha=alpha)}_optimal.json", "w") as f:
        out = []
        for idx, weight in enumerate(top_k_mean_weights):
            out.append({"domain": columns[idx], "weight": weight})

        logger.info(out)
        f.write(json.dumps(out))

    _plot_distributions(
        prior=prior_distributions,
        prediction=top_k_mean_weights,
        metric_name=metric_name,
        alpha=alpha,
        columns=columns.to_list(),
        output_dir=output_dir,
    )

    return top_k_mean_weights


def _plot_distributions(
    prior: np.ndarray,
    prediction: np.ndarray,
    metric_name: str,
    alpha: float,
    columns: list[str],
    output_dir: str = BASE_OUTPUT_DIR,
):
    df = pd.DataFrame(
        data=np.concatenate(
            [
                np.array([prior]),
                np.array([prediction]),
            ],
            axis=0,
        ),
        columns=columns,
    )
    df = pd.melt(df)
    df["type"] = (["Original"] + ["Optimal"]) * len(columns)

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
        ncol=3,
    )

    ax.yaxis.grid(True, linestyle="--", which="both", color="gray", alpha=0.7)
    ax.set_ylim(0, 0.5)

    ax.set_xlabel(
        "Domain",
        fontdict={
            "size": 28,
        },
    )
    ax.set_ylabel(
        "Weight",
        fontdict={
            "size": 28,
        },
    )

    plt.savefig(
        f"{_mk_plot_prefix(output_dir, metric_name, alpha=alpha)}_optimal.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )


def _mk_plot_prefix(outout_dir: str, metric: str, alpha: Optional[float] = None) -> str:
    def sanitize(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)

    return f"{outout_dir}{sanitize(metric)}" + (
        f"_alpha_{str(alpha).replace('.', '_')}" if alpha and alpha != 1.0 else ""
    )


if __name__ == "main":
    cli(obj={})
