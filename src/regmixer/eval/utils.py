import logging
import pathlib

import click
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from olmo_core.utils import prepare_cli_environment
from scipy.stats import spearmanr

from regmixer.eval.constants import WandbMetrics, GroupedWandbMetrics

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/"
DEFAULT_WORKSPACE = "ai2-llm/regmixer"
LGBM_HPS = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l1", "l2"],
    "seed": 42,
    "learning_rate": 1e-2,
    "verbosity": -1,
}


@click.group()
def cli():
    try:
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        logger.warning(f"Directory {OUTPUT_DIR} already exists, skipping creation...")

    prepare_cli_environment()


@cli.command()
@click.option(
    "--group",
    "-g",
    type=str,
    help="The group ID to fit the regression model against",
    required=True,
)
def fit(group: str):
    api = wandb.Api()
    all_runs = api.runs(
        path="ai2-llm/regmixer",
        filters={"display_name": {"$regex": f"{group}$"}},
    )
    logger.info(f"Found {len(all_runs)} runs that match group id filter, gathering samples...")
    filtered = {}
    for run in all_runs:
        try:
            group_id = run.config["trainer"]["callbacks"]["wandb"]["group"]
            run_id = run.display_name
            if group_id == group:
                existing = filtered.get(run_id)
                # If we have a run with no samples, we need to update metrics for the run.
                if existing and existing[1].shape[0] < 1:
                    filtered[run_id] = (
                        run_id,
                        run.history(
                            samples=1, pandas=(True), keys=[metric.value for metric in WandbMetrics]
                        ),
                    )
                    logger.info(
                        f"Found run {run_id} with {existing[1].shape[0]} samples, adding more..."
                    )
                    continue

                new_run = (
                    run_id,
                    run.history(
                        samples=1, pandas=(True), keys=[metric.value for metric in WandbMetrics]
                    ),
                )
                if new_run[1].shape[0] > 0:
                    filtered[run_id] = new_run
                    logger.info(
                        f"Found newer run instance for '{run_id}' for group {group}, added {filtered[run_id][1].shape} samples..."
                    )
                    continue
                else:
                    logger.warning(f"Run instance '{run_id}:{run.id}' has no samples, skipping...")

        except KeyError:
            raise KeyError("'{group}' experiment group not found!")

    filtered = list(filtered.values())
    logger.info(f"Found {len(filtered)} runs for group {group} to fit regression...")
    for run in filtered:
        logger.info(f"Sampled run: {run[0]} with shape: {run[1].shape}")

    # TODO: Get rid of this once we have real random configs
    ratios = np.random.uniform(size=len(filtered))
    run_ratios = []
    for idx, run in enumerate(filtered):
        run_ratios.append(
            {"run": run[0], "index": idx, "source1": ratios[idx], "source2": 1.0 - ratios[idx]}
        )

    run_metrics = [
        {"run": run[0], "index": idx, **_build_run_metrics(run[1])}
        for idx, run in enumerate(filtered)
    ]

    logger.info(run_metrics)
    config = pd.DataFrame(run_ratios)
    metrics = pd.DataFrame(run_metrics)

    X_train = config[config.columns[2:]].values
    Y_train = metrics[metrics.columns[2:]].values
    X_test = config[config.columns[2:]].values
    Y_test = metrics[metrics.columns[2:]].values

    np.random.seed(42)
    predictor = []

    indexed_metrics = list(enumerate(GroupedWandbMetrics.mmlu_bpb.value))

    for i, metric in indexed_metrics:
        target = Y_train[:, i]
        test_target = Y_test[:, i]

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
        logger.info(f"{i}: {metric.name} :: Correlation: {corr}")

        predictor.append(regression)

    for i, metric in indexed_metrics:
        _build_plot(Y_test, X_test, i, predictor, metric_name=metric.name)
        _simulate(
            index=i,
            predictor=predictor,
            df_config=config,
            # TODO: Grab the global distribution somehow here maybe from the config that was generated
            # right now this uses the first distributions as the base prior
            prior_distributions=config[config.columns[2:]].head(1).values[0],
            metric_name=metric.name,
        )


def _build_plot(
    Y_test: np.ndarray,
    X_test: np.ndarray,
    index: int,
    predictor: list[lgb.LGBMRegressor],
    metric_name: str,
):
    keys = {"pred": "Pred Loss", "true": "True Loss"}
    data = {keys["true"]: Y_test[:, index], keys["pred"]: predictor[index].predict(X_test)}
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

    plt.savefig(f"{OUTPUT_DIR}{metric_name}_fit.png", bbox_inches="tight")


def _build_run_metrics(history) -> dict[str, float]:
    df = pd.DataFrame(history)
    metrics = {}
    for metric in WandbMetrics:
        try:
            logger.info(df.loc[:, metric.value])
            metrics[metric.name] = df.loc[:, metric.value].tail(1).values[0]
        except KeyError:
            logger.warning(f"Metric {metric.value} not found in history, skipping...")
            continue

    return metrics


def _simulate(
    index: int,
    predictor: list[lgb.LGBMRegressor],
    prior_distributions: list[float],
    df_config: pd.DataFrame,
    metric_name: str,
    n_samples: int = 100000,
):
    np.random.seed(42)

    samples = np.random.dirichlet(prior_distributions * 1, n_samples)
    simulation = predictor[index].predict(samples)

    plt.hist(simulation, bins=32)

    plt.xlabel("Pred Loss")
    plt.ylabel("Frequency")

    k = 128
    top_k_samples = samples[np.argsort(simulation)[0:k]]
    top_k_samples.shape

    optimal_source_weights = np.mean(top_k_samples, axis=0)
    columns = df_config.columns[2:]

    df = pd.DataFrame(
        data=np.concatenate([np.array([prior_distributions]), top_k_samples], axis=0),
        columns=columns,
    )
    df = pd.melt(df)
    df["type"] = (["Original"] + ["Optimal"] * top_k_samples.shape[0]) * len(columns)
    df.info()

    logger.info("Original: ")
    logger.info(prior_distributions)
    print()
    logger.info("Optimal: ")
    logger.info(optimal_source_weights)

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
    ax.xaxis.set_tick_params(labelsize=14)
    ax.tick_params(axis="x", labelrotation=45)

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

    plt.savefig(f"{OUTPUT_DIR}{metric_name}_optimal.png", bbox_inches="tight", pad_inches=0.1)


if __name__ == "main":
    cli()
