import random

import click
import pandas as pd
import wandb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

from regmixer.eval.constants import Metrics

DEFAULT_WORKSPACE = "ai2-llm/regmixer"
HYPERPARAMETERS = {
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
    pass


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
                            samples=1000, pandas=(True), keys=[metric.value for metric in Metrics]
                        ),
                    )
                )
        except KeyError:
            raise KeyError("'{group}' experiment group not found!")

    filtered = filtered * 100
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

    config = pd.DataFrame(run_ratios)
    metrics = pd.DataFrame(run_metrics)

    X_train = config[config.columns[2:]].values
    Y_train = metrics[metrics.columns[2:]].values

    X_test = config[config.columns[2:]].values
    Y_test = metrics[metrics.columns[2:]].values

    np.random.seed(42)
    predictor = []

    for i, metric in enumerate(Metrics):
        target = Y_train[:, i]
        test_target = Y_test[:, i]

        gbm = lgb.LGBMRegressor(**HYPERPARAMETERS)

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
        print(i, metric.name, f"Correlation: {corr}")

        predictor.append(regression)

    metric_index = 0
    _build_plot(Y_test, X_test, metric_index, predictor)
    # TODO: Grab the global distribution somehow here
    _simulate(
        index=metric_index,
        predictor=predictor,
        df_config=config,
        prior_distributions=config[config.columns[2:]].head(1).values[0],
    )


def _build_plot(
    Y_test: np.ndarray, X_test: np.ndarray, index: int, predictor: list[lgb.LGBMRegressor]
):
    data = {"True Loss": Y_test[:, index], "Pred Loss": predictor[index].predict(X_test)}

    graph = sns.jointplot(
        data,
        x="Pred Loss",
        y="True Loss",
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

    r, _ = spearmanr(data["Pred Loss"], data["True Loss"])

    (phantom,) = graph.ax_joint.plot([], [], linestyle="", alpha=0)

    graph.ax_joint.legend(
        [phantom],
        [f"Correlation: {np.round(r * 100, decimals=2)}"],  # noqa
        edgecolor="black",
        fancybox=False,
        prop={
            "size": 24,
        },
        handlelength=-0.5,
    )

    graph.ax_joint.set_ylabel("True Loss", fontdict={"size": 32})
    graph.ax_joint.set_xlabel("Pred Loss", fontdict={"size": 32})

    graph.ax_marg_x.remove()
    graph.ax_marg_y.remove()

    graph.ax_joint.grid(True, ls="dashed")
    graph.ax_joint.spines[["right", "top"]].set_visible(True)

    plt.savefig("fit.pdf", bbox_inches="tight")


def _build_run_metrics(history) -> dict[str, float]:
    df = pd.DataFrame(history)
    metrics = {}
    for metric in Metrics:
        # TODO: Fix this once we have real metrics
        # metrics[metric.name] = df.loc[:, metric.value].tail(1).values[0]
        metrics[metric.name] = random.uniform(3, 8)

    return metrics


def _simulate(
    index: int,
    predictor: list[lgb.LGBMRegressor],
    prior_distributions: list[float],
    df_config: pd.DataFrame,
    n_samples: int = 100000,
):
    np.random.seed(42)
    print(prior_distributions)

    samples = np.random.dirichlet(prior_distributions * 1, n_samples)
    simulation = predictor[index].predict(samples)

    plt.hist(simulation, bins=32)

    plt.xlabel("Pred Loss")
    plt.ylabel("Frequency")

    k = 128
    top_k_samples = samples[np.argsort(simulation)[0:k]]
    top_k_samples.shape

    optimal_data_mixture = np.mean(top_k_samples, axis=0)
    print(optimal_data_mixture)

    columns = df_config.columns[2:]

    df = pd.DataFrame(
        data=np.concatenate([np.array([prior_distributions]), top_k_samples], axis=0),
        columns=columns,
    )
    df = pd.melt(df)
    df["type"] = (["Original"] + ["Optimal"] * top_k_samples.shape[0]) * len(columns)
    df.info()

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

    pal = {
        "Original": "#105257",
        "Optimal": "#F0529C",
    }
    sns.barplot(data=df, x="variable", y="value", hue="type", palette=pal)

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

    plt.savefig("optimal.pdf", bbox_inches="tight", pad_inches=0.1)


if __name__ == "main":
    cli()
