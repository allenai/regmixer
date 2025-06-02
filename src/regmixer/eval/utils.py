import json
import logging
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union, List
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from tqdm import tqdm
from wandb.apis.public import Run
import hashlib 
import os 
import yaml
import boto3

from cookbook.aliases import ExperimentConfig as CookbookExperimentConfig
from cookbook.utils.data import get_token_counts_and_ratios

from regmixer.synthesize_mixture import calculate_priors
from regmixer.eval.constants import WandbMetrics, GroupedWandbMetrics
from regmixer.eval.law import ScalingLaw
from regmixer.aliases import SourceConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_OUTPUT_DIR = "output/"
# Match regmix setup: https://github.com/sail-sg/regmix/blob/main/regression_fitting/regression.ipynb
LGBM_HPS = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l1", "l2"],
    "seed": 42,
    "num_iterations": 10000,
    "learning_rate": 1e-2,
    "verbosity": -1,
    "early_stopping_round": 3,
}

"""def mixing_law(x, param):
    log_c_i, b_i = param[0], param[1]
    t_i = param[2:]
    result = torch.exp(log_c_i) + torch.exp(b_i + torch.matmul(x[:, :-1], t_i))
    return result

def init_params_law(idx, num_domains=3):
    for log_c_i in np.linspace(-2, 1.5, 10):
        for b_i in np.linspace(-10, 1, 20):
            for _ in range(30):
                ts = [-np.random.rand() if i == idx else np.random.rand() * 0.1 for i in range(num_domains-1)]
                yield [log_c_i, b_i] + ts
"""


class Regressor():
    def fit(self, x, y, idx, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, x):
        if not hasattr(self, "model"):
            raise AttributeError("Subclasses must define self.model before calling predict()")
        return self.model.predict(x)


class LightGBMRegressor(Regressor):
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**LGBM_HPS)

    def fit(self, x, y, idx, **kwargs):
        target = y[:, idx]
        self.model = self.model.fit(
            x,
            target,
            eval_set=[(x, target)],
            eval_metric="l2",
        )
    

class LinearRegressor(Regressor):
    def __init__(self, **kwargs):
        self.model = LinearRegression()

    def fit(self, x, y, idx, **kwargs):
        target = y[:, idx]
        self.model = self.model.fit(x, target)

    
class LogLinearRegressor(Regressor):
    def __init__(self, params=None, **kwargs):
        np.random.seed(42)
        random.seed(42)

        if params is None:
            self.model = ScalingLaw(mixing_law)
        else:
            self.model = params

    def fit(self, x, y, idx, early_stopping=0.0, max_step=100, delta=0.02):
        target = y[:, idx]
        self.model = self.model.fit(
            x,
            target,
            init_params_log_linear_law(idx, num_domains = x.shape[-1]),
            max_step=max_step,
            delta=delta,
            eps=early_stopping
        )

    def predict(self, x):
        return mixing_law(torch.tensor(x, dtype=torch.float), torch.tensor(self.model, dtype=torch.float)).numpy()
    
class LogNonLinearRegressor(Regressor):
    def __init__(self, params=None, B_mask = None, **kwargs):
        np.random.seed(42)
        random.seed(42)

        self.B_mask = B_mask
        if params is None:
            self.model = ScalingLaw(nonlinear_mixing_law)
        else:
            self.model = params

    def fit(self, x, y, idx, early_stopping=0.0, max_step=100, delta=0.02):
        for i, params in enumerate(init_params_log_nonlinear_law(idx, self.B_mask, num_domains = x.shape[-1])):
            print(nonlinear_mixing_law(torch.tensor(x, dtype=torch.float), torch.tensor(params, dtype=torch.float), B_mask=self.B_mask))
            if i == 0:
                break
        
        target = y[:, idx]
        self.model = self.model.fit(
            x,
            target,
            init_params_log_nonlinear_law(idx, self.B_mask, num_domains = x.shape[-1]),
            B_mask=self.B_mask,
            max_step=max_step,
            delta=delta,
            eps=early_stopping
        )

    def predict(self, x):
        breakpoint()
        return nonlinear_mixing_law(torch.tensor(x, dtype=torch.float), torch.tensor(self.model, dtype=torch.float), B_mask=self.B_mask).numpy()


class SearchRegressor(Regressor):
    def __init__(self, **kwargs):
        pass 

    def fit(self, x, y, idx):
        target = y[:, idx]
        self.model = {tuple(row): target[i] for i, row in enumerate(x)}

    def predict(self, x):
        preds = []
        for row in x:
            if tuple(row) in self.model:
                preds.append(self.model[tuple(row)])
            else:
                preds.append(np.inf)
        return preds

        
    def get_searched_weights(self):
        return [np.array(weight) for weight, _ in self.model.items()]


def mixing_law(x, param, **kwargs):
    log_c_i = param[0]
    t_i = param[1:]
    result = torch.exp(log_c_i) + torch.exp(torch.matmul(x, t_i))
    return result

def nonlinear_mixing_law(x, param, B_mask=None):
    """
    Vectorized implementation of nonlinear mixing law:
    Y = exp(log_c + x @ t + sum_k B_k * x_i_k * x_j_k)

    Parameters:
        x:       (batch_size, d) input data
        param:   tensor of shape (1 + d + len(B_mask),)
                 [log_c, t (d,), B_params (len(B_mask),)]
        B_mask:  list of (i, j) tuples specifying quadratic terms

    Returns:
        Tensor of shape (batch_size,) with predicted outputs
    """
    log_c_i = param[0]
    d = x.shape[1]
    t_i = param[1:1 + d]                      # Linear weights len(domains)
    B_params = param[1 + d:]                  # Quadratic weights len(B_mask)

    lin_term = torch.matmul(x, t_i)           # (batch_size,)

    quad_term = None
    if B_mask is not None and len(B_mask) > 0:
        B_mask = torch.tensor(B_mask, dtype=torch.long, device=x.device)  # (num_terms, 2)
        x_i = x[:, B_mask[:, 0]]              # (batch_size, num_terms)
        x_j = x[:, B_mask[:, 1]]              # (batch_size, num_terms)
        quad_term = (x_i * x_j) @ B_params    # (batch_size,)

    if quad_term is None:
        return torch.exp(log_c_i) + torch.exp(lin_term)
    else:
        return torch.exp(log_c_i) + torch.exp(lin_term) + torch.exp(quad_term)

def init_params_log_linear_law(idx, num_domains=3):
    for log_c_i in np.linspace(-2, 1.5, 10): # originally (-2, 1.5, 10)
        for _ in range(30):
            ts = [-np.random.rand() if i == idx else np.random.rand() * 0.1 for i in range(num_domains)]
            yield [log_c_i] + ts

def init_params_log_nonlinear_law(idx, B_mask, num_domains=3):
    for log_c_i in np.linspace(-2, 1.5, 10): # originally (-2, 1.5, 10)
        for _ in range(30):
            lin_params = [-np.random.rand() if i == idx else np.random.rand() * 0.1 for i in range(num_domains)]
            quadratic_params = [np.random.rand() * 0.1 for i in range(len(B_mask))]
            yield [log_c_i] + lin_params + quadratic_params


REGRESSION_TYPES = {
    "lightgbm": LightGBMRegressor,
    "linear": LinearRegressor,
    "log_linear": LogLinearRegressor,
    "log_nonlinear": LogNonLinearRegressor,
    "search": SearchRegressor 
}




class Proposer():
    def __init__(self):
        pass

    def propose(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    

class SimulationProposer(Proposer):
    def propose(self,
        index: int,
        predictor: list[Regressor],
        prior_distributions: dict,
        num_samples: int = 1_000_000,
        seed: int = 1337,
        search_iterations: int = 10,
        opt_avg_metric: bool = False,
        constrain_objective: bool = False,
        final_cookbook_path: Optional[Path] = None,
        manual_token_constraint_path: Optional[Path] = None,
        repetition_factor: float = 1.0,
        obj_weights: Optional[list] = None,
        temperature: Optional[float] = None,
        reference_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        min_weight = 1e-5
        min_dirichlet = 1
        max_dirichlet = 100
        search_dirichlet_factor = 2.0

        search_prior = np.array(list(prior_distributions.values()))

        
        if temperature is not None:
            search_prior = search_prior**temperature
            search_prior = search_prior / np.sum(search_prior)


        best_weights = np.zeros(len(prior_distributions))

        if constrain_objective:
            # just need a desired token count and available token count 
            assert final_cookbook_path is not None or manual_token_constraint_path is not None
            if final_cookbook_path is not None:
                with open(final_cookbook_path, "r") as f:
                    data = yaml.safe_load(f)

                final_config = CookbookExperimentConfig(**data, path=final_cookbook_path)
                desired_tokens = final_config.max_tokens

                token_universe = get_token_counts_and_ratios(
                    final_config.dataset.sources, final_config.dataset.dtype, True
                )
                available_tokens_per_source = {path: relative_size * token_universe[1] for path, relative_size in token_universe[0].items()}
                # ensures that order of sources in simulations and in the constraint dictionary are aligned
                available_tokens_per_source = {source: available_tokens_per_source[source] for source, _ in prior_distributions.items()}
            elif manual_token_constraint_path is not None:
                with open(manual_token_constraint_path, "r") as f:
                    data = yaml.safe_load(f)
                desired_tokens = data['requested_tokens']
                available_tokens_per_source = {source: data['available_tokens'][source] for source, _ in prior_distributions.items()}

        # Multi-step search leveraging iterative prior results
        for search_step in tqdm(
            range(search_iterations), desc=f"Searching in {num_samples} candidate samples"
        ):
            offset = np.log(search_dirichlet_factor * (search_step + 1))
            alphas = np.exp(
                np.random.uniform(
                    low=np.log(min_dirichlet) + offset,
                    high=np.log(max_dirichlet) + offset,
                    size=num_samples,
                )
            )

            # generate simulations by sampling from dirichlet distribution with parameter prior * alpha 
            simulations = (
                torch.distributions.Dirichlet(torch.from_numpy(alphas[:, None] * search_prior))
                .sample()
                .numpy()
            )

            # Filter out invalid simulations from the population
            if temperature is None:
                # keep this for reproducibility...
                simulations = simulations[np.all(simulations <= 6.5 * np.array(list(prior_distributions.values())), axis=1)]

            if constrain_objective:
                original_simulation_size = len(simulations)
                simulations = simulations[((simulations*desired_tokens) <= np.array(list(available_tokens_per_source.values()))*repetition_factor).all(axis=1)]
                logger.info(f"Removed {original_simulation_size - len(simulations)} out of {original_simulation_size} simulations that would repeat tokens at the final run scale.")

                if len(simulations) == 0:
                    continue 

            predictions = np.array([reg.predict(simulations) for reg in predictor])
            if reference_scores is not None:
                # If reference scores are provided, filter simulations based on them
                tol_range = [0, 0.05, 0.1, 0.15, 0.2]
                for tol in tol_range:
                    # we allow for predicted scores to be within a tolerance of the reference scores
                    # if the current tol results in no remaining simulations, we increase tol 
                    pareto_idxs = np.where(np.all(predictions.T < reference_scores+tol, axis=1))[0]
                    if len(pareto_idxs) != 0:
                        logger.info(f"Using eps={tol} for enforcing pareto improvements")
                        break 

                if len(pareto_idxs) == 0:
                    logger.info(f"No simulations passed the pareto filter.")
                    continue

                # filter both the simulations and corresponding predictions down 
                simulations = simulations[pareto_idxs]
                predictions = predictions[:, pareto_idxs]  
                logger.info(f"Filtered simulations to {len(simulations)} based on reference scores.")

            if opt_avg_metric:
                if obj_weights is not None:
                    objs = np.average(predictions, axis=0, weights=obj_weights)
                    logger.info(f"Computing weighted average of predictions using {obj_weights}")
                else:
                    objs = predictions.mean(axis=0)
            else:
                objs = predictor[index].predict(simulations)

            # Take the best loss prediction as an index unless it's greater than 1e-3
            print(objs.min())
            best_mask = (objs - objs.min()) < 1e-3
            best_weights = simulations[best_mask].mean(0)

            # Zero out weights below min_weight threshold and normalize
            best_weights[best_weights < min_weight] = 0.0
            best_weights /= best_weights.sum()

            search_prior = (best_weights + search_prior) / 2


        if constrain_objective and not all(best_weights * desired_tokens <= np.array(list(available_tokens_per_source.values()))* repetition_factor):
            raise ValueError(f"Best weights are out of bounds!")

        return best_weights



class SearchProposer(Proposer):
    def propose(self,
        index:int,
        predictor: list[SearchRegressor],
        opt_avg_metric: bool = False,
        **kwargs
    ):
        searched_weights = predictor[0].get_searched_weights()
        best_performance = np.inf 
        best_weights = np.zeros(len(searched_weights[0]))
        for weight in searched_weights:
            if opt_avg_metric:
                pred = np.array([reg.predict(weight[None]) for reg in predictor]).mean(axis=0)[0]
            else:
                pred = predictor[index].predict(weight[None])[0]
            if pred < best_performance:
                best_performance = pred
                best_weights = weight

        return best_weights


PROPOSER_TYPES = {
    "simulation": SimulationProposer,
    "search": SearchProposer,
}


@dataclass
class RunInstance:
    id: str
    display_name: str
    config: dict
    samples: pd.DataFrame
    state: str 

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "config": self.config,
            "samples": self.samples.to_dict(),
            "state": self.state
        }


def get_output_dir(groups: list[str]) -> str:
    return f"{BASE_OUTPUT_DIR}{'_'.join(groups)}/"


def build_regression(
    idx: int,
    Y_train: np.ndarray,
    X_train: np.ndarray,
    regression_type: str,
    early_stopping: float,
    B_mask: Optional[List[Tuple[int, int]]] = None
) -> Regressor:
    logger.info(f"Building regression model, index: {idx}")
    reg = REGRESSION_TYPES[regression_type](B_mask=B_mask)
    reg.fit(X_train, Y_train, idx, early_stopping=early_stopping)
    return reg


def get_runs_from_api(
    api, workspace: str, groups: list[str], cache_path: Path, no_cache: bool, num_samples: int, eval_metric_group: GroupedWandbMetrics
) -> list[RunInstance]:

    wandb_runs = []
    for group in groups:
        wandb_runs.extend(
            api.runs(
                path=workspace,
                #filters={"display_name": {"$regex": f"^(?!.*larger).*{group}.*$"}},
                filters={"display_name": {"$regex": f".*{group}.*"}},
            )
        )

    # NOTE: Last writer wins and should be sorted created_at desc
    memo = {}
    for run in wandb_runs:
        memo[run.display_name] = run

    all_runs = []
    for run in memo.values():
        if run.state == "crashed":
            logger.warning(f"Run {run.display_name} has crashed; still using its final result")

        if run.state == "running":
            logger.warning(f"Run {run.display_name} is still running; skipping")
            continue 
        
        if run is not None:
            all_runs.append(mk_run_history(run, num_samples, eval_metric_group))

    all_runs = sorted(all_runs, key=lambda run: run.display_name.lower())
    if not no_cache:
        with open(cache_path, "w") as f:
            json.dump([run.as_dict() for run in all_runs], f)

    return all_runs


def mk_run_history(run: Run, samples: int, eval_metric_group: GroupedWandbMetrics) -> Any:
    if samples == 1:
        try:
            summary = [{metric: run.summary[metric] for metric in eval_metric_group.value if metric in run.summary}]
        except KeyError:
            print(run.id)
            print(run.summary.keys())

            breakpoint()
        return mk_run_instance(
            run, 
            summary,
            samples
        )
    else:
        return mk_run_instance(
            run,
            run.scan_history(keys=eval_metric_group.value),
            samples
        )


def mk_run_from_json(run: dict) -> RunInstance:
    return RunInstance(
        id=run["id"],
        display_name=run["display_name"],
        config=run["config"],
        samples=pd.DataFrame(run["samples"]),
        state=run["state"]
    )


def mk_run_instance(run: Run, history: list[Any], n_samples: int) -> RunInstance:
    samples = pd.DataFrame.from_records(history).tail(n_samples)
    #logger.info(
    #    f"Collected RunInstance for {run.display_name}:{run.id} with samples: {samples.shape}"
    #)
    return RunInstance(
        id=run.id,
        display_name=run.display_name,
        config=run.config,
        samples=samples,
        state=run.state
    )


def compute_mixture_neighborhood(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    ratios: pd.DataFrame,
    neighborhood: str,
    train_split: float
)-> tuple[np.ndarray, np.ndarray]:    
    neighborhood_size = int(train_split * len(X_train))
    logger.info(f"Computing neighborhood of size {neighborhood_size} for {neighborhood}")
    centroid = ratios[ratios.name == neighborhood][ratios.columns[3:]].values[0]
    distances = np.linalg.norm(X_train - centroid, axis=1)
    # Get indices of k smallest distances
    nearest_indices = np.argsort(distances)[:neighborhood_size]
    X_train = X_train[nearest_indices]
    Y_train = Y_train[nearest_indices]
    return X_train, Y_train

def plot_simulations(
    prior_distributions: np.ndarray,
    samples,
    columns: list[str],
    metric_name: str,
    regression_type: str,
    train_split: float,
    n_test: int,
    split_seed: int,
    n_samples: int,
    alpha: float,
    output_dir: str = BASE_OUTPUT_DIR,
):
    plt.close()
    df = pd.DataFrame(
        data=np.concatenate([np.array([prior_distributions]), samples], axis=0),
        columns=columns,
    )
    df["sample"] = df.index
    melted_df = df.melt(id_vars=["sample"], var_name="Domain", value_name="Weight")
    g = sns.FacetGrid(melted_df, col="sample", col_wrap=4, aspect=2)
    g.map_dataframe(sns.barplot, x="Domain", y="Weight", palette="viridis", hue="Domain")
    g.set(ylim=(0, 0.75))
    g.set_axis_labels("Domain", "Weight")

    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
        ax.yaxis.grid(True, linestyle="--", which="both", color="gray", alpha=0.7)

    plt.savefig(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha)}_sim_grid.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


def plot_correlation(
    Y_test: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    X_train: np.ndarray,
    index: int,
    predictors: list[Regressor],
    train_split: float,
    n_test: int,
    split_seed: int,
    n_samples: int,
    metric_name: str,
    regression_type: str,
    alpha: Optional[float] = None,
    output_dir: str = BASE_OUTPUT_DIR,
):
    plt.close()

    y_pred_train = predictors[index].predict(X_train)
    y_true_train = Y_train[:, index]


    corr_results = {}

    if train_split == 1 and n_test==0:
        # Only plot train if train and test are the same
        sns.regplot(
            x=y_pred_train,
            y=y_true_train,
            scatter_kws={"s": 64, "color": "#105257"},
            line_kws={"color": "#F0529C", "linewidth": 3, "linestyle": "dashed"},
            label="Train"
        )

        corr_train = np.corrcoef(y_pred_train, y_true_train)[0, 1]
        plt.legend(
            title=f"{metric_name} correlation",
            labels=[f"Train: {np.round(corr_train * 100, 2)}"],
            fontsize=14,
            title_fontsize=16,
        )

        corr_results['train'] = corr_train
    else:
        # Predict test
        y_pred_test = predictors[index].predict(X_test)
        y_true_test = Y_test[:, index]

        # Plot test
        sns.regplot(
            x=y_pred_test,
            y=y_true_test,
            scatter_kws={"s": 64, "color": "#105257"},
            line_kws={"color": "#F0529C", "linewidth": 3, "linestyle": "dashed"},
            label="Test"
        )

        # Plot train
        sns.regplot(
            x=y_pred_train,
            y=y_true_train,
            scatter_kws={"s": 64, "color": "#B0C4DE"},
            line_kws={"color": "#8888FF", "linewidth": 3, "linestyle": "dotted"},
            label="Train"
        )

        corr_test = np.corrcoef(y_pred_test, y_true_test)[0, 1]
        corr_train = np.corrcoef(y_pred_train, y_true_train)[0, 1]

        import matplotlib.patches as mpatches

        test_dot = mpatches.Patch(color="#105257", label=f"Test: {np.round(corr_test * 100, 2)}")
        train_dot = mpatches.Patch(color="#B0C4DE", label=f"Train: {np.round(corr_train * 100, 2)}")

        plt.legend(
            handles=[test_dot, train_dot],
            title=f"{metric_name} correlations",
            fontsize=14,
            title_fontsize=16,
        )

        corr_results['train'] = corr_train
        corr_results['test'] = corr_test

    # Common plot settings
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.grid(True, linestyle="dashed")
    plt.tight_layout()

    # Save figure
    plt.savefig(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha)}_fit.png"
    )
    plt.close()

    with open(f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha=alpha)}_correlations.json", "w") as f:
        f.write(json.dumps(corr_results))


def plot_interaction_matrix(
    output_dir: str, 
    predictors: list[Regressor], 
    regression_type: str, 
    domain_names: list[str],
    metric_names: list[str]
):
    metric_names = [metric.split("/")[-1].split(" ")[0] for metric in metric_names]
    interaction_matrix = np.zeros((len(metric_names), len(domain_names)))
    for i, predictor in enumerate(predictors):
        if regression_type == 'lightgbm':
            interaction_matrix[i] = predictor.model.feature_importances_
        elif regression_type == 'log_linear':
            interaction_matrix[i] = predictor.model[1:]

    plt.figure(figsize=(10, 8))
    plt.imshow(interaction_matrix, cmap='rainbow', aspect='auto')
    plt.colorbar(label='Influence')

    plt.xticks(ticks=np.arange(len(domain_names)), labels=domain_names, rotation=90)
    plt.yticks(ticks=np.arange(len(metric_names)), labels=metric_names)
    plt.title(f"Interaction matrix for {regression_type}")
    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/interaction_matrix.png",
        bbox_inches="tight",
    )
    plt.close()
    np.save(f"{output_dir}/interaction_matrix.npy", interaction_matrix)

def mk_run_metrics(
    history,
    samples: int,
    metrics: Tuple[str, list[str]],
    display_name: str,
    average: bool = False,
) -> dict[str, float]:
    df = pd.DataFrame(history)
    results = {}
    group_name, group_metrics = metrics
    in_loop_tasks = [task for task in df.columns if task in group_metrics]
    offline_tasks = [task for task in group_metrics if task not in in_loop_tasks]
    if average:
        raise NotImplementedError("Averaging the task is implemented but out of date!")
        result = np.mean(
            [df.loc[:, metric_name].tail(samples).mean() for metric_name in group_metrics]
        )

        results[group_name] = result
    else:
        for metric_name in in_loop_tasks:
            results[metric_name] = df.loc[:, metric_name].tail(samples).mean()

        if len(offline_tasks) > 0:
            # need to obtain offline results
            offline_results = get_offline_evals(display_name, offline_tasks)
            results.update(offline_results)

    return results


def get_offline_evals(display_name, tasks, dashboard="regmixer"):
    bucket = 'ai2-llm'
    prefix = f'evaluation/{dashboard}/{display_name}'

    s3 = boto3.client('s3')

    # Get list of all files under the prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    json_files = []
    jsonl_files = []

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('/'):
                if key.endswith('.jsonl'):
                    jsonl_files.append(key)
                elif key.endswith('.json'):
                    json_files.append(key)

    all_jsonl_data = []

    for key in jsonl_files:
        if key.endswith("metrics-all.jsonl"):
            obj = s3.get_object(Bucket=bucket, Key=key)
            for line in obj['Body'].iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        all_jsonl_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON line in {key}: {e}")
            
    offline_results = {}
    for task in tasks:
        task_data = [
            data for data in all_jsonl_data
            if (
                data['task_config'].get('metadata', {}).get('alias') == task
                or data['task_name'] == task
            )
        ]
        if len(task_data) == 0:
            raise ValueError(f"Task {task} not found in JSONL data for {display_name}")
        data = task_data[0]
        if 'bits_per_byte_corr' in data['metrics']:
            offline_results[data['task_config']['metadata']['alias']] = data['metrics']['bits_per_byte_corr']
        elif 'bits_per_byte_corr_macro' in data['metrics']:
            offline_results[data['task_config']['metadata']['alias']] = data['metrics']['bits_per_byte_corr_macro']
        elif 'bits_per_byte' in data['metrics']:
            offline_results[data['task_name']] = data['metrics']['bits_per_byte'] 
        else:
            raise ValueError(f"{data['task_name']} does not have bits_per_byte_corr or bits_per_byte_corr_macro in metrics {data['metrics'].keys()}.")
            
    return offline_results



def mk_weights_from_config(config: dict, priors: tuple) -> dict[str, float]:
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


def solve_log_linear(
    predictor: list[Regressor],
    prior_distributions: np.ndarray,
    df_config: pd.DataFrame,
    metric_name: str,
    regression_type: str,
    train_split: float,
    n_test: int,
    split_seed: int,
    n_samples: int, 
    alpha: float = 1.0,
    output_dir: str = BASE_OUTPUT_DIR,
    seed: int = 1337,
) -> np.ndarray:

    torch.manual_seed(seed)

    # Split params into biases (b) and t values
    t = [p[2:] for p in predictor]
    b = [p[1] for p in predictor]

    t = torch.tensor(t, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    # Initialize weights as a probability vector
    weights = torch.rand(len(t[0])-1, requires_grad=True)
    assert weights.sum() <= 1
    #weights = torch.nn.Parameter(raw_weights / raw_weights.sum())

    def objective(weights):
        return torch.sum(torch.exp(b + t @ weights))

    def project(w):
        """
        Projects a vector w (length n-1) so that:
        - Each entry is in [0, 1]
        - Sum of entries <= 1
        Returns the full probability vector (length n), where the last element is 1 - sum(w)
        """
        w.data.clamp_(0, 1)  # clamp each entry between 0 and 1

        total = w.sum()
        if total > 1:
            w.data.mul_(1 / total)  # rescale to make sum <= 1

        last = 1.0 - w.sum()
        last = torch.clamp(last, min=0.0, max=1.0)  # ensure numerical safety

        return torch.cat([w, last.unsqueeze(0)], dim=0)  # final full probability vector

    # Optimization
    optimizer = optim.Adam([weights], lr=0.001)
    n_iterations = 1000
    for i in range(n_iterations):
        optimizer.zero_grad()
        loss = objective(weights)
        loss.backward()
        optimizer.step()

        # Project onto the probability simplex
        with torch.no_grad():
            weights.data = project(weights.data)

        if (i + 1) % 100 == 0:
            print(f'Iteration {i+1}/{n_iterations}, Loss: {loss.item():.4f}')

    best_weights = weights.detach().cpu().numpy()

    plot_and_log_weights(
        prior=prior_distributions,
        prediction=best_weights,
        metric_name=metric_name,
        regression_type=regression_type,
        train_split=train_split,
        n_test=n_test,
        split_seed=split_seed,
        n_samples=n_samples,
        alpha=alpha,
        df_config=df_config,
        output_dir=output_dir,
    )

    return best_weights



def simulate2(
    index: int,
    predictor: list[Regressor],
    prior_distributions: np.ndarray,
    df_config: pd.DataFrame,
    metric_name: str,
    regression_type: str,
    train_split: float,
    n_test: int,
    split_seed: int,
    n_samples: int, 
    num_samples: int = 1_000_000,
    alpha: float = 1.0,
    output_dir: str = BASE_OUTPUT_DIR,
    seed: int = 1337,
    search_iterations: int = 10,
) -> np.ndarray:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


    def predict_average(weights, regressors) -> np.ndarray:
        if regression_type in ["linear", "lightgbm"]:
            return np.array([regressor.predict(weights) for regressor in regressors]).mean(axis=0)
        elif regression_type == "log_linear":
            return np.array([mixing_law(torch.tensor(weights, dtype=torch.float), torch.tensor(p, dtype=torch.float)).numpy() for p in regressors]).mean(axis=0)
        else:
            raise NotImplementedError(f"Regression type {regression_type} not supported.")

    min_weight = 1e-5
    min_dirichlet = 1
    max_dirichlet = 100
    search_dirichlet_factor = 2.0

    search_prior = prior_distributions
    best_weights = np.zeros(len(prior_distributions))

    # Multi-step search leveraging iterative prior results
    for search_step in tqdm(
        range(search_iterations), desc=f"Searching in {num_samples} candidate samples"
    ):
        offset = np.log(search_dirichlet_factor * (search_step + 1))
        alphas = np.exp(
            np.random.uniform(
                low=np.log(min_dirichlet) + offset,
                high=np.log(max_dirichlet) + offset,
                size=num_samples,
            )
        )

        # generate simulations by sampling from dirichlet distribution with parameter prior * alpha 
        simulations = (
            torch.distributions.Dirichlet(torch.from_numpy(alphas[:, None] * search_prior))
            .sample()
            .numpy()
        )

        # Filter out invalid simulations from the population
        simulations = simulations[np.all(simulations <= 6.5 * prior_distributions, axis=1)]

        if index != -1:
            preds = predictor[index].predict(simulations)
        else:
            preds = predict_average(
                weights=simulations,
                regressors=predictor
            )

        # Take the best loss prediction as an index unless it's greater than 1e-3
        print(preds.min())
        best_mask = (preds - preds.min()) < 1e-3
        best_weights = simulations[best_mask].mean(0)

        # Zero out weights below min_weight threshold and normalize
        best_weights[best_weights < min_weight] = 0.0
        best_weights /= best_weights.sum()

        search_prior = (best_weights + search_prior) / 2

    if not type(best_weights) == np.ndarray:
        raise ValueError(f"Simulation must be of type np.ndarray, got {type(best_weights)}")

    plot_and_log_weights(
        prior=prior_distributions,
        prediction=best_weights,
        metric_name=metric_name,
        regression_type=regression_type,
        train_split=train_split,
        n_test=n_test,
        split_seed=split_seed,
        n_samples=n_samples,
        alpha=alpha,
        df_config=df_config,
        output_dir=output_dir,
    )

    return best_weights






def plot_and_log_weights(
    prior: np.ndarray,
    prediction: np.ndarray,
    metric_name: str,
    regression_type: str,
    train_split: float,
    n_test: int,
    split_seed: int,
    n_samples: int,
    alpha: float,
    df_config: pd.DataFrame,
    output_dir: str = BASE_OUTPUT_DIR,
):
    
    logger.info(f":::::::::{metric_name}:::::::::")
    logger.info("Predicted optimal weights:")

    columns = df_config.columns[3:].to_list()
    with open(f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha=alpha)}_optimal.json", "w") as f:
        out = [
            {"domain": columns[idx], "weight": weight} for idx, weight in enumerate(prediction)
        ]
        logger.info(out)
        f.write(json.dumps(out))


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
    df["type"] = (["Corpus"] + ["Optimal"]) * len(columns)

    plt.rc("axes", unicode_minus=False)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 16,
        }
    )

    _, ax = plt.subplots(figsize=(12, 10), layout="compressed")
    ax.ticklabel_format(useMathText=True)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.tick_params(axis="x", labelrotation=90)

    pallette = {
        "Corpus": "#105257",
        "Optimal": "#F0529C",
    }

    df_sorted = df[df["type"] == "Corpus"].sort_values(by="value", ascending=False)
    df["variable"] = pd.Categorical(df["variable"], categories=df_sorted["variable"], ordered=True)
    sns.barplot(data=df, x="variable", y="value", hue="type", palette=pallette, ax=ax)

    ax.legend(
        edgecolor="black",
        fancybox=False,
        prop={
            "size": 18,
        },
        handlelength=0.4,
        ncol=3,
    )

    ax.yaxis.grid(True, linestyle="--", which="both", color="gray", alpha=0.7)
    ax.set_ylim(0, 0.4)

    ax.set_xlabel(
        "Domain",
        fontdict={
            "size": 26,
        },
    )
    ax.set_ylabel(
        "Weight",
        fontdict={
            "size": 26,
        },
    )

    plt.savefig(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha=alpha)}_optimal.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )


def mk_output_prefix(output_dir: str, metric: str, regression_type: str, train_split: float, n_test: int, split_seed: int, n_samples: int, alpha: Optional[float] = None) -> str:
    def sanitize(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)

    return os.path.join(output_dir, sanitize(metric)) + (
        f"_alpha_{str(alpha).replace('.', '_')}" if alpha and alpha != 1.0 else ""
     ) + (
        f"_{regression_type}_reg" if regression_type != "lightgbm" else ""
     ) + (
        f"_trainsplit_{train_split}" if train_split != 1.0 else "" 
     ) + (
        f"_ntest_{n_test}" if n_test != 0 else ""
     ) + (
        f"_seed_{split_seed}" if split_seed != 0 else ""
     ) + (
         f"_{n_samples}_samples" if n_samples != 10 else ""
     )


def save_eval_config(eval_config: dict, output_dir: str) -> str:
    # Serialize dict in a stable way
    config_str = json.dumps(eval_config, sort_keys=True)
    
    # Hash it (short hash for readability)
    hash_str = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
    
    # Create directory
    folder_path = os.path.join(output_dir, hash_str)
    os.makedirs(folder_path, exist_ok=True)

    # Save config JSON inside
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(eval_config, f, indent=2)

    print(f"[INFO] Saved config to {config_path}")
    return folder_path



def filter_constrained_swarm(
    final_cookbook_path: Path,
    run_ratios: List,
    run_metrics: List
) -> Tuple[List, List]:
    assert final_cookbook_path is not None, "final_cookbook_path must be set to determine how to construct swarm to be unconstrained."

    with open(final_cookbook_path, "r") as f:
        data = yaml.safe_load(f)

    final_config = CookbookExperimentConfig(**data, path=final_cookbook_path)
    desired_tokens = final_config.max_tokens


    #logger.warning(f"Using hardcoded token counts!")
    #with open("cache/priors_cache_217af510306ab626a507634c64ca7ca8.json", "r") as f:
    #    available_tokens_per_source = json.load(f)['token_counts']

    token_universe = get_token_counts_and_ratios(
        final_config.dataset.sources, final_config.dataset.dtype, True
    )
    available_tokens_per_source = {path: relative_size * token_universe[1] for path, relative_size in token_universe[0].items()}

    original_swarm_size = len(run_ratios)

    valid_runs = [
        run['run'] for run in run_ratios if
        all([
            run[source] * desired_tokens <= num_available_tokens for source, num_available_tokens in available_tokens_per_source.items()
        ])
    ]

    run_ratios = [run for run in run_ratios if run['run'] in valid_runs]
    run_metrics = [run for run in run_metrics if run['run'] in valid_runs]


    logger.info(f"Removed {original_swarm_size - len(run_ratios)} swarm runs that would repeat tokens at the final run scale.")

    return run_ratios, run_metrics


def calculate_priors_with_manual(
    source_configs: list[SourceConfig],
    dtype,
    use_cache: bool,
    manual_prior: Optional[dict[str, float]] = None,
):

    priors = calculate_priors(
        source_configs=source_configs,
        dtype=dtype,
        use_cache=use_cache,
    )

    if manual_prior is not None:
        logger.info(f"Adjusting priors with manual prior weights: {manual_prior}")
        for source_config in sorted(source_configs, key=lambda x: x.name):
            if source_config.topics:
                # adjust each topic weight by the manual prior  
                weights = np.array([priors[0][f"{source_config.name}:{topic.name}"] for topic in source_config.topics])
                normalized_weights = weights / weights.sum()
                if manual_prior is not None and source_config.name in manual_prior:
                    for i, topic in enumerate(source_config.topics):
                        priors[0][f"{source_config.name}:{topic.name}"] = manual_prior[source_config.name] * normalized_weights[i]
            else:
                # directly overwrite source weight with manual prior
                if manual_prior is not None and source_config.name in manual_prior:
                    priors[0][source_config.name] = manual_prior[source_config.name]

    return priors 


def aggregate_mmlu(metrics: pd.DataFrame, metrics_to_index: list):
    logger.info("Aggregating MMLU metrics...")

    def add_weighted_dot_column(df: pd.DataFrame, weights: dict, output_col: str, metrics_to_index: list):
        weight_series = pd.Series(weights)
        df[output_col] = df[weight_series.index].dot(weight_series)
        df.drop(columns=weight_series.index, inplace=True)
        columns_to_remove = set(weight_series.index)
        metrics_to_index = [col for col in metrics_to_index if col not in columns_to_remove]
        metrics_to_index.append(output_col)
        return metrics_to_index

    stem_weights = {'mmlu_abstract_algebra:rc::olmes': 0.03313452617627568, 'mmlu_astronomy:rc::olmes': 0.05036447978793903, 'mmlu_college_biology:rc::olmes': 0.04771371769383698, 'mmlu_college_chemistry:rc::olmes': 0.03313452617627568, 'mmlu_college_computer_science:rc::olmes': 0.03313452617627568, 'mmlu_college_mathematics:rc::olmes': 0.03313452617627568, 'mmlu_college_physics:rc::olmes': 0.033797216699801194, 'mmlu_computer_security:rc::olmes': 0.03313452617627568, 'mmlu_conceptual_physics:rc::olmes': 0.07786613651424784, 'mmlu_electrical_engineering:rc::olmes': 0.04804506295559974, 'mmlu_elementary_mathematics:rc::olmes': 0.12524850894632206, 'mmlu_high_school_biology:rc::olmes': 0.10271703114645461, 'mmlu_high_school_chemistry:rc::olmes': 0.06726308813783963, 'mmlu_high_school_computer_science:rc::olmes': 0.03313452617627568, 'mmlu_high_school_mathematics:rc::olmes': 0.08946322067594434, 'mmlu_high_school_physics:rc::olmes': 0.050033134526176276, 'mmlu_high_school_statistics:rc::olmes': 0.07157057654075547, 'mmlu_machine_learning:rc::olmes': 0.03711066931742876}
    other_weights = {'mmlu_anatomy:rc::olmes': 0.04164096236890808, 'mmlu_business_ethics:rc::olmes': 0.030845157310302282, 'mmlu_clinical_knowledge:rc::olmes': 0.08173966687230105, 'mmlu_college_medicine:rc::olmes': 0.05336212214682295, 'mmlu_global_facts:rc::olmes': 0.030845157310302282, 'mmlu_human_aging:rc::olmes': 0.06878470080197409, 'mmlu_management:rc::olmes': 0.03177051202961135, 'mmlu_marketing:rc::olmes': 0.07217766810610735, 'mmlu_medical_genetics:rc::olmes': 0.030845157310302282, 'mmlu_miscellaneous:rc::olmes': 0.24151758173966686, 'mmlu_nutrition:rc::olmes': 0.09438618136952498, 'mmlu_professional_accounting:rc::olmes': 0.08698334361505243, 'mmlu_professional_medicine:rc::olmes': 0.08389882788402221, 'mmlu_virology:rc::olmes': 0.05120296113510179}
    social_sciences_weights = {'mmlu_econometrics:rc::olmes': 0.03704907377315567, 'mmlu_high_school_geography:rc::olmes': 0.06434839129021774, 'mmlu_high_school_government_and_politics:rc::olmes': 0.06272343191420214, 'mmlu_high_school_macroeconomics:rc::olmes': 0.12674683132921677, 'mmlu_high_school_microeconomics:rc::olmes': 0.07734806629834254, 'mmlu_high_school_psychology:rc::olmes': 0.17712057198570036, 'mmlu_human_sexuality:rc::olmes': 0.04257393565160871, 'mmlu_professional_psychology:rc::olmes': 0.19889502762430938, 'mmlu_public_relations:rc::olmes': 0.03574910627234319, 'mmlu_security_studies:rc::olmes': 0.07962300942476438, 'mmlu_sociology:rc::olmes': 0.0653233669158271, 'mmlu_us_foreign_policy:rc::olmes': 0.032499187520311994}
    humanities_weights = {'mmlu_formal_logic:rc::olmes': 0.026780021253985122, 'mmlu_high_school_european_history:rc::olmes': 0.03506907545164718, 'mmlu_high_school_us_history:rc::olmes': 0.04335812964930925, 'mmlu_high_school_world_history:rc::olmes': 0.050371944739638685, 'mmlu_international_law:rc::olmes': 0.0257173219978746, 'mmlu_jurisprudence:rc::olmes': 0.022954303931987247, 'mmlu_logical_fallacies:rc::olmes': 0.034643995749202974, 'mmlu_moral_disputes:rc::olmes': 0.07353878852284804, 'mmlu_moral_scenarios:rc::olmes': 0.1902231668437832, 'mmlu_philosophy:rc::olmes': 0.06609989373007438, 'mmlu_prehistory:rc::olmes': 0.06886291179596174, 'mmlu_professional_law:rc::olmes': 0.32603613177470775, 'mmlu_world_religions:rc::olmes': 0.03634431455897981}

    metrics_to_index = add_weighted_dot_column(metrics, stem_weights, "mmlu_stem", metrics_to_index)
    metrics_to_index = add_weighted_dot_column(metrics, other_weights, "mmlu_other", metrics_to_index)
    metrics_to_index = add_weighted_dot_column(metrics, social_sciences_weights, "mmlu_social_sciences", metrics_to_index)
    metrics_to_index = add_weighted_dot_column(metrics, humanities_weights, "mmlu_humanities", metrics_to_index)

    return metrics, metrics_to_index