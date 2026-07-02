"""Local static grid-search backend for neural network sweeps.

It enumerates the same W&B-style ``sweep_configs`` grids that
``run_nn_model_experiments`` used, trains each config, and writes per-run metrics
to ``sweep_results/<experiment>.csv``. ``scripts/p2_propose.py`` reads that file
back instead of querying W&B run history.
"""

import itertools
import os

import pandas as pd
import yaml

from multievolve.utils.paths import get_output_root

# sweep_configs lives in the predictors package, next to this utils package.
SWEEP_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "predictors", "sweep_configs"
)

_CONFIG_MAP = {
    ("Fcn", "standard", "grid"): "fcn_standard_grid_sweep.yaml",
    ("Fcn", "custom", "grid"): "fcn_custom_grid_sweep.yaml",
    ("Fcn", "test", "test"): "fcn_test_sweep.yaml",
    ("Cnn", "standard", "grid"): "cnn_standard_grid_sweep.yaml",
    ("Cnn", "custom", "grid"): "cnn_custom_grid_sweep.yaml",
    ("Cnn", "test", "test"): "cnn_test_sweep.yaml",
}


def _grid_configs(sweep_config):
    """Cartesian product of a W&B-style parameter spec (``value`` / ``values``)."""
    params = sweep_config["parameters"]
    keys = list(params.keys())
    choices = [params[k]["values"] if "values" in params[k] else [params[k]["value"]] for k in keys]
    for combo in itertools.product(*choices):
        yield dict(zip(keys, combo))


def results_dir():
    return os.path.join(get_output_root(), "sweep_results")


def results_path(experiment_name):
    return os.path.join(results_dir(), f"{experiment_name}.csv")


def run_local_sweep(
    splits,
    features,
    models,
    experiment_name,
    use_cache=False,
    sweep_depth="standard",
    search_method="grid",
    count=10,
    show_plots=True,
):
    """Train every (split, feature, model, grid-config) and record metrics.

    The static backend supports ``grid`` and ``test``. W&B-backed Bayesian sweeps
    are intentionally unsupported in this offline workflow; ``count`` is kept
    only for API compatibility with the old function signature.
    """
    if search_method == "bayes":
        raise NotImplementedError(
            "Bayesian sweeps required W&B and are not supported by the local static backend."
        )

    rows = []
    for split in splits:
        for feature in features:
            for model in models:
                yaml_file = _CONFIG_MAP.get((model.__name__, sweep_depth, search_method))
                if yaml_file is None:
                    raise ValueError(
                        f"Invalid sweep configuration: model={model.__name__}, "
                        f"sweep_depth={sweep_depth}, search_method={search_method}."
                    )
                with open(os.path.join(SWEEP_CONFIG_DIR, yaml_file)) as f:
                    sweep_config = yaml.safe_load(f)
                for config in _grid_configs(sweep_config):
                    instance = model(
                        split, feature, use_cache=use_cache, config=config, show_plots=show_plots
                    )
                    test_stats = instance.run_model()  # stats_dict['test']
                    rows.append(
                        {
                            **config,
                            "Model": instance.model_name,
                            "Feature": feature.name,
                            "Split Method": instance.split_method,
                            "Test Loss": test_stats["MSE"],
                            "Spearman - Test": test_stats["Spearman r"],
                            "Pearson - Test": test_stats["Pearson r"],
                            "name": instance.file_attrs.get("model_name", ""),
                        }
                    )

    os.makedirs(results_dir(), exist_ok=True)
    pd.DataFrame(rows).to_csv(results_path(experiment_name), index=False)
    print(f"Wrote {len(rows)} sweep result(s) to {results_path(experiment_name)}")


def load_sweep_results(experiment_name):
    """Read the rows written by run_local_sweep."""
    path = results_path(experiment_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No sweep results at {path}; run training (p1_train) for "
            f"experiment '{experiment_name}' first."
        )
    return pd.read_csv(path)
