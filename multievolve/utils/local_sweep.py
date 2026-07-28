"""Content-addressed local grid search with atomic per-job checkpoints."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from multievolve.utils.paths import get_output_root
from multievolve.utils.reproducibility import atomic_write_json, seed_everything, sha256_json, stable_seed

ARTIFACT_SCHEMA_VERSION = 3
SWEEP_CONFIG_DIR = Path(__file__).resolve().parents[1] / "predictors" / "sweep_configs"

_CONFIG_MAP = {
    ("Fcn", "standard", "grid"): "fcn_standard_grid_sweep.yaml",
    ("Fcn", "custom", "grid"): "fcn_custom_grid_sweep.yaml",
    ("Fcn", "test", "test"): "fcn_test_sweep.yaml",
    ("Cnn", "standard", "grid"): "cnn_standard_grid_sweep.yaml",
    ("Cnn", "custom", "grid"): "cnn_custom_grid_sweep.yaml",
    ("Cnn", "test", "test"): "cnn_test_sweep.yaml",
}


def _grid_configs(sweep_config):
    params = sweep_config["parameters"]
    keys = list(params)
    choices = [
        params[key]["values"] if "values" in params[key] else [params[key]["value"]]
        for key in keys
    ]
    for combination in itertools.product(*choices):
        yield dict(zip(keys, combination))


def results_dir():
    return Path(get_output_root()) / "sweep_results"


def experiment_dir(experiment_name):
    if (
        not experiment_name
        or experiment_name in {".", ".."}
        or Path(experiment_name).name != experiment_name
    ):
        raise ValueError("experiment name must be a single non-empty path component")
    return results_dir() / experiment_name


def results_path(experiment_name):
    return experiment_dir(experiment_name) / "results.csv"


def manifest_path(experiment_name):
    return experiment_dir(experiment_name) / "manifest.json"


def jobs_dir(experiment_name):
    return experiment_dir(experiment_name) / "jobs"


def load_manifest(experiment_name):
    path = manifest_path(experiment_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No schema-v{ARTIFACT_SCHEMA_VERSION} training manifest at {path}; "
            "run training for this experiment first."
        )
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"incompatible training manifest schema at {path}: "
            f"expected {ARTIFACT_SCHEMA_VERSION}, got {manifest.get('schema_version')}"
        )
    return manifest


def initialize_manifest(experiment_name, manifest):
    """Create a manifest or validate that an existing run has the same identity."""
    path = manifest_path(experiment_name)
    if path.exists():
        existing = load_manifest(experiment_name)
        if existing.get("run_identity") != manifest.get("run_identity"):
            raise ValueError(
                f"experiment '{experiment_name}' already exists with different inputs or settings; "
                "choose a new experiment name"
            )
    atomic_write_json(path, manifest)
    return manifest


def update_manifest(experiment_name, manifest):
    existing = load_manifest(experiment_name)
    if existing.get("run_identity") != manifest.get("run_identity"):
        raise ValueError("refusing to replace a manifest with a different run identity")
    atomic_write_json(manifest_path(experiment_name), manifest)


def _atomic_write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_job(path, *, job_id, run_identity):
    try:
        with path.open(encoding="utf-8") as handle:
            job = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        job.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or job.get("job_id") != job_id
        or job.get("run_identity") != run_identity
        or not isinstance(job.get("result"), dict)
    ):
        return None
    return job


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
    seed=42,
    deterministic=False,
    device="auto",
    run_identity=None,
):
    """Train missing fold/config jobs and reconstruct a stable result table."""
    if search_method == "bayes":
        raise NotImplementedError("Bayesian sweeps are unavailable in the local static backend")
    if run_identity is None:
        raise ValueError("run_identity is required for checkpointed local sweeps")

    rows = []
    completed = 0
    executed = 0
    for fold_index, split in enumerate(splits):
        for feature in features:
            for model in models:
                yaml_file = _CONFIG_MAP.get((model.__name__, sweep_depth, search_method))
                if yaml_file is None:
                    raise ValueError(
                        f"invalid sweep configuration: model={model.__name__}, "
                        f"depth={sweep_depth}, method={search_method}"
                    )
                with (SWEEP_CONFIG_DIR / yaml_file).open(encoding="utf-8") as handle:
                    sweep_config = yaml.safe_load(handle)

                for config in _grid_configs(sweep_config):
                    condition = json.dumps(config, sort_keys=True, separators=(",", ":"))
                    model_seed = stable_seed(
                        seed,
                        "train",
                        fold_index,
                        feature.name,
                        model.__name__,
                        condition,
                    )
                    job_contract = {
                        "schema_version": ARTIFACT_SCHEMA_VERSION,
                        "run_identity": run_identity,
                        "fold_index": fold_index,
                        "split_name": split.splits["split_name"],
                        "feature": feature.name,
                        "model": model.__name__,
                        "config": config,
                        "model_seed": model_seed,
                    }
                    job_id = sha256_json(job_contract)
                    job_path = jobs_dir(experiment_name) / f"{job_id}.json"
                    checkpoint = _load_job(
                        job_path,
                        job_id=job_id,
                        run_identity=run_identity,
                    )
                    if checkpoint is not None:
                        rows.append(checkpoint["result"])
                        completed += 1
                        print(f"Reusing completed sweep job {job_id[:12]}")
                        continue

                    seed_everything(model_seed, deterministic=deterministic)
                    model_config = {
                        **config,
                        "seed": model_seed,
                        "dataloader_seed": stable_seed(model_seed, "dataloader"),
                        "deterministic": deterministic,
                        "device": device,
                        "cache_identity": job_id,
                        "force_retrain": True,
                    }
                    instance = model(
                        split,
                        feature,
                        use_cache=use_cache,
                        config=model_config,
                        show_plots=show_plots,
                    )
                    test_stats = instance.run_model()
                    result = {
                        **config,
                        "Model": instance.model_name,
                        "Feature": feature.name,
                        "Split Method": instance.split_method,
                        "Test Loss": test_stats["MSE"],
                        "Spearman - Test": test_stats["Spearman r"],
                        "Pearson - Test": test_stats["Pearson r"],
                        "name": instance.file_attrs.get("model_name", ""),
                        "Model Seed": model_seed,
                        "Best Epoch": instance.best_epoch,
                        "Best Validation Loss": instance.best_validation_loss,
                        "Stopped Epoch": instance.stopped_epoch,
                        "Job ID": job_id,
                    }
                    atomic_write_json(
                        job_path,
                        {
                            **job_contract,
                            "job_id": job_id,
                            "result": result,
                        },
                    )
                    rows.append(result)
                    executed += 1

    if not rows:
        raise ValueError("sweep produced no jobs")

    preferred_columns = [
        "layer_size",
        "num_layers",
        "learning_rate",
        "batch_size",
        "optimizer",
        "epochs",
        "Model",
        "Feature",
        "Split Method",
        "Test Loss",
        "Spearman - Test",
        "Pearson - Test",
        "name",
        "Model Seed",
        "Best Epoch",
        "Best Validation Loss",
        "Stopped Epoch",
        "Job ID",
    ]
    available = {key for row in rows for key in row}
    columns = [key for key in preferred_columns if key in available]
    columns.extend(sorted(available.difference(columns)))
    results = (
        pd.DataFrame(rows, columns=columns)
        .sort_values("Job ID")
        .reset_index(drop=True)
    )
    _atomic_write_csv(results_path(experiment_name), results)
    print(
        f"Wrote {len(results)} sweep result(s) to {results_path(experiment_name)} "
        f"({completed} reused, {executed} executed)"
    )
    return results


def load_sweep_results(experiment_name):
    load_manifest(experiment_name)
    path = results_path(experiment_name)
    if not path.exists():
        raise FileNotFoundError(f"No sweep results at {path}; run training first")
    return pd.read_csv(path)
