"""Content-addressed local grid search with atomic per-job checkpoints."""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

import pandas as pd
import yaml

from multievolve.utils.paths import get_output_root
from multievolve.utils.reproducibility import (
    atomic_write_json,
    resolve_device,
    runtime_identity,
    seed_everything,
    sha256_array,
    sha256_file,
    sha256_json,
    stable_seed,
)

ARTIFACT_SCHEMA_VERSION = 4
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


def _load_grid(model, sweep_depth, search_method):
    yaml_file = _CONFIG_MAP.get((model.__name__, sweep_depth, search_method))
    if yaml_file is None:
        raise ValueError(
            f"invalid sweep configuration: model={model.__name__}, "
            f"depth={sweep_depth}, method={search_method}"
        )
    path = SWEEP_CONFIG_DIR / yaml_file
    with path.open(encoding="utf-8") as handle:
        return path, yaml.safe_load(handle)


def _split_content_identity(split):
    values = split.splits
    arrays = {
        key: sha256_array(values[key])
        for key in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test")
        if key in values
    }
    scaler = values.get("target_scaler")
    scaler_identity = None
    if scaler is not None:
        scaler_identity = {
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "feature_range": list(scaler.feature_range),
        }
    return {
        "split_name": values.get("split_name"),
        "arrays": arrays,
        "target_scaler": scaler_identity,
    }


def _public_object_parameters(value):
    parameters = {}
    for key, item in vars(value).items():
        if key.startswith("_") or key in {"protein", "use_cache", "device"}:
            continue
        if isinstance(item, Path):
            item = str(item)
        try:
            sha256_json(item)
        except (TypeError, ValueError):
            continue
        parameters[key] = item
    return parameters


def _fallback_sweep_contract(
    splits,
    features,
    models,
    *,
    sweep_depth,
    search_method,
    seed,
    deterministic,
    device,
):
    seed_everything(seed, deterministic=deterministic)
    actual_device = resolve_device(device)
    grids = []
    for model in models:
        path, config = _load_grid(model, sweep_depth, search_method)
        grids.append(
            {
                "model": f"{model.__module__}.{model.__qualname__}",
                "path": path.name,
                "sha256": sha256_file(path),
                "configs": list(_grid_configs(config)),
            }
        )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "command": "api-sweep",
        "splits": [_split_content_identity(split) for split in splits],
        "features": [
            {
                "class": f"{feature.__class__.__module__}.{feature.__class__.__qualname__}",
                "name": getattr(feature, "name", None),
                "parameters": _public_object_parameters(feature),
            }
            for feature in features
        ],
        "models": [f"{model.__module__}.{model.__qualname__}" for model in models],
        "grids": grids,
        "sweep_depth": sweep_depth,
        "search_method": search_method,
        "seed": seed,
        "deterministic": deterministic,
        "device": actual_device,
        "software": runtime_identity(actual_device),
    }


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
    merged = dict(manifest)
    if path.exists():
        existing = load_manifest(experiment_name)
        if existing.get("run_identity") != manifest.get("run_identity"):
            raise ValueError(
                f"experiment '{experiment_name}' already exists with different inputs or settings; "
                "choose a new experiment name"
            )
        for key in ("completion", "model_seeds", "grid_sha256"):
            if key in existing and key not in merged:
                merged[key] = existing[key]
    atomic_write_json(path, merged)
    return merged


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


_REQUIRED_RESULT_KEYS = {
    "Model",
    "Feature",
    "Split Method",
    "Test Loss",
    "Spearman - Test",
    "Pearson - Test",
    "Model Seed",
    "Best Epoch",
    "Best Validation Loss",
    "Stopped Epoch",
    "Job ID",
}


def _load_job(path, *, job_id, run_identity, expected_contract=None):
    try:
        with path.open(encoding="utf-8") as handle:
            job = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    contract = job.get("job_contract")
    result = job.get("result")
    if (
        job.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or job.get("job_id") != job_id
        or job.get("run_identity") != run_identity
        or not isinstance(contract, dict)
        or sha256_json(contract) != job_id
        or (expected_contract is not None and contract != expected_contract)
        or not isinstance(result, dict)
        or job.get("result_sha256") != sha256_json(result)
        or not _REQUIRED_RESULT_KEYS.issubset(result)
        or result.get("Job ID") != job_id
        or result.get("Model Seed") != contract.get("model_seed")
        or result.get("Feature") != contract.get("feature")
        or result.get("Split Method") != contract.get("split_name")
    ):
        return None
    for key, value in contract.get("config", {}).items():
        if result.get(key) != value:
            return None
    try:
        if not math.isfinite(float(result["Test Loss"])):
            return None
    except (TypeError, ValueError):
        return None
    return job


def _results_frame(rows):
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
    return pd.DataFrame(rows, columns=columns).sort_values("Job ID").reset_index(drop=True)


def sweep_completion(results, experiment_name):
    return {
        "job_ids": results["Job ID"].tolist(),
        "job_count": len(results),
        "results_sha256": sha256_file(results_path(experiment_name)),
    }


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

    auto_manifest = run_identity is None
    if auto_manifest:
        contract = _fallback_sweep_contract(
            splits,
            features,
            models,
            sweep_depth=sweep_depth,
            search_method=search_method,
            seed=seed,
            deterministic=deterministic,
            device=device,
        )
        run_identity = sha256_json(contract)
        initialize_manifest(
            experiment_name,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_identity": run_identity,
                "contract": contract,
                "provenance": {"runtime": contract["software"]},
            },
        )

    rows = []
    completed = 0
    executed = 0
    for fold_index, split in enumerate(splits):
        for feature in features:
            for model in models:
                _, sweep_config = _load_grid(model, sweep_depth, search_method)

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
                        expected_contract=job_contract,
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
                            "schema_version": ARTIFACT_SCHEMA_VERSION,
                            "job_id": job_id,
                            "run_identity": run_identity,
                            "job_contract": job_contract,
                            "result": result,
                            "result_sha256": sha256_json(result),
                        },
                    )
                    rows.append(result)
                    executed += 1

    results = _results_frame(rows)
    _atomic_write_csv(results_path(experiment_name), results)
    if auto_manifest:
        manifest = load_manifest(experiment_name)
        manifest["completion"] = sweep_completion(results, experiment_name)
        update_manifest(experiment_name, manifest)
    print(
        f"Wrote {len(results)} sweep result(s) to {results_path(experiment_name)} "
        f"({completed} reused, {executed} executed)"
    )
    return results


def load_sweep_results(experiment_name):
    """Load a hash-validated result table, rebuilding it from valid jobs if needed."""
    manifest = load_manifest(experiment_name)
    completion = manifest.get("completion")
    job_ids = completion.get("job_ids") if isinstance(completion, dict) else None
    expected_sha256 = (
        completion.get("results_sha256") if isinstance(completion, dict) else None
    )
    if (
        not isinstance(job_ids, list)
        or not job_ids
        or any(not isinstance(job_id, str) for job_id in job_ids)
        or len(set(job_ids)) != len(job_ids)
        or completion.get("job_count") != len(job_ids)
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise ValueError(
            f"training manifest for '{experiment_name}' is incomplete; resume training first"
        )

    path = results_path(experiment_name)
    if path.is_file() and sha256_file(path) == expected_sha256:
        return pd.read_csv(path)

    rows = []
    for job_id in job_ids:
        checkpoint = _load_job(
            jobs_dir(experiment_name) / f"{job_id}.json",
            job_id=job_id,
            run_identity=manifest["run_identity"],
        )
        if checkpoint is None:
            raise ValueError(
                f"sweep artifact {job_id} is missing or invalid; resume training first"
            )
        rows.append(checkpoint["result"])

    results = _results_frame(rows)
    if len(results) != completion.get("job_count"):
        raise ValueError("sweep completion job count does not match its valid artifacts")
    _atomic_write_csv(path, results)
    if sha256_file(path) != expected_sha256:
        raise ValueError("reconstructed sweep results do not match the completion manifest")
    return results
