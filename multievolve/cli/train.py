#!/usr/bin/env python3

"""Train MULTI-evolve neural-network models with a local static grid."""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

from multievolve.featurizers import OneHotFeaturizer  # noqa: E402
from multievolve.predictors import Fcn, run_nn_model_experiments  # noqa: E402
from multievolve.splitters import KFoldProteinSplitter  # noqa: E402
from multievolve.utils.local_sweep import write_manifest  # noqa: E402
from multievolve.utils.reproducibility import (  # noqa: E402
    resolve_device,
    runtime_identity,
    seed_everything,
    sha256_file,
    sha256_json,
    stable_seed,
)


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _seed(value):
    value = int(value)
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError("must satisfy 0 <= seed < 2**32")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Train MULTI-evolve neural network models")
    parser.add_argument("-e", "--experiment-name", required=True)
    parser.add_argument("-p", "--protein-name", required=True)
    parser.add_argument(
        "-wt",
        "--wt-files",
        required=True,
        help="Comma-separated paths to wildtype FASTA files",
    )
    parser.add_argument("-t", "--training-dataset-fname", required=True)
    parser.add_argument("-m", "--mode", required=True, choices=["test", "standard"])
    parser.add_argument("--seed", type=_seed, default=42)
    parser.add_argument(
        "--split-seed",
        type=_seed,
        default=None,
        help="Fold-assignment seed (default: --seed)",
    )
    parser.add_argument("--cv-folds", type=_positive_int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    args.wt_files = [path.strip() for path in args.wt_files.split(",")]
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.cv_folds < 2:
        raise SystemExit("error: --cv-folds must be at least 2")
    return args


def main():
    args = parse_args()

    try:
        seed_everything(args.seed, deterministic=args.deterministic)
        actual_device = resolve_device(args.device)
        fold_splitter = KFoldProteinSplitter(
            args.protein_name,
            args.training_dataset_fname,
            args.wt_files,
            csv_has_header=True,
            use_cache=True,
            random_state=args.split_seed,
            y_scaling=True,
            val_split=0.15,
        )
        splits = fold_splitter.generate_splits(n_splits=args.cv_folds)
        features = [OneHotFeaturizer(protein=args.protein_name, use_cache=True)]

        if args.mode == "test":
            print("Running in test mode")
            sweep_depth = "test"
            search_method = "test"
        else:
            print("Running in standard mode")
            sweep_depth = "standard"
            search_method = "grid"

        print(f"Running experiments for {args.experiment_name} with {args.protein_name}...")
        sweep_results = run_nn_model_experiments(
            splits,
            features,
            [Fcn],
            experiment_name=args.experiment_name,
            use_cache=False,
            sweep_depth=sweep_depth,
            search_method=search_method,
            show_plots=True,
            seed=args.seed,
            deterministic=args.deterministic,
            device=args.device,
        )
        write_manifest(
            args.experiment_name,
            {
                "schema_version": 2,
                "command": "train",
                "experiment_name": args.experiment_name,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "deterministic": args.deterministic,
                "fold_count": args.cv_folds,
                "dataset_sha256": sha256_file(args.training_dataset_fname),
                "wt_fasta_sha256": [sha256_file(path) for path in args.wt_files],
                "feature": "OneHot",
                "mode": args.mode,
                "fold_assignment_sha256": sha256_json(
                    fold_splitter.data["fold"].astype(int).tolist()
                ),
                "fold_scalers": [
                    {
                        "fold_index": fold_index,
                        "split_name": split.splits["split_name"],
                        "validation_seed": stable_seed(
                            args.split_seed,
                            "validation",
                            f"kfold-{fold_index}",
                            -1,
                        ),
                        "data_min": split.splits["target_scaler"].data_min_.tolist(),
                        "data_max": split.splits["target_scaler"].data_max_.tolist(),
                    }
                    for fold_index, split in enumerate(splits)
                ],
                "model_seeds": sweep_results["Model Seed"].astype(int).tolist(),
                "grid_sha256": sha256_json(
                    sweep_results[
                        [
                            "layer_size",
                            "num_layers",
                            "learning_rate",
                            "batch_size",
                            "optimizer",
                            "epochs",
                        ]
                    ]
                    .drop_duplicates()
                    .to_dict(orient="records")
                ),
                "runtime": runtime_identity(actual_device),
            },
        )
    except Exception as exc:
        print(f"Error running training: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
