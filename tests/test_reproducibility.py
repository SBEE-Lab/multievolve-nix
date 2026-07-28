import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from multievolve.predictors.neural_net_regressors import BaseNN, Fcn
from multievolve.splitters.base_splitters import KFoldProteinSplitter
from multievolve.utils.other_utils import performance_report
from multievolve.utils.data_utils import TorchDataProcessor
from multievolve.utils.reproducibility import resolve_device, stable_seed


class _NumericFeaturizer:
    name = "numeric"

    def featurize(self, values):
        return np.asarray([[float(value)] for value in values])


class _Split:
    def __init__(self, scaler=None):
        self.splits = {
            "X_train": np.asarray(["0", "1", "2", "3"]),
            "X_val": np.asarray(["4"]),
            "X_test": np.asarray(["5"]),
            "y_train": np.asarray([0.0, 1.0, 2.0, 3.0]),
            "y_val": np.asarray([4.0]),
            "y_test": np.asarray([5.0]),
            "split_name": "test",
            "target_scaler": scaler,
        }
        self.file_attrs = {"dataset_dir": "/tmp", "dataset_name": "test"}


class ReproducibilityTests(unittest.TestCase):
    def test_device_selection_does_not_silently_fallback(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                resolve_device("cuda")

    def test_stable_seed_is_process_stable_and_separates_parts(self):
        expected = stable_seed(42, "train", 2, "condition")
        code = (
            "from multievolve.utils.reproducibility import stable_seed; "
            "print(stable_seed(42, 'train', 2, 'condition'))"
        )
        observed = int(subprocess.check_output([sys.executable, "-c", code], text=True))
        self.assertEqual(observed, expected)
        self.assertGreaterEqual(expected, 0)
        self.assertLess(expected, 2**32)
        self.assertNotEqual(expected, stable_seed(42, "train", 3, "condition"))

    def test_seeded_train_loader_order_is_reproducible(self):
        def order(seed):
            processor = TorchDataProcessor(_Split(), _NumericFeaturizer(), 2, seed=seed)
            return [item for batch in processor.setup_train_loader() for item in batch[2]]

        self.assertEqual(order(11), order(11))
        self.assertNotEqual(order(11), order(12))

    def test_validation_and_test_loaders_preserve_order(self):
        processor = TorchDataProcessor(_Split(), _NumericFeaturizer(), 1, seed=42)
        self.assertEqual([batch[2][0] for batch in processor.setup_val_loader()], ["4"])
        self.assertEqual([batch[2][0] for batch in processor.setup_test_loader()], ["5"])

    def test_performance_report_supports_signed_targets(self):
        report = performance_report([-2.0, -1.0, 1.0], [-1.5, -0.5, 0.8])
        self.assertAlmostEqual(report["MSE"], 0.18)
        self.assertTrue(np.isfinite(report["NDCG"]))
        self.assertAlmostEqual(report["Spearman r"], 1.0)

    def test_performance_report_handles_single_or_constant_samples(self):
        single = performance_report([-2.0], [-1.5])
        self.assertTrue(np.isnan(single["Pearson r"]))
        constant = performance_report([-1.0, -1.0], [0.0, 1.0])
        self.assertTrue(np.isnan(constant["Spearman r"]))
        self.assertTrue(np.isfinite(constant["NDCG"]))

    def test_inverse_transform_uses_fold_scaler(self):
        scaler = MinMaxScaler().fit(np.asarray([2.0, 6.0]).reshape(-1, 1))
        processor = TorchDataProcessor(_Split(scaler), _NumericFeaturizer(), 2, seed=42)
        np.testing.assert_allclose(
            processor.inverse_transform_targets([0.0, 0.5, 1.0]),
            [2.0, 4.0, 6.0],
        )

    def test_kfold_scaler_fits_training_labels_and_does_not_mutate_global_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            old_root = os.environ.get("MULTIEVOLVE_ROOT")
            os.environ["MULTIEVOLVE_ROOT"] = directory
            try:
                fasta = Path(directory) / "wt.fasta"
                fasta.write_text(">wt\nACDE\n")
                data = pd.DataFrame(
                    {
                        "mutation": ["ACDE", "VCDE", "ASDE", "ACNE"] * 3,
                        "property_value": np.arange(12, dtype=float),
                    }
                )
                splitter = KFoldProteinSplitter(
                    "test",
                    data,
                    str(fasta),
                    csv_has_header=True,
                    use_cache=False,
                    random_state=17,
                    y_scaling=True,
                    val_split=0.25,
                )

                np.random.seed(123)
                expected_next = np.random.random()
                np.random.seed(123)
                splits = splitter.generate_splits(2)
                self.assertEqual(np.random.random(), expected_next)

                for fold in splits:
                    train_values = fold.data.loc[fold.data["group"] == 0, 1].to_numpy(float)
                    scaler = fold.splits["target_scaler"]
                    self.assertEqual(float(scaler.data_min_[0]), float(train_values.min()))
                    self.assertEqual(float(scaler.data_max_[0]), float(train_values.max()))
                    restored = scaler.inverse_transform(
                        np.asarray(fold.splits["y_train"]).reshape(-1, 1)
                    ).ravel()
                    np.testing.assert_allclose(np.sort(restored), np.sort(train_values))
            finally:
                if old_root is None:
                    os.environ.pop("MULTIEVOLVE_ROOT", None)
                else:
                    os.environ["MULTIEVOLVE_ROOT"] = old_root

    def test_fcn_seeds_layers_before_initialization(self):
        config = {
            "layer_size": 4,
            "num_layers": 1,
            "learning_rate": 0.001,
            "batch_size": 2,
            "optimizer": "adam",
            "epochs": 1,
            "seed": 123,
            "device": "cpu",
            "deterministic": False,
        }
        first = Fcn(_Split(), _NumericFeaturizer(), config=config)
        second = Fcn(_Split(), _NumericFeaturizer(), config=config)
        for name, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[name])
        self.assertIn("cpu __ deterministic0", first.file_attrs["model_name"])

    def test_early_stopping_snapshot_is_independent_and_restorable(self):
        holder = BaseNN.__new__(BaseNN)
        torch.nn.Module.__init__(holder)
        holder.val_loss_min = float("inf")
        holder.val_loss_delta_min = 0.00001
        holder.epochs_no_improve = 0
        holder.patience = 2
        holder.best_epoch = None
        holder.best_validation_loss = None
        holder.best_state_dict = None
        holder.stopped_epoch = None

        model = torch.nn.Linear(1, 1)
        self.assertFalse(holder.early_stopping_check(1.0, 0, model))
        expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        with torch.no_grad():
            model.weight.add_(10)
        self.assertFalse(holder.early_stopping_check(2.0, 1, model))
        self.assertTrue(holder.early_stopping_check(2.0, 2, model))
        self.assertIsNotNone(holder.best_state_dict)
        model.load_state_dict(holder.best_state_dict)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, expected[name])
        self.assertEqual(holder.best_epoch, 0)
        self.assertEqual(holder.stopped_epoch, 2)


if __name__ == "__main__":
    unittest.main()
