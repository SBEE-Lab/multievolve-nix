import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from multievolve.cli.propose import _compatible_training_manifest, _load_model_checkpoint
from multievolve.predictors.neural_net_regressors import run_nn_model_experiments
from multievolve.utils.local_sweep import (
    ARTIFACT_SCHEMA_VERSION,
    initialize_manifest,
    jobs_dir,
    load_manifest,
    load_sweep_results,
    manifest_path,
    results_path,
    run_local_sweep,
)
from multievolve.utils.reproducibility import (
    atomic_write_json,
    canonical_csv_sha256,
    canonical_fasta_sha256,
    sha256_json,
)


class _Split:
    def __init__(self, name):
        self.splits = {"split_name": name}


class _Feature:
    name = "onehot"


class Fcn:
    runs = 0

    def __init__(self, split, feature, **kwargs):
        self.model_name = "fcn"
        self.split_method = split.splits["split_name"]
        self.file_attrs = {"model_name": f"{self.split_method}-fcn"}
        self.best_epoch = 1
        self.best_validation_loss = 0.5
        self.stopped_epoch = 2

    def run_model(self):
        type(self).runs += 1
        return {"MSE": 0.25, "Spearman r": 0.5, "Pearson r": 0.75}


class CheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.old_root = os.environ.get("MULTIEVOLVE_ROOT")
        os.environ["MULTIEVOLVE_ROOT"] = self.directory.name

    def tearDown(self):
        if self.old_root is None:
            os.environ.pop("MULTIEVOLVE_ROOT", None)
        else:
            os.environ["MULTIEVOLVE_ROOT"] = self.old_root
        self.directory.cleanup()

    def test_canonical_hashes_ignore_nonsemantic_formatting(self):
        root = Path(self.directory.name)
        fasta_a = root / "a.fasta"
        fasta_b = root / "b.fasta"
        fasta_a.write_text(">first\nac\nde\n")
        fasta_b.write_text(">other header\nACDE\n")
        self.assertEqual(
            canonical_fasta_sha256(fasta_a),
            canonical_fasta_sha256(fasta_b),
        )

        csv_a = root / "a.csv"
        csv_b = root / "b.csv"
        csv_a.write_bytes(b"mutation,property_value\r\nWT,1.0\r\nA1V,2.0\r\n")
        csv_b.write_text("mutation,property_value\nWT,1.0\nA1V,2.0\n\n")
        self.assertEqual(canonical_csv_sha256(csv_a), canonical_csv_sha256(csv_b))

    def test_atomic_json_does_not_leave_temporary_file(self):
        path = Path(self.directory.name) / "nested" / "value.json"
        atomic_write_json(path, {"value": 1})
        self.assertEqual(json.loads(path.read_text()), {"value": 1})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_model_checkpoint_requires_matching_artifact_hash(self):
        root = Path(self.directory.name)
        artifact = root / "model.pth"
        artifact.write_bytes(b"model-state")
        checkpoint_path = root / "checkpoint.json"
        contract = {"schema_version": ARTIFACT_SCHEMA_VERSION, "fold_index": 0}
        job_id = sha256_json(contract)
        atomic_write_json(
            checkpoint_path,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "job_id": job_id,
                "job_contract": contract,
                "model_artifact": {
                    "path": str(artifact),
                    "sha256": hashlib.sha256(b"model-state").hexdigest(),
                },
            },
        )
        self.assertIsNotNone(
            _load_model_checkpoint(checkpoint_path, job_id, expected_contract=contract)
        )
        artifact.write_bytes(b"corrupt")
        self.assertIsNone(
            _load_model_checkpoint(checkpoint_path, job_id, expected_contract=contract)
        )

    def test_manifest_rejects_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "single non-empty path component"):
            initialize_manifest("..", {"run_identity": "unsafe"})

    def test_manifest_rejects_stale_schema(self):
        atomic_write_json(
            manifest_path("stale"),
            {"schema_version": ARTIFACT_SCHEMA_VERSION - 1, "run_identity": "old"},
        )
        with self.assertRaisesRegex(ValueError, "incompatible training manifest schema"):
            load_manifest("stale")

    def test_manifest_rejects_reusing_experiment_for_another_identity(self):
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_identity": "first",
        }
        initialize_manifest("example", manifest)
        initialize_manifest("example", manifest)
        with self.assertRaisesRegex(ValueError, "different inputs or settings"):
            initialize_manifest(
                "example",
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "run_identity": "second",
                },
            )

    def test_manifest_resume_preserves_completion_metadata(self):
        complete = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_identity": "same",
            "completion": {"job_ids": ["job"], "job_count": 1},
        }
        initialize_manifest("preserve", complete)
        observed = initialize_manifest(
            "preserve",
            {"schema_version": ARTIFACT_SCHEMA_VERSION, "run_identity": "same"},
        )
        self.assertEqual(observed["completion"], complete["completion"])
        self.assertEqual(load_manifest("preserve")["completion"], complete["completion"])

    def test_sweep_reuses_jobs_and_recovers_only_a_missing_job(self):
        Fcn.runs = 0
        arguments = {
            "splits": [_Split("fold-0"), _Split("fold-1")],
            "features": [_Feature()],
            "models": [Fcn],
            "experiment_name": "resume",
            "sweep_depth": "test",
            "search_method": "test",
            "show_plots": False,
            "seed": 42,
            "deterministic": True,
            "device": "cpu",
            "run_identity": "run-identity",
        }
        first = run_local_sweep(**arguments)
        self.assertEqual(Fcn.runs, 2)

        second = run_local_sweep(**arguments)
        self.assertEqual(Fcn.runs, 2)
        self.assertEqual(first.to_csv(index=False), second.to_csv(index=False))

        job_path = next(jobs_dir("resume").glob("*.json"))
        job_path.unlink()
        third = run_local_sweep(**arguments)
        self.assertEqual(Fcn.runs, 3)
        self.assertEqual(first.to_csv(index=False), third.to_csv(index=False))

        job_path = next(jobs_dir("resume").glob("*.json"))
        corrupted = json.loads(job_path.read_text())
        corrupted["result"] = {}
        corrupted["result_sha256"] = sha256_json({})
        job_path.write_text(json.dumps(corrupted))
        fourth = run_local_sweep(**arguments)
        self.assertEqual(Fcn.runs, 4)
        self.assertEqual(first.to_csv(index=False), fourth.to_csv(index=False))

    def test_results_table_is_reconstructed_from_valid_jobs(self):
        Fcn.runs = 0
        results = run_local_sweep(
            splits=[_Split("fold-0"), _Split("fold-1")],
            features=[_Feature()],
            models=[Fcn],
            experiment_name="reconstruct",
            sweep_depth="test",
            search_method="test",
            show_plots=False,
            seed=42,
            deterministic=True,
            device="cpu",
            run_identity=None,
        )
        original = results.to_csv(index=False)
        results_path("reconstruct").write_text("tampered\n")
        rebuilt = load_sweep_results("reconstruct")
        self.assertEqual(original, rebuilt.to_csv(index=False))

    def test_public_sweep_api_derives_identity_when_omitted(self):
        Fcn.runs = 0
        arguments = {
            "splits": [_Split("fold-0")],
            "features": [_Feature()],
            "models": [Fcn],
            "experiment_name": "automatic",
            "sweep_depth": "test",
            "search_method": "test",
            "show_plots": False,
            "seed": 42,
            "deterministic": True,
            "device": "cpu",
        }
        first = run_nn_model_experiments(**arguments)
        second = run_nn_model_experiments(**arguments)
        self.assertEqual(Fcn.runs, 1)
        self.assertEqual(first.to_csv(index=False), second.to_csv(index=False))

        changed = dict(arguments)
        changed["experiment_name"] = "automatic-changed"
        changed["splits"] = [_Split("fold-changed")]
        run_nn_model_experiments(**changed)
        self.assertNotEqual(
            load_manifest("automatic")["run_identity"],
            load_manifest("automatic-changed")["run_identity"],
        )

    def test_step_two_rejects_scientifically_different_inputs(self):
        runtime = {"torch": "1", "cuda": None, "source": {"revision": "abc"}}
        manifest = {
            "contract": {
                "dataset_canonical_sha256": "dataset",
                "wt_fasta_canonical_sha256": ["wt"],
                "feature": "OneHot",
                "split_seed": 42,
                "software": runtime,
            }
        }
        _compatible_training_manifest(
            manifest,
            dataset_sha256="dataset",
            wt_sha256=["wt"],
            split_seed=42,
            runtime=runtime,
        )
        with self.assertRaisesRegex(ValueError, "training dataset"):
            _compatible_training_manifest(
                manifest,
                dataset_sha256="changed",
                wt_sha256=["wt"],
                split_seed=42,
                runtime=runtime,
            )


if __name__ == "__main__":
    unittest.main()
