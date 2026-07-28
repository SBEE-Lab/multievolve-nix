import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from multievolve.app import stream_command_output
from multievolve.app_io import (
    AppInputError,
    prepare_uploaded_inputs,
    validate_identifier,
    validate_upload_filename,
)


class _Upload:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def getbuffer(self):
        return memoryview(self.content)


_FASTA = b">wildtype\nACDEFG\n"
_DATASET = b"mutation,property_value\nA1V,1.0\n"
_MUTATIONS = b"A1V\nC2S\n"


class AppInputSecurityTests(unittest.TestCase):
    def test_identifier_policy_accepts_documented_components(self):
        for value in ("APEX", "apex.v2", "run_01", "round-2"):
            with self.subTest(value=value):
                self.assertEqual(validate_identifier(value, "Name"), value)

    def test_identifier_policy_rejects_path_and_ambiguous_components(self):
        invalid = (
            "",
            ".",
            "..",
            " two",
            "two words",
            "../outside",
            "nested/name",
            r"nested\name",
            "/absolute",
            "name,other",
            "<script>",
            "éclair",
            "a\nname",
            "a" * 129,
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(AppInputError):
                    validate_identifier(value, "Name")

    def test_upload_filename_policy_is_bounded_and_role_specific(self):
        self.assertEqual(
            validate_upload_filename(
                "model.V1.FASTA",
                label="FASTA",
                allowed_suffixes=frozenset({".fa", ".fasta"}),
            ),
            "model.V1.FASTA",
        )
        maximum_length_name = f"{'a' * 249}.fasta"
        self.assertEqual(
            validate_upload_filename(
                maximum_length_name,
                label="FASTA",
                allowed_suffixes=frozenset({".fa", ".fasta"}),
            ),
            maximum_length_name,
        )
        for value in (
            "../outside.fasta",
            r"..\outside.fasta",
            "/outside.fasta",
            ".hidden.fasta",
            "two words.fasta",
            "one.fasta,../../outside",
            "<img>.fasta",
            "a\nname.fasta",
            f"{'a' * 250}.fasta",
            "sequence.csv",
        ):
            with self.subTest(value=value):
                with self.assertRaises(AppInputError):
                    validate_upload_filename(
                        value,
                        label="FASTA",
                        allowed_suffixes=frozenset({".fa", ".fasta"}),
                    )

    def test_output_root_not_working_directory_controls_project_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            state_root = base / "state"
            other_cwd = base / "cwd"
            other_cwd.mkdir()
            previous_cwd = Path.cwd()
            try:
                os.chdir(other_cwd)
                with patch.dict(
                    os.environ,
                    {"MULTIEVOLVE_ROOT": str(state_root)},
                    clear=False,
                ):
                    prepared = prepare_uploaded_inputs(
                        "APEX",
                        wt_file_aa=_Upload("apex.fasta", _FASTA),
                    )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(prepared.project_dir, state_root / "proteins" / "APEX")
            self.assertEqual(
                prepared.wt_file_aa,
                state_root / "proteins" / "APEX" / "apex.fasta",
            )
            self.assertFalse((other_cwd / "proteins").exists())

    def test_maximum_length_upload_name_can_be_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = f"{'a' * 249}.fasta"
            prepared = prepare_uploaded_inputs(
                "APEX",
                wt_file_aa=_Upload(filename, _FASTA),
                output_root=Path(directory) / "state",
            )
            self.assertEqual(prepared.wt_file_aa.name, filename)
            self.assertEqual(prepared.wt_file_aa.read_bytes(), _FASTA)

    def test_state_root_rejects_the_cli_list_delimiter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "state,other"
            with self.assertRaises(AppInputError):
                prepare_uploaded_inputs(
                    "APEX",
                    wt_file_aa=_Upload("wt.fasta", _FASTA),
                    output_root=root,
                )
            self.assertFalse(root.exists())

    def test_malicious_project_and_upload_names_create_no_outside_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "state"
            outside_project = base / "outside-project"
            with self.assertRaises(AppInputError):
                prepare_uploaded_inputs(
                    str(outside_project),
                    wt_file_aa=_Upload("wt.fasta", _FASTA),
                    output_root=root,
                )
            self.assertFalse(outside_project.exists())
            self.assertFalse(root.exists())

            outside_upload = base / "outside.csv"
            with self.assertRaises(AppInputError):
                prepare_uploaded_inputs(
                    "APEX",
                    dataset_file=_Upload("../../outside.csv", _DATASET),
                    output_root=root,
                )
            self.assertFalse(outside_upload.exists())
            self.assertFalse((root / "proteins" / "APEX").exists())

    def test_rejects_symlinked_projects_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "state"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "proteins").symlink_to(outside, target_is_directory=True)

            with self.assertRaises(AppInputError):
                prepare_uploaded_inputs(
                    "APEX",
                    wt_file_aa=_Upload("wt.fasta", _FASTA),
                    output_root=root,
                )
            self.assertEqual(list(outside.iterdir()), [])

    def test_rejects_symlinked_project(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "state"
            projects = root / "proteins"
            outside = base / "outside"
            projects.mkdir(parents=True)
            outside.mkdir()
            (projects / "APEX").symlink_to(outside, target_is_directory=True)

            with self.assertRaises(AppInputError):
                prepare_uploaded_inputs(
                    "APEX",
                    wt_file_aa=_Upload("wt.fasta", _FASTA),
                    output_root=root,
                )
            self.assertEqual(list(outside.iterdir()), [])

    def test_rejects_symlinked_upload_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "state"
            project = root / "proteins" / "APEX"
            project.mkdir(parents=True)
            outside = base / "outside.fasta"
            outside.write_bytes(b"preserve\n")
            (project / "wt.fasta").symlink_to(outside)

            with self.assertRaises(AppInputError):
                prepare_uploaded_inputs(
                    "APEX",
                    wt_file_aa=_Upload("wt.fasta", _FASTA),
                    output_root=root,
                )
            self.assertEqual(outside.read_bytes(), b"preserve\n")

    def test_malformed_input_does_not_replace_valid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "state"
            valid = prepare_uploaded_inputs(
                "APEX",
                dataset_file=_Upload("dataset.csv", _DATASET),
                output_root=root,
            )
            self.assertEqual(valid.dataset_file.read_bytes(), _DATASET)

            with self.assertRaises(AppInputError):
                prepare_uploaded_inputs(
                    "APEX",
                    dataset_file=_Upload(
                        "dataset.csv",
                        b"wrong,columns\nA1V,1.0\n",
                    ),
                    output_root=root,
                )
            self.assertEqual(valid.dataset_file.read_bytes(), _DATASET)

    def test_malformed_fasta_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "state"
            with self.assertRaises(AppInputError):
                prepare_uploaded_inputs(
                    "APEX",
                    wt_file_aa=_Upload("wt.fasta", b"not fasta\n"),
                    output_root=root,
                )
            self.assertFalse((root / "proteins" / "APEX").exists())

    def test_duplicate_destinations_fail_before_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "state"
            with self.assertRaises(AppInputError):
                prepare_uploaded_inputs(
                    "APEX",
                    dataset_file=_Upload("input.csv", _DATASET),
                    mutations_file=_Upload("input.csv", _MUTATIONS),
                    output_root=root,
                )
            self.assertFalse((root / "proteins" / "APEX").exists())

    def test_all_workflow_paths_are_absolute_and_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "state"
            prepared = prepare_uploaded_inputs(
                "APEX.v2",
                wt_files_aa=[
                    _Upload("chain_1.fa", _FASTA),
                    _Upload("chain_2.fasta", _FASTA),
                ],
                wt_file_dna=_Upload("wildtype_dna.fasta", _FASTA),
                dataset_file=_Upload("dataset.csv", _DATASET),
                mutations_file=_Upload("mutations.csv", _MUTATIONS),
                pdb_files=[
                    _Upload("model_1.PDB", b"structure\n"),
                    _Upload("model_2.cif", b"structure\n"),
                ],
                output_root=root,
            )
            paths = (
                *prepared.wt_files_aa,
                prepared.wt_file_dna,
                prepared.dataset_file,
                prepared.mutations_file,
                *prepared.pdb_files,
            )
            for path in paths:
                with self.subTest(path=path):
                    self.assertTrue(path.is_absolute())
                    self.assertEqual(path.parent, prepared.project_dir)
                    self.assertTrue(path.is_file())
            self.assertEqual(
                list(prepared.project_dir.glob(".*.tmp")),
                [],
            )


class TerminalOutputSecurityTests(unittest.TestCase):
    def test_child_output_is_streamed_as_text_without_a_shell(self):
        payload = "</code></pre><img src=x onerror=alert(1)>"
        ansi_payload = "\x1b[31m& second line\x1b[0m"
        process = MagicMock()
        process.stdout.readline.side_effect = [
            f"{payload}\n",
            f"{ansi_payload}\n",
            "",
        ]
        process.wait.return_value = 17
        placeholder = MagicMock()
        command = ["python", "-c", "print('untrusted')"]

        with patch("multievolve.app.subprocess.Popen", return_value=process) as popen:
            return_code = stream_command_output(command, placeholder)

        self.assertEqual(return_code, 17)
        popen.assert_called_once_with(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.assertNotIn("shell", popen.call_args.kwargs)
        placeholder.markdown.assert_not_called()
        self.assertEqual(placeholder.code.call_count, 3)
        self.assertEqual(placeholder.code.call_args_list[0].args[0], payload)
        self.assertEqual(
            placeholder.code.call_args_list[-1].args[0],
            f"{payload}\n{ansi_payload}",
        )
        for call in placeholder.code.call_args_list:
            self.assertEqual(
                call.kwargs,
                {
                    "language": None,
                    "wrap_lines": True,
                    "height": 400,
                },
            )


if __name__ == "__main__":
    unittest.main()
