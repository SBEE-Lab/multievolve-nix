"""Validated filesystem boundary for browser-provided Streamlit inputs."""

from __future__ import annotations

import io
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from Bio import SeqIO

from multievolve.utils.paths import get_output_root

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_UPLOAD_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")

_FASTA_SUFFIXES = frozenset({".fa", ".fasta"})
_CSV_SUFFIXES = frozenset({".csv"})
_STRUCTURE_SUFFIXES = frozenset({".cif", ".pdb"})


class AppInputError(ValueError):
    """A browser-provided value failed the application input contract."""


class UploadedFile(Protocol):
    """The subset of Streamlit's UploadedFile API used by this module."""

    name: str

    def getbuffer(self): ...


@dataclass(frozen=True)
class PreparedInputs:
    """Canonical paths for one validated and persisted app submission."""

    project_dir: Path
    wt_files_aa: tuple[Path, ...] = ()
    wt_file_aa: Path | None = None
    wt_file_dna: Path | None = None
    dataset_file: Path | None = None
    mutations_file: Path | None = None
    pdb_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _PendingUpload:
    role: str
    index: int
    filename: str
    data: bytes


def validate_identifier(value: str, label: str) -> str:
    """Require a bounded artifact name with no path or delimiter syntax."""
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise AppInputError(
            f"{label} must start with a letter or number and contain only letters, "
            "numbers, '.', '_', or '-' (maximum 128 characters)"
        )
    return value


def validate_upload_filename(
    filename: str,
    *,
    label: str,
    allowed_suffixes: frozenset[str],
) -> str:
    """Validate an uploaded basename and its role-specific suffix."""
    if (
        not isinstance(filename, str)
        or _UPLOAD_NAME_PATTERN.fullmatch(filename) is None
    ):
        raise AppInputError(
            f"{label} filename must start with a letter or number and contain only "
            "letters, numbers, '.', '_', or '-' (maximum 255 characters)"
        )
    if Path(filename).suffix.lower() not in allowed_suffixes:
        expected = ", ".join(sorted(allowed_suffixes))
        raise AppInputError(f"{label} filename must end with one of: {expected}")
    return filename


def _upload_bytes(uploaded_file: UploadedFile, label: str) -> bytes:
    try:
        return bytes(uploaded_file.getbuffer())
    except Exception as exc:
        raise AppInputError(f"Could not read {label}") from exc


def _validate_fasta(data: bytes, label: str) -> None:
    try:
        SeqIO.read(io.StringIO(data.decode("utf-8")), "fasta")
    except Exception as exc:
        raise AppInputError(
            f"{label} must contain exactly one valid FASTA record"
        ) from exc


def _validate_training_dataset(data: bytes) -> None:
    try:
        frame = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise AppInputError("Training dataset must be a valid CSV file") from exc
    required_columns = {"mutation", "property_value"}
    if not required_columns.issubset(frame.columns):
        raise AppInputError(
            "Training dataset must contain 'mutation' and 'property_value' columns"
        )


def _validate_mutation_pool(data: bytes) -> None:
    try:
        frame = pd.read_csv(io.BytesIO(data), header=None)
    except Exception as exc:
        raise AppInputError("Mutation pool must be a valid non-empty CSV file") from exc
    if frame.empty:
        raise AppInputError("Mutation pool file is empty")


def _pending_upload(
    uploaded_file: UploadedFile,
    *,
    role: str,
    index: int = 0,
) -> _PendingUpload:
    label = {
        "wt_files_aa": "Wildtype amino acid FASTA",
        "wt_file_aa": "Wildtype amino acid FASTA",
        "wt_file_dna": "Wildtype DNA FASTA",
        "dataset_file": "Training dataset",
        "mutations_file": "Mutation pool",
        "pdb_files": "Structure",
    }[role]
    suffixes = {
        "wt_files_aa": _FASTA_SUFFIXES,
        "wt_file_aa": _FASTA_SUFFIXES,
        "wt_file_dna": _FASTA_SUFFIXES,
        "dataset_file": _CSV_SUFFIXES,
        "mutations_file": _CSV_SUFFIXES,
        "pdb_files": _STRUCTURE_SUFFIXES,
    }[role]
    filename = validate_upload_filename(
        uploaded_file.name,
        label=label,
        allowed_suffixes=suffixes,
    )
    data = _upload_bytes(uploaded_file, label)

    if role in {"wt_files_aa", "wt_file_aa", "wt_file_dna"}:
        _validate_fasta(data, label)
    elif role == "dataset_file":
        _validate_training_dataset(data)
    elif role == "mutations_file":
        _validate_mutation_pool(data)

    return _PendingUpload(role=role, index=index, filename=filename, data=data)


def _canonical_projects_root(output_root: str | os.PathLike[str] | None) -> Path:
    root = Path(get_output_root() if output_root is None else output_root).resolve()
    if "," in str(root):
        raise AppInputError(
            "The configured MULTI-evolve state root may not contain a comma"
        )
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AppInputError(
            "The configured MULTI-evolve state root is not writable"
        ) from exc
    if not root.is_dir():
        raise AppInputError("The configured MULTI-evolve state root is not a directory")

    projects = root / "proteins"
    if projects.is_symlink():
        raise AppInputError("The application proteins directory may not be a symlink")
    try:
        projects.mkdir(exist_ok=True)
    except OSError as exc:
        raise AppInputError(
            "Could not create the application proteins directory"
        ) from exc
    if not projects.is_dir() or projects.resolve() != projects:
        raise AppInputError(
            "The application proteins directory is not a safe directory"
        )
    return projects


def resolve_project_directory(
    protein_name: str,
    *,
    output_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Create and return a canonical direct child of the projects root."""
    name = validate_identifier(protein_name, "Protein name")
    projects = _canonical_projects_root(output_root)
    project = projects / name
    if project.is_symlink():
        raise AppInputError("The selected protein project may not be a symlink")
    try:
        project.mkdir(exist_ok=True)
    except OSError as exc:
        raise AppInputError("Could not create the selected protein project") from exc
    if not project.is_dir() or project.resolve().parent != projects:
        raise AppInputError(
            "The selected protein project is outside the projects directory"
        )
    return project.resolve()


def _upload_destination(project: Path, filename: str) -> Path:
    destination = project / filename
    if destination.is_symlink():
        raise AppInputError(f"Upload destination '{filename}' may not be a symlink")
    if destination.exists() and not destination.is_file():
        raise AppInputError(f"Upload destination '{filename}' is not a regular file")
    if destination.resolve().parent != project:
        raise AppInputError(f"Upload destination '{filename}' is outside the project")
    return destination


def _atomic_write(destination: Path, data: bytes) -> None:
    temporary = destination.with_name(f".upload-{secrets.token_hex(16)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
        os.replace(temporary, destination)
    except OSError as exc:
        raise AppInputError(
            f"Could not save uploaded file '{destination.name}'"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def prepare_uploaded_inputs(
    protein_name: str,
    *,
    wt_files_aa: list[UploadedFile] | tuple[UploadedFile, ...] | None = None,
    wt_file_aa: UploadedFile | None = None,
    wt_file_dna: UploadedFile | None = None,
    dataset_file: UploadedFile | None = None,
    mutations_file: UploadedFile | None = None,
    pdb_files: list[UploadedFile] | tuple[UploadedFile, ...] | None = None,
    output_root: str | os.PathLike[str] | None = None,
) -> PreparedInputs:
    """Validate one app submission before persisting any uploaded file."""
    validate_identifier(protein_name, "Protein name")
    if wt_files_aa and wt_file_aa is not None:
        raise AppInputError(
            "Provide either multiple amino acid FASTA files or one FASTA file, not both"
        )

    pending: list[_PendingUpload] = []
    for index, uploaded_file in enumerate(wt_files_aa or ()):
        pending.append(_pending_upload(uploaded_file, role="wt_files_aa", index=index))
    for role, uploaded_file in (
        ("wt_file_aa", wt_file_aa),
        ("wt_file_dna", wt_file_dna),
        ("dataset_file", dataset_file),
        ("mutations_file", mutations_file),
    ):
        if uploaded_file is not None:
            pending.append(_pending_upload(uploaded_file, role=role))
    for index, uploaded_file in enumerate(pdb_files or ()):
        pending.append(_pending_upload(uploaded_file, role="pdb_files", index=index))

    filenames = [item.filename for item in pending]
    if len(filenames) != len(set(filenames)):
        raise AppInputError(
            "Uploaded files in one submission must have unique filenames"
        )

    project = resolve_project_directory(protein_name, output_root=output_root)
    destinations = {
        (item.role, item.index): _upload_destination(project, item.filename)
        for item in pending
    }
    for item in pending:
        _atomic_write(destinations[(item.role, item.index)], item.data)

    return PreparedInputs(
        project_dir=project,
        wt_files_aa=tuple(
            destinations[("wt_files_aa", index)]
            for index in range(len(wt_files_aa or ()))
        ),
        wt_file_aa=destinations.get(("wt_file_aa", 0)),
        wt_file_dna=destinations.get(("wt_file_dna", 0)),
        dataset_file=destinations.get(("dataset_file", 0)),
        mutations_file=destinations.get(("mutations_file", 0)),
        pdb_files=tuple(
            destinations[("pdb_files", index)] for index in range(len(pdb_files or ()))
        ),
    )
