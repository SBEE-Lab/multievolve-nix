"""Reproducibility helpers shared by training and proposal workflows."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def stable_seed(base_seed: int, *parts: str | int) -> int:
    """Derive a stable 32-bit seed from a base seed and identity parts."""
    payload = json.dumps(
        [int(base_seed), *parts],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch before constructing a model."""
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must satisfy 0 <= seed < 2**32")

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        if deterministic:
            torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def resolve_device(requested: str) -> str:
    """Resolve an explicit or automatic PyTorch device without silent fallback."""
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return "cuda:0"
    if requested != "auto":
        raise ValueError(f"unsupported device: {requested}")
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sha256_json(value: Any) -> str:
    """Hash a JSON-serializable value with canonical formatting."""
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: Any) -> str:
    """Hash an array's shape, dtype, and values deterministically."""
    array = np.asarray(values)
    metadata = {"dtype": str(array.dtype), "shape": list(array.shape)}
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if array.dtype.hasobject:
        digest.update(
            json.dumps(
                array.tolist(),
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def source_tree_sha256(root: str | Path) -> str:
    """Hash installed MULTI-evolve Python and sweep-configuration sources."""
    root = Path(root)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".yaml", ".yml"}
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_csv_sha256(path: str | Path, *, header: bool = True) -> str:
    """Hash parsed CSV content while ignoring byte-level formatting differences."""
    import pandas as pd

    frame = pd.read_csv(path, header=0 if header else None, keep_default_na=False)
    canonical = frame.to_csv(index=False, header=header, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_fasta_sha256(path: str | Path) -> str:
    """Hash the uppercase FASTA sequence independently of its header and wrapping."""
    sequence = "".join(
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(">")
    ).upper()
    if not sequence:
        raise ValueError(f"FASTA contains no sequence: {path}")
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON file in its destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity() -> dict[str, Any]:
    """Describe a checkout, including its dirty content, or its installed path."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty_sha256 = None
        if status:
            digest = hashlib.sha256()
            digest.update(
                subprocess.run(
                    ["git", "-C", str(root), "diff", "--binary", "HEAD"],
                    check=True,
                    capture_output=True,
                ).stdout
            )
            untracked = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
                check=True,
                capture_output=True,
            ).stdout.split(b"\0")
            for relative_bytes in sorted(path for path in untracked if path):
                relative = relative_bytes.decode("utf-8")
                digest.update(relative_bytes)
                digest.update(sha256_file(root / relative).encode("ascii"))
            dirty_sha256 = digest.hexdigest()
        return {
            "revision": revision,
            "dirty": bool(status),
            "dirty_sha256": dirty_sha256,
            "location": None,
        }
    except (OSError, subprocess.CalledProcessError):
        package_root = Path(__file__).resolve().parents[1]
        return {
            "revision": None,
            "dirty": None,
            "dirty_sha256": None,
            "location": str(root),
            "content_sha256": source_tree_sha256(package_root),
        }


def runtime_identity(device: str) -> dict[str, Any]:
    """Return runtime versions relevant to numerical reproducibility."""
    import torch

    device_name = None
    if device.startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.device(device))

    torch_version = getattr(torch, "__version__", None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda": cuda_version,
        "cudnn": torch.backends.cudnn.version(),
        "device": device,
        "device_name": device_name,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "executable": sys.executable,
        "source": source_identity(),
    }
