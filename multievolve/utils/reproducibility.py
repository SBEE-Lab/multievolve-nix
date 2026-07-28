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


def source_identity() -> dict[str, Any]:
    """Describe the source checkout when available."""
    try:
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


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
