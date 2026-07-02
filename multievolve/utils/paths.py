"""Filesystem path helpers for installed and source checkouts."""

import os


def get_output_root() -> str:
    """Writable root for runtime outputs.

    MULTI-evolve writes datasets, split caches, feature caches, models, and
    proposal results under ``proteins/``. In an installed package, deriving this
    root from ``__file__`` points at the read-only install directory. Default to
    the caller's working directory instead, with ``MULTIEVOLVE_ROOT`` as an
    explicit override.
    """
    return os.environ.get("MULTIEVOLVE_ROOT", os.getcwd())
