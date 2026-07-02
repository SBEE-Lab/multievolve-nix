"""Console entry point for launching the MULTI-evolve Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run ``streamlit run`` against the packaged app module."""
    from streamlit.web import cli as streamlit_cli

    import multievolve.app

    app_path = Path(multievolve.app.__file__).resolve()
    extra_args = sys.argv[1:]
    sys.argv = ["streamlit", "run", str(app_path), *extra_args]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
