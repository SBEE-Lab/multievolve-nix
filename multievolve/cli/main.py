"""Unified command-line interface for MULTI-evolve."""

from __future__ import annotations

import argparse
import importlib
import sys

_COMMANDS = {
    "train": ("train neural network models", "multievolve.cli.train"),
    "propose": ("propose multi-mutant variants using trained models", "multievolve.cli.propose"),
    "design-oligos": ("generate MULTI-assembly mutagenic oligos", "multievolve.cli.assembly_design"),
    "plm-zeroshot": (
        "nominate mutations with PLM zero-shot ensembles",
        "multievolve.cli.plm_zeroshot_ensemble",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multievolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=sorted(_COMMANDS), help="subcommand to run")
    parser.epilog = "commands:\n" + "\n".join(
        f"  {name:<14} {description}" for name, (description, _) in sorted(_COMMANDS.items())
    )
    return parser


def main(argv: list[str] | None = None) -> object:
    """Dispatch to a MULTI-evolve subcommand."""
    if argv is None:
        argv = sys.argv[1:]

    parser = _parser()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return None

    command, rest = argv[0], argv[1:]
    if command not in _COMMANDS:
        parser.error(f"argument command: invalid choice: {command!r} (choose from {', '.join(sorted(_COMMANDS))})")

    _, module_name = _COMMANDS[command]

    # Let the existing command parser own all command-specific options,
    # including `--help`.
    sys.argv = [f"multievolve {command}", *rest]
    module = importlib.import_module(module_name)
    return module.main()


if __name__ == "__main__":
    main()
