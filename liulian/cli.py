"""Command-line interface for liulian (placeholder for MVP1).

A full CLI with subcommands (``liulian run``, ``liulian validate``, etc.)
is planned for v1+.  This module currently exposes a minimal ``main()``
that prints version info.
"""

from __future__ import annotations

import argparse
import sys

from liulian import __version__


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``liulian`` CLI.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        prog="liulian",
        description="LIULIAN — Research OS for spatiotemporal model experimentation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"liulian {__version__}",
    )
    # Placeholder subcommand — extend in v1+
    parser.add_argument(
        "command",
        nargs="?",
        choices=["info"],
        help="Subcommand to run (MVP1: only 'info' is available).",
    )

    args = parser.parse_args(argv)

    if args.command == "info":
        print(f"liulian {__version__}")
        print("Liquid Intelligence and Unified Logic for Interactive Adaptive Networks")
        print('"Where Space and Time Converge in Intelligence"')
    elif args.command is None:
        parser.print_help()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
