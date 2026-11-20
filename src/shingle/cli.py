"""`shingle` command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from home_display.shingle import config, fetch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shingle", description="Shingle Creek at Campbell stage forecast")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="download/update historical inputs into the data dir")
    p_fetch.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_fetch.add_argument("--refresh", action="store_true", help="ignore cache and refetch everything")

    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        fetch.fetch_history(args.data_dir, refresh=args.refresh)
