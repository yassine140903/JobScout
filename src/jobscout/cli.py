"""CLI entry point: `jobscout init`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobscout.config import load_config
from jobscout.db import init_db, insert_jobs_bulk, table_counts
from jobscout.fixtures import get_fixtures


def cmd_init(args: argparse.Namespace) -> None:
    """Create the database, load fixtures, and print a summary."""
    config = load_config(Path(args.config))
    db_path = Path(config["db_path"])

    print(f"Initializing database at {db_path} ...")
    conn = init_db(db_path)

    fixtures = get_fixtures()
    inserted = insert_jobs_bulk(conn, fixtures)
    print(f"Loaded {inserted} fixture postings ({len(fixtures)} total, duplicates skipped).")

    counts = table_counts(conn)
    for table, count in counts.items():
        print(f"  {table}: {count} rows")

    conn.close()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jobscout",
        description="Local-first, CV-driven job discovery tool.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Create database and load fixtures")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()