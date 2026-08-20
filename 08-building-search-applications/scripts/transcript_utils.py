"""Shared helpers for the transcript preparation scripts."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.python import (  # noqa: E402
    Counter,
    create_azure_openai_client,
    run_worker_threads,
)


def configure_logging(name: str, level: int) -> logging.Logger:
    """Configure logging and return a logger with the given name."""
    logging.basicConfig(level=level)
    return logging.getLogger(name)


def build_parser(
    *,
    include_playlist: bool = False,
    include_minutes: bool = False,
) -> argparse.ArgumentParser:
    """Build the common transcript script argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder")
    parser.add_argument("--verbose", action="store_true")
    if include_playlist:
        parser.add_argument("-p", "--playlist")
    if include_minutes:
        parser.add_argument("-m", "--minutes")
    return parser


def parse_arguments(
    logger: logging.Logger,
    *,
    include_playlist: bool = False,
    include_minutes: bool = False,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse and validate common transcript script arguments."""
    args = build_parser(
        include_playlist=include_playlist,
        include_minutes=include_minutes,
    ).parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.folder:
        logger.error("Transcript folder not provided")
        raise SystemExit(1)

    if include_playlist and not args.playlist:
        logger.error("Playlist ID not provided")
        raise SystemExit(1)

    return args


def clean_text(text: str) -> str:
    """clean the text"""
    text = text.replace("\n", " ")  # remove new lines
    text = text.replace("&#39;", "'")
    text = text.replace(">>", "")  # remove '>>'
    text = text.replace("  ", " ")  # remove double spaces
    text = text.replace("[inaudible]", "")  # [inaudible]

    return text


def convert_time_to_seconds(value: str) -> int:
    """convert time to seconds"""
    time_value = value.split(":")
    if len(time_value) == 3:
        h, m, s = time_value
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
        return 0


def output_path(folder: str, name: str) -> str:
    """Return a path in the transcript output directory."""
    return os.path.join(folder, "output", name)


__all__ = [
    "Counter",
    "build_parser",
    "clean_text",
    "configure_logging",
    "convert_time_to_seconds",
    "create_azure_openai_client",
    "output_path",
    "parse_arguments",
    "run_worker_threads",
]
