"""Tests for the transcript preparation helpers."""

import logging
import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "08-building-search-applications" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from transcript_utils import (  # noqa: E402
    clean_text,
    configure_logging,
    convert_time_to_seconds,
    output_path,
    parse_arguments,
)


def test_clean_text_removes_transcript_artifacts():
    value = "Hello\nthere &#39; >>  world [inaudible]"

    assert clean_text(value) == "Hello there '  world "


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:01:20", 80),
        ("01:02:03", 3723),
        ("80", 0),
    ],
)
def test_convert_time_to_seconds(value, expected):
    assert convert_time_to_seconds(value) == expected


def test_parse_arguments_returns_common_options():
    logger = logging.getLogger("transcript-utils-test")

    args = parse_arguments(logger, argv=["-f", "transcripts", "--verbose"])

    assert args.folder == "transcripts"
    assert args.verbose is True


def test_parse_arguments_supports_playlist_and_minutes():
    logger = logging.getLogger("transcript-utils-test")

    args = parse_arguments(
        logger,
        include_playlist=True,
        include_minutes=True,
        argv=["-f", "transcripts", "-p", "playlist-id", "-m", "7"],
    )

    assert args.playlist == "playlist-id"
    assert args.minutes == "7"


def test_parse_arguments_missing_folder_exits(caplog):
    logger = configure_logging("transcript-utils-missing-folder", logging.WARNING)

    with caplog.at_level(logging.ERROR, logger=logger.name), pytest.raises(SystemExit) as exc:
        parse_arguments(logger, argv=[])

    assert exc.value.code == 1
    assert "Transcript folder not provided" in caplog.text


def test_parse_arguments_missing_playlist_exits(caplog):
    logger = configure_logging("transcript-utils-missing-playlist", logging.WARNING)

    with caplog.at_level(logging.ERROR, logger=logger.name), pytest.raises(SystemExit) as exc:
        parse_arguments(logger, include_playlist=True, argv=["-f", "transcripts"])

    assert exc.value.code == 1
    assert "Playlist ID not provided" in caplog.text


def test_output_path_uses_output_directory():
    assert output_path("transcripts", "master.json") == os.path.join(
        "transcripts", "output", "master.json"
    )
