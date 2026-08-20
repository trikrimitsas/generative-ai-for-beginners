"""This script removes the text from the enriched transcript and saves it as a new json file."""

import json
import logging

from transcript_utils import configure_logging, output_path, parse_arguments

logger = configure_logging(__name__, logging.WARNING)
args = parse_arguments(logger)
TRANSCRIPT_FOLDER = args.folder


# load video list from json file
input_file = output_path(TRANSCRIPT_FOLDER, "master_enriched.json")
with open(input_file, encoding="utf-8") as f:
    segments = json.load(f)

total_segments = len(segments)

# create a lambda function to remove the text from each dictionary in the list


def remove_text(video_segments):
    """This function removes the text from each dictionary in the list."""
    return [
        {k: v for k, v in seg.items() if k != "text" and k != "description"}
        for seg in video_segments
    ]


lite = remove_text(segments)

# save the embeddings to a json file
output_file = output_path(TRANSCRIPT_FOLDER, "master_enriched_lite.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(lite, f)
