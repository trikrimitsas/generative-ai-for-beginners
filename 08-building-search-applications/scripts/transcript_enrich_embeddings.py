"""This script will take a text column and create embeddings for each text using the OpenAI API."""

import json
import logging
import os
import queue
import re
import time

import dotenv
import tiktoken
from openai import BadRequestError
from rich.progress import Progress
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from transcript_utils import (
    configure_logging,
    convert_time_to_seconds,
    create_azure_openai_client,
    output_path,
    parse_arguments,
    run_worker_threads,
)

# import dotenv
dotenv.load_dotenv()

API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
RESOURCE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
EMBEDDINGS_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002"
)
PROCESSING_THREADS = 6
OPENAI_REQUEST_TIMEOUT = 60

client = create_azure_openai_client(
    endpoint=RESOURCE_ENDPOINT,
    api_key=API_KEY,
    api_version="2024-10-21",
)

logger = configure_logging(__name__, logging.WARNING)

args = parse_arguments(logger)
TRANSCRIPT_FOLDER = args.folder

tokenizer = tiktoken.get_encoding("cl100k_base")

total_segments = 0
current_segment = 0
output_segments = []


logger.debug("Starting OpenAI Embeddings")


# load sessions_list from json file
input_file = output_path(TRANSCRIPT_FOLDER, "master_enriched.json")
with open(input_file, encoding="utf-8") as f:
    segments = json.load(f)

total_segments = len(segments)


def normalize_text(s, sep_token=" \n "):  # noqa: S107
    """normalize text by removing extra spaces and newlines"""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r". ,", "", s)
    # remove all instances of multiple spaces
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()

    return s


@retry(
    wait=wait_random_exponential(min=6, max=30),
    stop=stop_after_attempt(20),
    retry=retry_if_not_exception_type(BadRequestError),
)
def get_text_embedding(text: str):
    """get the embedding for a text"""

    response = client.embeddings.create(
        model=EMBEDDINGS_DEPLOYMENT_NAME, input=text, timeout=OPENAI_REQUEST_TIMEOUT
    )
    return response.data[0].embedding


def process_queue(progress, task):
    """process the queue"""
    while not q.empty():
        segment = q.get()

        if "ada_v2" in segment:
            output_segments.append(segment.copy())
            continue

        logger.debug(segment["title"])
        text = segment["text"]

        if len(tokenizer.encode(text)) > 8191:
            continue

        text = normalize_text(text)
        segment["text"] = text

        embedding = get_text_embedding(text)
        if embedding is None:
            output_segments.append(segment.copy())
            continue

        segment["ada_v2"] = embedding.copy()

        output_segments.append(segment.copy())
        progress.update(task, advance=1)
        q.task_done()
        time.sleep(0.2)


logger.debug("Total segments to be processed: %s", len(segments))

# add segment list to a queue
q = queue.Queue()
for segment in segments:
    q.put(segment)

with Progress() as progress:
    task1 = progress.add_task("[green]Enriching Embeddings...", total=total_segments)
    run_worker_threads(process_queue, PROCESSING_THREADS, args=(progress, task1))


# sort the output segments by videoId and start
output_segments.sort(key=lambda x: (x["videoId"], convert_time_to_seconds(x["start"])))

logger.debug("Total segments processed: %s", len(output_segments))

# save the embeddings to a json file
output_file = output_path(TRANSCRIPT_FOLDER, "master_enriched.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_segments, f)
