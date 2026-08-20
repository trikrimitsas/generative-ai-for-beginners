"""Summarize a youtube transcript using chatgpt"""

import json
import logging
import os
import queue
import sys
import threading

import dotenv
from openai import BadRequestError
from rich.progress import Progress
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from transcript_utils import (
    Counter,
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
AZURE_OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini")
MAX_TOKENS = 512
PROCESSOR_THREADS = 10
OPENAI_REQUEST_TIMEOUT = 30
MAX_ERRORS = 100

client = create_azure_openai_client(endpoint=RESOURCE_ENDPOINT, api_key=API_KEY)

logger = configure_logging(__name__, logging.WARNING)

args = parse_arguments(logger)
TRANSCRIPT_FOLDER = args.folder

segments = []
output_segments = []
total_segments = 0


counter = Counter()
error_counter = Counter()
# Set when a worker hits an unrecoverable error so the other threads stop and the
# script exits non-zero instead of writing a partially enriched output file.
abort = threading.Event()


class SummaryError(RuntimeError):
    """Raised when the model response can't be used as a summary."""


@retry(
    wait=wait_random_exponential(min=10, max=45),
    stop=stop_after_attempt(20),
    retry=retry_if_not_exception_type((BadRequestError, SummaryError)),
)
def chatgpt_summary(text):
    """generate a summary using chatgpt"""

    messages = [
        {
            "role": "system",
            "content": "You're an AI Assistant for video, write an authoritative 60 word summary.Avoid starting sentences with 'This video'.",
        },
        {"role": "user", "content": text},
    ]

    response = client.responses.create(
        model=AZURE_OPENAI_MODEL_DEPLOYMENT_NAME,
        input=messages,
        max_output_tokens=MAX_TOKENS,
        timeout=OPENAI_REQUEST_TIMEOUT,
        store=False,
    )

    # print(response)

    text = response.output_text or text
    finish_reason = response.status

    # print(finish_reason)
    if finish_reason != "completed":
        # Raise instead of calling exit(): a SystemExit raised on a worker thread is
        # silently discarded and the script would carry on as if nothing happened.
        raise SummaryError(
            f"Incomplete response (status: {finish_reason}). "
            f"Increase MAX_TOKENS (currently {MAX_TOKENS}) and try again."
        )

    return text


def process_queue(progress, task):
    """process the queue"""
    while not q.empty():
        if abort.is_set():
            return

        segment = q.get()

        text = segment.get("text")

        # Think about this some more. Idea is to reduce processing time
        # text_hash = hash(text)

        # # check if there is a summary already in the segment and the hash is the same
        # # If found then don't generate a new summary
        # if "summary" in segment and "text_hash" in segment and text_hash == segment["text_hash"]:
        #     output_segments.append(segment.copy())
        #     q.task_done()
        #     continue

        # get a summary of the text using chatgpt
        try:
            summary = chatgpt_summary(text)
        except BadRequestError as invalid_request_error:
            # The model rejected this segment, so fall back to the raw text, but keep
            # count so a run that fails for most segments doesn't look successful.
            logger.warning(
                "Segment rejected by the model, falling back to the raw text: %s",
                invalid_request_error,
            )
            summary = text
            if error_counter.increment() > MAX_ERRORS:
                logger.error("More than %d segments failed. Aborting...", MAX_ERRORS)
                abort.set()
                q.task_done()
                return
        except Exception:
            # Anything else (auth, network, quota, a bad response) is unrecoverable
            # for this run - report it with a traceback and stop the whole script.
            logger.exception("Unexpected error while summarizing a segment. Aborting...")
            abort.set()
            q.task_done()
            return

        count = counter.increment()
        progress.update(task, advance=1)
        logger.debug("Processed %d segments of %d", count, total_segments)

        # add the summary and text hash to the segment dictionary
        segment["summary"] = summary

        output_segments.append(segment.copy())
        q.task_done()


logger.debug("Starting OpenAI summarization")

# load the segments from a json file
input_file = output_path(TRANSCRIPT_FOLDER, "master_transcriptions.json")
with open(input_file, encoding="utf-8") as f:
    segments = json.load(f)

total_segments = len(segments)

logger.debug("Total segments to be processed: %s", len(segments))

# add segment list to a queue
q = queue.Queue()
for segment in segments:
    q.put(segment)

with Progress() as progress:
    task1 = progress.add_task("[purple]Enriching Summaries...", total=total_segments)

    run_worker_threads(process_queue, PROCESSOR_THREADS, args=(progress, task1))

if abort.is_set():
    logger.error("Summarization failed, the enriched output file was not written")
    sys.exit(1)


# sort the output segments by videoId and start
output_segments.sort(key=lambda x: (x["videoId"], convert_time_to_seconds(x["start"])))

logger.debug("Total segments processed: %s", len(output_segments))

# save the output segments to a json file
output_file = output_path(TRANSCRIPT_FOLDER, "master_enriched.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_segments, f, ensure_ascii=False, indent=4)
