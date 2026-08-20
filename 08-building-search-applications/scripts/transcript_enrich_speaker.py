"""This script will get the speaker name from the YouTube video metadata and the first minute of the transcript using the OpenAI Functions entity extraction."""

import glob
import json
import logging
import os
import queue
import time

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
    clean_text,
    configure_logging,
    create_azure_openai_client,
    parse_arguments,
    run_worker_threads,
)

logger = configure_logging(__name__, logging.WARNING)

# import dotenv
dotenv.load_dotenv()

API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
RESOURCE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
TRANSCRIPT_FOLDER = "transcripts"
PROCESSING_THREADS = 10
SEGMENT_MIN_LENGTH_MINUTES = 3
OPENAI_REQUEST_TIMEOUT = 60

OPENAI_MAX_TOKENS = 512
AZURE_OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini")


client = create_azure_openai_client(endpoint=RESOURCE_ENDPOINT, api_key=API_KEY)

args = parse_arguments(logger)
TRANSCRIPT_FOLDER = args.folder

get_speaker_name = {
    "name": "get_speaker_name",
    "description": "Get the speaker names for the session.",
    "parameters": {
        "type": "object",
        "properties": {
            "speakers": {
                "type": "string",
                "description": "The speaker names.",
            }
        },
        "required": ["speaker_name"],
    },
}


# Responses API uses a flat tool format (name/description/parameters at the top level)
openai_functions = [{"type": "function", **get_speaker_name}]


# these maps are used to make the function name string to the function call
definition_map = {"get_speaker_name": get_speaker_name}

q = queue.Queue()

errors = 0


@retry(
    wait=wait_random_exponential(min=6, max=10),
    stop=stop_after_attempt(4),
    retry=retry_if_not_exception_type(BadRequestError),
)
def get_speaker_info(text):
    """Gets the OpenAI functions from the text."""

    function_name = None
    arguments = None

    response_1 = client.responses.create(
        model=AZURE_OPENAI_MODEL_DEPLOYMENT_NAME,
        input=[
            {
                "role": "system",
                "content": "You are an AI assistant that can extract speaker names from text as a list of comma separated names. Try and extract the speaker names from the title. Speaker names are usually less than 3 words long.",
            },
            {"role": "user", "content": text},
        ],
        tools=openai_functions,
        max_output_tokens=OPENAI_MAX_TOKENS,
        timeout=OPENAI_REQUEST_TIMEOUT,
        tool_choice={"type": "function", "name": "get_speaker_name"},
        store=False,
    )

    # The model's response includes a function call. We extract the arguments from it.
    tool_calls = [item for item in response_1.output if item.type == "function_call"]

    if tool_calls:
        function_name = tool_calls[0].name
        arguments = json.loads(tool_calls[0].arguments)

    return function_name, arguments


def get_first_segment(file_name):
    """Gets the first segment from the filename"""

    text = ""
    current_seconds = None
    segment_begin_seconds = None
    segment_finish_seconds = None

    vtt = file_name.replace(".json", ".json.vtt")

    with open(vtt, encoding="utf-8") as json_file:
        json_vtt = json.load(json_file)

        for segment in json_vtt:
            current_seconds = segment.get("start")

            if segment_begin_seconds is None:
                segment_begin_seconds = current_seconds
                # calculate the finish time from the segment_begin_time
                segment_finish_seconds = segment_begin_seconds + SEGMENT_MIN_LENGTH_MINUTES * 60

            if current_seconds < segment_finish_seconds:
                # add the text to the transcript
                text += clean_text(segment.get("text")) + " "

    return text


def process_queue(progress, task):
    """process the queue"""
    while not q.empty():
        filename = q.get()
        progress.update(task, advance=1)
        if errors > 100:
            logger.error("Too many errors. Exiting...")
            exit(1)

        with open(filename, encoding="utf-8") as json_file:
            metadata = json.load(json_file)

            base_text = (
                "The title is: "
                + metadata["title"]
                + " "
                + metadata["description"]
                + " "
                + get_first_segment(filename)
            )
            # replace new line with empty string
            base_text = base_text.replace("\n", " ")

            function_name, arguments = get_speaker_info(base_text)
            speakers = arguments.get("speakers", "")
            if speakers == "":
                print(f"From function call: {filename}\t---MISSING SPEAKER---")
                continue
            else:
                print(f"From function call: {filename}\t{speakers}")

            metadata["speaker"] = speakers
            # SECURITY: Use context manager to properly close file handles
            with open(filename, "w", encoding="utf-8") as out_file:
                json.dump(metadata, out_file)

        q.task_done()
        time.sleep(0.2)


logger.debug("Transcription folder %s", TRANSCRIPT_FOLDER)
logger.debug("Starting Speaker Update")

# load all the transcript json files into the queue
folder = os.path.join(TRANSCRIPT_FOLDER, "*.json")

for filename in glob.glob(folder):
    # load the json file
    q.put(filename)


logger.debug("Starting speaker name update. Files to be processed: %s", q.qsize())
start_time = time.time()
with Progress() as progress:
    task1 = progress.add_task("[blue]Enriching Speaker Data...", total=q.qsize())
    run_worker_threads(process_queue, PROCESSING_THREADS, args=(progress, task1))

finish_time = time.time()
logger.debug("Finished speaker name update. Total time taken: %s", finish_time - start_time)
