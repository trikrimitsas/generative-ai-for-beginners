from openai import OpenAI
import os
import re
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()


def sanitize_prompt_input(value: str, max_length: int = 500) -> str:
    """Validate and sanitize user input before interpolating it into a prompt."""
    if len(value) > max_length:
        raise ValueError(f"Input too long. Maximum {max_length} characters allowed.")
    # Strip characters commonly used in prompt injection / template attacks
    sanitized = re.sub(r'[<>{}[\]|\\`]', '', value).strip()
    if not sanitized:
        raise ValueError("Input cannot be empty")
    return sanitized


# configure OpenAI service client 
client = OpenAI()
deployment="gpt-5-mini"

# add your completion code
try:
    persona = sanitize_prompt_input(input("Tell me the historical character I want to be: "), 200)
    question = sanitize_prompt_input(input("Ask your question about the historical character: "), 500)
except ValueError as e:
    print(f"Input validation error: {e}")
    exit(1)
prompt = f"""
You are going to play as a historical character {persona}. 

Whenever certain questions are asked, you need to remember facts about the timelines and incidents and respond the accurate answer only. Don't create content yourself. If you don't know something, tell that you don't remember.

Provide answer for the question: {question}
"""
# make a request using the Responses API
response = client.responses.create(model=deployment, input=prompt, store=False)

# print response
print(response.output_text)

#  very unhappy _____.

# Once upon a time there was a very unhappy mermaid.