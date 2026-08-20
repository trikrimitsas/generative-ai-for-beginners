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


# configure the OpenAI client against the Azure OpenAI (Microsoft Foundry) v1 endpoint
client = OpenAI(
  api_key=os.environ['AZURE_OPENAI_API_KEY'],  
  base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/",
  )

deployment=os.environ['AZURE_OPENAI_DEPLOYMENT']

# add your completion code
try:
    question = sanitize_prompt_input(input("Ask your questions on python language to your study buddy: "), 500)
except ValueError as e:
    print(f"Input validation error: {e}")
    exit(1)
prompt = f"""
You are an expert on the python language.

Whenever certain questions are asked, you need to provide response in below format.

- Concept
- Example code showing the concept implementation
- explanation of the example and how the concept is done for the user to understand better.

Provide answer for the question: {question}
"""
# make a request using the Responses API
response = client.responses.create(model=deployment, input=prompt, store=False)

# print response
print(response.output_text)

#  very unhappy _____.

# Once upon a time there was a very unhappy mermaid.
