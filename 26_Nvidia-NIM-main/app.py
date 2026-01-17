import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

completion = client.chat.completions.create(
  model="openai/gpt-oss-20b",
  messages=[{"content":"hi ","role":"user"}],
  temperature=1,
  top_p=1,
  max_tokens=4096,
  stream=True
)

for chunk in completion:
    if not chunk.choices:
        continue  # skip empty chunks safely

    delta = chunk.choices[0].delta

    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        print(reasoning, end="")

    if delta.content:
        print(delta.content, end="")


