from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
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
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

