import os
from openai import OpenAI

# Load your API key from an environment variable
client = OpenAI(
    # This isn't technically needed as we are passing in the default key
    api_key=os.getenv("OPENAI_API_KEY")
)

chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}],
)
