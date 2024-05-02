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
print(chat_completion)

# ChatCompletion(id='chatcmpl-sOXJ9vt92vStdAqwOxSASONPSdAUg6yN', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=' Hello! How can I help you today? ', role='assistant', function_call=None, tool_calls=None))], created=1714527205, model='gpt-3.5-turbo', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=11, prompt_tokens=10, total_tokens=21))
