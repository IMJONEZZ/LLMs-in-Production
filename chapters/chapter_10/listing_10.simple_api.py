import argparse
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import torch
import uvicorn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from threading import Thread


# Torch settings
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define stopping behavior
stop_tokens = ["def", "class", "Instruction", "Output"]
stop_token_ids = [589, 823, 9597, 2301]


class StopOnTokens(StoppingCriteria):
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        stop_ids = stop_token_ids
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("Deci/DeciCoder-1b")
tokenizer.add_special_tokens(
    {"additional_special_tokens": stop_tokens},
    replace_additional_special_tokens=False,
)
model = AutoModelForCausalLM.from_pretrained(
    "Deci/DeciCoder-1b", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model = model.to(device)


# Define streamer
streamer = TextIteratorStreamer(
    tokenizer, timeout=10, skip_prompt=True, skip_special_tokens=True
)


async def stream_results() -> AsyncGenerator[bytes, None]:
    for response in streamer:
        # It's typical to return streamed responses byte encoded
        yield (response).encode("utf-8")


# Run FastAPI
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate LLM Response

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    # RAG will go here
    # ...

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=1024,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )

    # Start a seperate thread to generate results async in stream
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return StreamingResponse(stream_results())


if __name__ == "__main__":
    # Start Service - Defaults to localhost on port 8000
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")
