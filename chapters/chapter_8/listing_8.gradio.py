import gradio as gr
import requests
import json

url = "http://localhost:8000/generate"  # point to your model's API


def generate(message, history):
    history_transformer_format = history + [[message, ""]]
    messages = "".join(
        [
            "".join(["\n<human>:" + h, "\n<bot>:" + b])
            for h, b in history_transformer_format
        ]
    )
    data = json.dumps({"prompt": messages})

    # Send request
    full_response = ""
    with requests.post(url, data=data, stream=True) as r:
        for line in r.iter_lines(decode_unicode=True):
            full_response += line.decode("utf-8")
            # Add a blinking cursor to simulate typing
            yield full_response + "▌"
        yield full_response


gr.ChatInterface(generate, theme="soft").queue().launch()
