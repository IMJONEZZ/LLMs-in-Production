import openai

import base64
from io import BytesIO
from PIL import Image


def encode_image(image_path, max_image=512):
    with Image.open(image_path) as img:
        width, height = img.size
        max_dim = max(width, height)
        if max_dim > max_image:
            scale_factor = max_image / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height))

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str


client = openai.OpenAI(
    base_url="http://0.0.0.0:1234/v1",  # replace with your server's ip address and port
    api_key="1234",  # replace with your server's api key
)
image_file = "myImage.jpg"
max_size = 512  # set to maximum dimension to allow (512=1 tile, 2048=max)
encoded_string = encode_image(image_file, max_size)

completion = client.chat.completions.with_raw_response.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "system",
            "content": "You are an expert at analyzing images with computer vision. In case of error,\nmake a full report of the cause of: any issues in receiving, understanding, or describing images",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Building a website can be done in 10 simple steps:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_string}"
                    },
                },
            ],
        },
    ],
    max_tokens=500,
)

chat = completion.parse()
print(chat.choices[0].message.content)
