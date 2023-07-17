from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    LMSDiscreteScheduler,
)
from torch import autocast
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np

from tqdm.auto import tqdm

# Easiest
access_token = "Your HF Access Token"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", use_auth_token=access_token
).to(
    "cuda"
)  # use revision='fp16' and torch_dtype=torch.float16 for low memory

prompt = "a photo of a horse riding an astronaut on Mars"
image = pipe(prompt).images[0]
image.save("./chapters/chapter_4/images/horse_rides_astronaut.png")


# Medium
def dummy(images, **kwargs):
    return images, False


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


pipe.safety_checker = dummy
n_images = 3
prompts = [
    "masterpiece, best quality, a photo of a horse riding an astronaut, trending on artstation, photorealistic, qhd, rtx on, 8k"
] * n_images
with autocast("cuda"):
    images = pipe(prompts, num_inference_steps=28).images
image_grid(images, rows=1, cols=3)
i = 1
for image in images:
    image.save(f"./chapters/chapter_4/images/{prompts[0][27:40] + i}.png")


# Custom
# vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder='vae', use_auth_token=access_token)
# vae.save_pretrained('./models/vae')
vae = AutoencoderKL.from_pretrained("./models/vae/").to("cuda")

# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer.save_pretrained('./tokenizers/')
tokenizer = CLIPTokenizer.from_pretrained("./tokenizers/")
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
# text_encoder.save_pretrained('./models/text_encoder')
text_encoder = CLIPTextModel.from_pretrained("./models/text_encoder/").to(
    "cuda"
)

# model = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder='unet', use_auth_token=access_token).to("cuda")
# model.save_pretrained('./models/sd_v1-5')
model = UNet2DConditionModel.from_pretrained("./models/sd_v1-5/").to("cuda")

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)


def get_text_embeds(prompt):
    # Tokenize text and get embeddings
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]

    # Do the same for unconditional embeddings
    uncond_input = tokenizer(
        [""] * len(prompt),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[
            0
        ]

    # Cat for final embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


test_embeds = get_text_embeds(["an amazingly cool anime character"])
print(test_embeds)
print(test_embeds.shape)


def produce_latents(
    text_embeddings,
    height=512,
    width=512,
    num_inference_steps=28,
    guidance_scale=11,
    latents=None,
    return_all_latents=False,
):
    if latents is None:
        latents = torch.randn(
            (
                text_embeddings.shape[0] // 2,
                model.in_channels,
                height // 8,
                width // 8,
            )
        )
    latents = latents.to("cuda")

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0]

    latent_hist = [latents]
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / (
                (sigma**2 + 1) ** 0.5
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred = model(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                )["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
            latent_hist.append(latents)

    if not return_all_latents:
        return latents

    all_latents = torch.cat(latent_hist, dim=0)
    return all_latents


test_latents = produce_latents(test_embeds)
print(test_latents)
print(test_latents.shape)


def decode_img_latents(latents):
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        imgs = vae.decode(latents)["sample"]

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1)
    imgs = (imgs + 1.0) * 127.5
    imgs = imgs.numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images


imgs = decode_img_latents(test_latents)


def prompt_to_img(
    prompts,
    height=512,
    width=512,
    num_inference_steps=28,
    guidance_scale=11,
    latents=None,
):
    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeddings
    text_embeds = get_text_embeds(prompts)

    # Text embeddings -> img latents
    latents = produce_latents(
        text_embeds,
        height=height,
        width=width,
        latents=latents,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # Img latents -> imgs
    imgs = decode_img_latents(latents)

    return imgs


imgs = prompt_to_img(
    ["Super cool fantasty knight, intricate armor, 8k"] * 4,
    512,
    512,
    28,
    11,
)

image_grid(imgs, rows=2, cols=2)
