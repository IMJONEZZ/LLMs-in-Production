# vanilla tensorrt container images
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
# torch_tensorrt container images
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags


# docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.09-py3
# pip install transformers==4.40.1


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch_tensorrt

# Prepare model and tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer("The cat is on the table.", return_tensors="pt")[
    "input_ids"
].cuda()
model = GPT2LMHeadModel.from_pretrained(
    "gpt2", use_cache=False, return_dict=False, torchscript=True
).cuda()
model.eval()

# Convert to Torchscript IR
traced_model = torch.jit.trace(model, tokens)

# Compile Model with TensorRT
compile_settings = {
    "inputs": [
        torch_tensorrt.Input(
            # For static size
            shape=[1, 7],
            # For dynamic sizing:
            # min_shape=[1, 3],
            # opt_shape=[1, 128],
            # max_shape=[1, 1024],
            dtype=torch.int32,  # Datatype of input tensor.
            # Allowed options torch.(float|half|int8|int32|bool)
        )
    ],
    "truncate_long_and_double": True,
    "enabled_precisions": {torch.half},  # Run with FP16
    "ir": "torchscript",
}
trt_model = torch_tensorrt.compile(traced_model, **compile_settings)

# Save compiled model
torch.jit.save(trt_model, "trt_model.ts")

# Run inference
trt_model = torch.jit.load("trt_model.ts")
tokens.half()
tokens = tokens.type(torch.int)
logits = trt_model(tokens)
results = torch.softmax(logits[-1], dim=-1).argmax(dim=-1)
print(tokenizer.batch_decode(results))
# ['\n was a the way.\n']
