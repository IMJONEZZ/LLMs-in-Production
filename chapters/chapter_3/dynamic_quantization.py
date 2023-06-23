import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Ensure the qnnpack engine is supported
print("Supported engines: ", torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = "qnnpack"

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure the model is using a supported dtype
assert (
    model.dtype == torch.float32
), "Only float32 models are supported for quantization"

# Use PyTorch's quantization library
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

print("Quantization complete!")
