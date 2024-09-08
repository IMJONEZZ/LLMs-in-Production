from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
from time import perf_counter

model_id = "nvidia/Llama3-ChatQA-1.5-8B"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
    torch_dtype=torch.float16,
)

# Traditional Generation
system = (
    "This is a chat between a user and an artificial intelligence "
    "assistant. The assistant gives helpful, detailed, and polite answers "
    "to the user's questions based on the context. The assistant should "
    "also indicate when the answer cannot be found in the context."
)
question = (
    "Please give a full and complete answer for the question. "
    "Can you help me find a place to eat?"
)
response = (
    "Sure, there are many locations near you that are wonderful "
    "to eat at, have you tried La Dolce Vite?"
)
question_2 = (
    "Please give a full and complete answer for the question. "
    "I'm looking for somewhere near me that serves noodles."
)


prompt = f"""System: {system}

User: {question}

Assistant: {response}

User: {question_2}

Assistant:"""
start = perf_counter()
inputs = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt").to(
    device
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
text_outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=128,
    eos_token_id=terminators,
)
response = text_outputs[0][inputs.input_ids.shape[-1] :]
end = perf_counter() - start
print(
    f"\n\nFull Response: {tokenizer.batch_decode(text_outputs)}"
    f"\n\nOnly Answer Response: {tokenizer.decode(response)}"
)
print(f"\nTime to execute: {end}\n")

start = perf_counter()
# Embedding
with torch.no_grad():
    hidden_outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
    )

    embeddings_to_cache = hidden_outputs.hidden_states[-1]

end = perf_counter() - start
print(f"Embeddings: {embeddings_to_cache}")
print(f"\nTime to execute: {end}\n")

# Finding the LM Head layer
for key, module in model._modules.items():
    if key == "lm_head":
        print(f"This is the layer to pass to by itself:\n{module}")
with torch.no_grad():
    start = perf_counter()
    outputs = model._modules["lm_head"](embeddings_to_cache)
    end = perf_counter() - start
    print(f"Outputs: {outputs}")
    print(f"\nTime to execute: {end}\n")


# Custom Trainable classifier
class CustomLinearClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(CustomLinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(0.1)
        self.ff = torch.nn.Linear(4096, num_labels, dtype=torch.float16)

    def forward(self, input_ids=None, targets=None):
        sequence = self.dropout(input_ids)

        logits = self.ff(sequence[:, 0, :].view(-1, 4096))

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.num_labels), targets.view(-1)
            )
            return logits, loss
        else:
            return logits


custom_LMHead = CustomLinearClassifier(128256).to(device)

with torch.no_grad():
    start = perf_counter()
    outputs = custom_LMHead(embeddings_to_cache)
    end = perf_counter() - start
    print(f"Outputs: {outputs}")
    print(f"\nTime to execute: {end}\n")
