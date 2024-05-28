from trl import SFTTrainer
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from transformers import BitsAndBytesConfig
import torch
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/Jamba-v0.1", device_map="auto"
)

dataset = load_dataset("Abirate/english_quotes", split="train")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-3,
)
lora_config = LoraConfig(
    r=8,
    target_modules=["embed_tokens", "x_proj", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)

trainer.train()

tokenizer.save_pretrained("./JAMBA/")
model.save_pretrained("./JAMBA/")

del model

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True, llm_int8_skip_modules=["mamba"]
)
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/Jamba-v0.1",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
)
input_ids = tokenizer(
    "In the recent Super Bowl LVIII,", return_tensors="pt"
).to(model.device)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=216)

print(tokenizer.batch_decode(outputs))
