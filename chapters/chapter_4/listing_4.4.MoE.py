from transformers import (
    AutoTokenizer,
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import torch

# Load and format the dataset
dataset = load_dataset("text", data_files="./data/crimeandpunishment.txt")
dataset = dataset.filter(lambda sentence: len(sentence["text"]) > 1)
print(f"Dataset 1: {dataset['train'][0]}")

# Instantiate our tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

# Establish our SwitchTransformers config
config = SwitchTransformersConfig(
    decoder_start_token_id=tokenizer.pad_token_id
)

# Instantiate our model from the config
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    "google/switch-base-8",
    config=config,
    device_map="auto",
    torch_dtype=torch.float16,
)


# Create a tokenize function
def tokenize(batch):
    return tokenizer(
        str(batch), padding="max_length", truncation=True, max_length=256
    )


# tokenize our whole dataset (so we never have to do it again)
tokenized_dataset = dataset.map(tokenize, batched=False)
print(f"Tokenized: {tokenized_dataset['train'][0]}")

# Create a data collator to format the data for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.0
)  # Causal Language Modeling - Does not use mask


# Establish training arguments
train_args = TrainingArguments(
    output_dir="./chapters/chapter_4/models/MoE/",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=5000,
    save_total_limit=2,
    report_to="none",
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

# Train and save the model
trainer.train()
trainer.save_model("./chapters/chapter_4/models/MoE/")
tokenizer.save_pretrained("./chapters/chapter_4/models/MoE/")

# Load the saved model
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    "./chapters/chapter_4/models/MoE/",
    device_map="auto",
    torch_dtype=torch.float16,
)

# Test the saved model
input = "To be or not <extra_id_0> <extra_id_0>"
tokenized_inputs = tokenizer(input, return_tensors="pt")
out = model.generate(
    input_ids=tokenized_inputs["input_ids"].to("cuda"),
    attention_mask=tokenized_inputs["attention_mask"],
    max_length=256,
    num_beams=5,
    temperature=0.7,
    top_k=50,
    top_p=0.90,
    no_repeat_ngram_size=2,
)
print(f"To be or not {tokenizer.decode(out[0], skip_special_tokens=True)}")
