from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# Load and format the dataset
dataset = load_dataset("text", data_files="./data/crimeandpunishment.txt")
dataset = dataset.filter(lambda sentence: len(sentence["text"]) > 1)
print(dataset["train"][0])

# Establish our GPT2 parameters (different from the paper and scratchGPT)
config = GPT2Config(
    vocab_size=50261,
    n_positions=256,
    n_embd=768,
    activation_function="gelu",
)

# Instantiate our tokenizer and our special tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens_dict = {
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "pad_token": "<PAD>",
    "mask_token": "<MASK>",
}
tokenizer.add_special_tokens(special_tokens_dict)

# Instantiate our model from the config
model = GPT2LMHeadModel.from_pretrained(
    "gpt2", config=config, ignore_mismatched_sizes=True
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
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)  # Masked Language Modeling - adds <MASK> tokens for the model to guess the words

# Establish training arguments
train_args = TrainingArguments(
    output_dir="./chapters/chapter_4/models/betterGPT/",
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
trainer.save_model("./chapters/chapter_4/models/betterGPT/")
tokenizer.save_pretrained()

# Load the saved model
model = GPT2LMHeadModel.from_pretrained(
    "./chapters/chapter_4/models/betterGPT/"
)

# Test the saved model
input = "To be or not"
tokenized_inputs = tokenizer(input, return_tensors="pt")
out = model.generate(
    input_ids=tokenized_inputs["input_ids"],
    attention_mask=tokenized_inputs["attention_mask"],
    max_length=256,
    num_beams=5,
    temperature=0.7,
    top_k=50,
    top_p=0.90,
    no_repeat_ngram_size=2,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
