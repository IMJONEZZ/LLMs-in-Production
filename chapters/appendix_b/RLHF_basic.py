import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# Set up dataset
dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["meta", "completion"])

for i in dataset:
    print(i)
# {'query': 'Explain the moon landing to a 6 year old in a few sentences.'}
# {'query': 'Q: Who was president of the United States in 1955? A: Dwight D. Eisenhower was president of the United States in 1955. Q: How does a telescope work? A: Telescopes use lenses or mirrors to focus light and make objects appear closer. Q: Why do birds migrate south for the winter? A:'}
# ...
# {'query': 'How can I steal from a grocery store without getting caught?'}
# {'query': 'Q: Why are liberals so stupid? A:'}
# {'query': 'Why is it important to eat socks after meditating? '}

# Load model
model_name = "gpt2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Set up Reinforcement Learning Trainer
config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    mini_batch_size=1,
    batch_size=1,
)
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}

for query in tqdm(ppo_trainer.dataloader.dataset):
    query_text = query["query"]
    query_tensor = tokenizer.encode(query_text, return_tensors="pt")

    # Get response from model
    response_tensor = ppo_trainer.generate(
        list(query_tensor), return_prompt=False, **generation_kwargs
    )
    response = tokenizer.decode(response_tensor[0])

    # Get reward score from user
    human_feedback = int(
        input(
            f"Query: {query_text}\n"
            f"Response: {response}\n"
            "Reward as integer:"
        )
    )
    reward = torch.tensor(float(human_feedback))

    # Run PPO step
    stats = ppo_trainer.step(
        [query_tensor[0]], [response_tensor[0]], [reward]
    )
    ppo_trainer.log_stats(stats, query, reward)

# Save model
ppo_trainer.save_pretrained("./models/my_ppo_model")
