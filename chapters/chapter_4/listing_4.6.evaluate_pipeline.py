import torch
from transformers import pipeline
from datasets import Dataset, load_dataset
from evaluate import evaluator
import evaluate
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pull model, data, and metrics
pipe = pipeline(
    "text-generation", model="gpt2", pad_token_id=50256, device=device
)
wino_bias = load_dataset("sasha/wino_bias_prompt1", split="test")
polarity = evaluate.load("regard")
task_evaluator = evaluator("text-generation")


# Prepare dataset
def prepare_dataset(wino_bias, pronoun):
    data = wino_bias.filter(
        lambda example: example["bias_pronoun"] == pronoun
    ).shuffle()
    df = data.to_pandas()
    df["prompts"] = df["prompt_phrase"] + " " + df["bias_pronoun"]
    return Dataset.from_pandas(df)


female_prompts = prepare_dataset(wino_bias, "she")
male_prompts = prepare_dataset(wino_bias, "he")

# Run through evaluation pipeline
female_results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=female_prompts,
    input_column="prompts",
    metric=polarity,
)
male_results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=male_prompts,
    input_column="prompts",
    metric=polarity,
)


# Analyze results
def flatten_results(results):
    flattened_results = []
    for result in results["regard"]:
        item_dict = {}
        for item in result:
            item_dict[item["label"]] = item["score"]
        flattened_results.append(item_dict)

    return pd.DataFrame(flattened_results)


# Print the mean polarity scores
print(flatten_results(female_results).mean())
# positive    0.129005
# negative    0.391423
# neutral     0.331425
# other       0.148147
print(flatten_results(male_results).mean())
# positive    0.118647
# negative    0.406649
# neutral     0.322766
# other       0.151938
