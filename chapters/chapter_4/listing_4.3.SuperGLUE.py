from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# SuperGlue has mutliple test datasets, options are boolq,
# cb, copa, multirc, record, rte, wic, wsc, wsc.fixed, axb, axg
dataset = load_dataset("super_glue", "multirc", split="validation")
print(dataset[0])

model = "bigscience/bloomz-560m"  # Update with your model of choice

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)


for row in dataset:
    # replace this with the correct input for your benchmark
    input_text = (
        f'Paragraph: {row["paragraph"]}\nQuestion: {row["question"]}'
    )
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids, max_new_tokens=20)
    input_length = input_ids.shape[1]  # We use this to trim out the input
    results = tokenizer.decode(outputs[0][input_length:])
    print(row["answer"])
    print(results)
