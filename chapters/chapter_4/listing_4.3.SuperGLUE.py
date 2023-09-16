# pip install jiant
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

dataset = load_dataset(
    "super_glue"
)  # Options are boolq, cb, copa, multirc, record, rte, wic, wsc, wsc.fixed, axb, axg
print(dataset[0])

tokenizer = AutoTokenizer.from_pretrained("<ModelYouWant>")
model = AutoModel.from_pretrained("<ModelYouWant>")


for row in dataset:
    input_text = row[
        "text"
    ]  # replace this with the correct input for your task
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))
