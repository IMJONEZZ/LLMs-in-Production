from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# SuperGlue has mutliple test datasets, options are boolq,
# cb, copa, multirc, record, rte, wic, wsc, wsc.fixed, axb, axg
dataset = load_dataset("super_glue", "multirc", split="validation")
print(dataset[0])

# {
#   "paragraph": "What causes a change in motion? The application of a force."
#     " Any time an object changes motion, a force has been applied. In what "
#     "ways can this happen? Force can cause an object at rest to start "
#     "moving. Forces can cause objects to speed up or slow down. Forces can "
#     "cause a moving object to stop. Forces can also cause a change in "
#     "direction. In short, forces cause changes in motion. The moving "
#     "object may change its speed, its direction, or both. We know that "
#     "changes in motion require a force. We know that the size of the force "
#     "determines the change in motion. How much an objects motion changes "
#     "when a force is applied depends on two things. It depends on the "
#     "strength of the force. It also depends on the objects mass. Think "
#     "about some simple tasks you may regularly do. You may pick up a "
#     "baseball. This requires only a very small force. ",
#   "question": "Would the mass of a baseball affect how much force you have "
#     "to use to pick it up?",
#   "answer": "No",
#   "idx": {"paragraph": 0, "question": 0, "answer": 0},
#   "label": 0,
# }

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

# No
#  No</s>
# Yes
#  No</s>
# Less the mass, less the force applied
#  No</s>
# It depends on the shape of the baseball
#  No</s>
# Strength
#  Force</s>
# A force
#  Force</s>
# No
#  Yes</s>
