import os
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
)
import evaluate
import torch
import numpy as np

model_checkpoint = "roberta-large"
lr = 1e-3
batch_size = 16
num_epochs = 10

# Create model directory to save to
model_dir = "./models/LoRAPEFT"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

bionlp = load_dataset("tner/bionlp2004")

seqeval = evaluate.load("seqeval")

label_list = [
    "O",
    "B-DNA",
    "I-DNA",
    "B-protein",
    "I-protein",
    "B-cell_type",
    "I-cell_type",
    "B-cell_line",
    "I-cell_line",
    "B-RNA",
    "I-RNA",
]


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions, references=true_labels
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# This is a private repo you must be logged in for
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, add_prefix_space=True
    )
except OSError:
    import subprocess

    subprocess.run(["huggingface-cli", "login"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, add_prefix_space=True, use_auth_token=True
    )


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_bionlp = bionlp.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

id2label = {
    0: "O",
    1: "B-DNA",
    2: "I-DNA",
    3: "B-protein",
    4: "I-protein",
    5: "B-cell_type",
    6: "I-cell_type",
    7: "B-cell_line",
    8: "I-cell_line",
    9: "B-RNA",
    10: "I-RNA",
}
label2id = {
    "O": 0,
    "B-DNA": 1,
    "I-DNA": 2,
    "B-protein": 3,
    "I-protein": 4,
    "B-cell_type": 5,
    "I-cell_type": 6,
    "B-cell_line": 7,
    "I-cell_line": 8,
    "B-RNA": 9,
    "I-RNA": 10,
}

# This is a private repo
try:
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=11,
        id2label=id2label,
        label2id=label2id,
    )
except OSError:
    import subprocess

    subprocess.run(["huggingface-cli", "login"])
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=11,
        id2label=id2label,
        label2id=label2id,
        use_auth_token=True,
    )

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="all",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=model_dir,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_bionlp["train"],
    eval_dataset=tokenized_bionlp["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

config = PeftConfig.from_pretrained(model_dir)
inference_model = AutoModelForTokenClassification.from_pretrained(
    config.base_model_name_or_path,
    num_labels=11,
    id2label=id2label,
    label2id=label2id,
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(inference_model, model_dir)

text = (
    "The activation of IL-2 gene expression and NF-kappa B through CD28 "
    "requires reactive oxygen production by 5-lipoxygenase."
)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

tokens = inputs.tokens()
predictions = torch.argmax(logits, dim=2)

for token, prediction in zip(tokens, predictions[0].numpy()):
    print((token, model.config.id2label[prediction]))

# ('<s>', 'O')
# ('The', 'O')
# ('Ġactivation', 'O')
# ('Ġof', 'O')
# ('ĠIL', 'B-DNA')
# ('-', 'O')
# ('2', 'I-DNA')
# ('Ġgene', 'I-DNA')
# ('Ġexpression', 'O')
# ('Ġand', 'O')
# ('ĠNF', 'B-protein')
# ('-', 'O')
# ('k', 'I-protein')
# ('appa', 'I-protein')
# ('ĠB', 'I-protein')
# ('Ġthrough', 'O')
# ('ĠCD', 'B-protein')
# ('28', 'I-protein')
# ('Ġrequires', 'O')
# ('Ġreactive', 'O')
# ('Ġoxygen', 'O')
# ('Ġproduction', 'O')
# ('Ġby', 'O')
# ('Ġ5', 'B-protein')
# ('-', 'O')
# ('lip', 'I-protein')
# ('oxy', 'I-protein')
# ('gen', 'I-protein')
# ('ase', 'I-protein')
# ('.', 'O')
# ('</s>', 'O')
