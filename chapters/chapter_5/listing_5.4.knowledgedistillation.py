import os
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import load_dataset, load_metric

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def process(examples):
    tokenized_inputs = tokenizer(
        examples["sentence"], truncation=True, max_length=256
    )
    return tokenized_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(
        predictions=predictions, references=labels
    )
    return {
        "accuracy": acc["accuracy"],
    }


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size
        assert (
            outputs_student.logits.size() == outputs_teacher.logits.size()
        )

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = loss_function(
            F.log_softmax(
                outputs_student.logits / self.args.temperature, dim=-1
            ),
            F.softmax(
                outputs_teacher.logits / self.args.temperature, dim=-1
            ),
        ) * (self.args.temperature**2)
        # Return weighted student loss
        loss = (
            self.args.alpha * student_loss
            + (1.0 - self.args.alpha) * loss_logits
        )
        return (loss, outputs_student) if return_outputs else loss


if __name__ == "__main__":
    # Create model directory to save to
    model_dir = "./models/KDGPT/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define Teacher and Student models
    student_id = "gpt2"
    teacher_id = "gpt2-medium"

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    student_tokenizer = AutoTokenizer.from_pretrained(student_id)

    sample = "Here's our sanity check."

    assert teacher_tokenizer(sample) == student_tokenizer(sample), (
        "Tokenizers need to have the same output! "
        f"{teacher_tokenizer(sample)} != {student_tokenizer(sample)}"
    )

    del teacher_tokenizer
    del student_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dataset_id = "glue"
    dataset_config = "sst2"

    dataset = load_dataset(dataset_id, dataset_config)

    tokenized_dataset = dataset.map(process, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    print(tokenized_dataset["test"].features)

    # create label2id, id2label dicts for nice outputs for the model
    labels = tokenized_dataset["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # define training args
    training_args = DistillationTrainingArguments(
        output_dir=model_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=False,  # False if no GPU
        learning_rate=6e-5,
        seed=8855,
        # Evaluation strategies
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        # push to hub parameters
        push_to_hub=False,
        # distilation parameters
        alpha=0.5,
        temperature=4.0,
    )

    # define data_collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # define model
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # define student model
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # define metrics and metrics function
    accuracy_metric = load_metric("accuracy")

    trainer = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(model_dir)
