import evaluate

# Download a metric from Huggingface's hub
squad_metric = evaluate.load("squad")

# Example from the SQuAD dataset
predictions = [
    {
        "prediction_text": "Saint Bernadette",
        "id": "5733be284776f41900661182",
    },
    {"prediction_text": "Salma Hayek", "id": "56d4fa2e2ccc5a1400d833cd"},
    {"prediction_text": "1000 MB", "id": "57062c2552bb89140068992c"},
]
references = [
    {
        "answers": {
            "text": ["Saint Bernadette Soubirous"],
            "answer_start": [515],
        },
        "id": "5733be284776f41900661182",
    },
    {
        "answers": {
            "text": ["Salma Hayek and Frida Giannini"],
            "answer_start": [533],
        },
        "id": "56d4fa2e2ccc5a1400d833cd",
    },
    {
        "answers": {"text": ["1000 MB"], "answer_start": [437]},
        "id": "57062c2552bb89140068992c",
    },
]

results = squad_metric.compute(
    predictions=predictions, references=references
)
print(results)
# {'exact_match': 33.333333333333336, 'f1': 79.04761904761905}
