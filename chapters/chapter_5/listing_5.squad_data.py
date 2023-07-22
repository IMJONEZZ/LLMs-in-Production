import pandas as pd
from datasets import load_dataset
import datetime

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

path = "./data/qa.parquet"


def save_qa_to_parquet(path):
    # Load SQuAD dataset
    squad = load_dataset("squad", split="train[:5000]")
    # Extract questions and answers
    ids = squad["id"]
    questions = squad["question"]
    answers = [answer["text"][0] for answer in squad["answers"]]
    # Create dataframe
    qa = pd.DataFrame(
        zip(ids, questions, answers),
        columns=["question_id", "questions", "answers"],
    )
    # Add embeddings and timestamps
    qa["embeddings"] = qa.questions.apply(lambda x: model.encode(x))
    qa["created"] = datetime.datetime.utcnow()
    qa["datetime"] = qa["created"].dt.floor("h")
    # Save to parquet
    qa.to_parquet(path)


save_qa_to_parquet(path)
