import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_ckpt)

dataset = load_dataset("A Dataset")


def embed_text(example):
    embedding = model.encode(example["text"])
    return {"embedding": np.array(embedding, dtype=np.float32)}


print(f"Train 1: {dataset['train'][0]}")
embs_train = dataset["train"].map(embed_text, batched=False)
embs_test = dataset["test"].map(embed_text, batched=False)

embs_train.add_faiss_index("embedding")

idx, knn = 1, 3  # Select the first query and 3 nearest neighbors

query = np.array(embs_test[idx]["embedding"], dtype=np.float32)
scores, samples = embs_train.get_nearest_examples("embedding", query, k=knn)

print(f"QUERY LABEL: {embs_test[idx]['text_label']}")
print(
    f"QUERY TEXT: {embs_test[idx]['Tweet text'][:200].replace(rn, nl)} [...]\n"
)
print("=" * 50)
print("Retrieved Documents:")
for score, label, text in zip(
    scores, samples["text_label"], samples["Tweet text"]
):
    print("=" * 50)
    print(f"TEXT:\n{text[:200].replace(rn, nl)} [...]")
    print(f"SCORE: {score:.2f}")
    print(f"LABEL: {label}")
