import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Download embedding model and dataset
model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_ckpt)

embs_train = load_dataset("tweet_eval", "emoji", split="train[:1000]")
embs_test = load_dataset("tweet_eval", "emoji", split="test[:100]")


# Create embeddings
def embed_text(example):
    embedding = model.encode(example["text"])
    return {"embedding": np.array(embedding, dtype=np.float32)}


print(f"Train 1: {embs_train[0]}")
# Train 1: {'text': 'Sunday afternoon walking through Venice in the sun with @user ️ ️ ️ @ Abbot Kinney, Venice', 'label': 12}

embs_train = embs_train.map(embed_text, batched=False)
embs_test = embs_test.map(embed_text, batched=False)

# Add Faiss index which allows similarity search
embs_train.add_faiss_index("embedding")

# Run Query
idx, knn = 1, 3  # Select the first query and 3 nearest neighbors

query = np.array(embs_test[idx]["embedding"], dtype=np.float32)
scores, samples = embs_train.get_nearest_examples("embedding", query, k=knn)

# Print Results
print(f"QUERY LABEL: {embs_test[idx]['label']}")
print(f"QUERY TEXT: {embs_test[idx]['text'][:200]} [...]\n")
print("=" * 50)
print("Retrieved Documents:")
for score, label, text in zip(scores, samples["label"], samples["text"]):
    print("=" * 50)
    print(f"TEXT:\n{text[:200]} [...]")
    print(f"SCORE: {score:.2f}")
    print(f"LABEL: {label}")

# QUERY LABEL: 10
# QUERY TEXT: The calm before...... | w/ sofarsounds @user | : B. Hall.......#sofarsounds… [...]

# ==================================================
# Retrieved Documents:
# ==================================================
# TEXT:
# So much love for sofarsounds &amp; sofarsoundsla! Great times this eve #sofarla #actasif @ The… [...]
# SCORE: 0.85
# LABEL: 4
# ==================================================
# TEXT:
# Couch view lookin' like: #sundaze @ The Olivian Luxury Apartment Homes [...]
# SCORE: 1.11
# LABEL: 1
# ==================================================
# TEXT:
# I ️ Faure Requiem @user oboyddd @ Walt Disney Concert Hall [...]
# SCORE: 1.13
# LABEL: 0
