import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

#model_ckpt = "openlm-research/open_llama_3b_v2" #Try this if your GPU is big enough
model_ckpt = "EleutherAI/pythia-1.4b-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModel.from_pretrained(model_ckpt, 
                                  torch_dtype=torch.bfloat16, 
                                  #rope_scaling={"type": "dynamic", "factor": 2.0}
                                  ).to("cuda")

dataset = load_dataset("ought/raft", "twitter_complaints")
classes = [
        label.replace("_", " ")
        for label in dataset["train"].features["Label"].names
    ]
dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["Label"]]},
        batched=True,
        num_proc=1,
    )

def mean_pooling(model_output, attention_mask):
    #Extract token embeddings
    token_embeddings = model_output[0]
    #Compute the attention mask
    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    #Sum the embeddings
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    #Return the average as a single vector
    return sum_embeddings / sum_mask

def embed_text(examples):
    inputs = tokenizer(examples['Tweet text'], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    with torch.no_grad():
        model_output = model(**inputs)
    pooled_embeds = mean_pooling(model_output, inputs['attention_mask'])
    return {"embedding": pooled_embeds.cpu().numpy()}

print(f"Train 1: {dataset['train'][0]}")
embs_train = dataset['train'].map(embed_text, batched=True, batch_size=16)
embs_test = dataset['test'].map(embed_text, batched=True, batch_size=16)

embs_train.add_faiss_index("embedding")

idx, knn = 100, 5 #Select the first query and 3 nearest neighbors

rn, nl = "\r\n\r\n", "\n" #Remove newlines

query = np.array(embs_test[idx]['embedding'], dtype=np.float32)
scores, samples = embs_train.get_nearest_examples("embedding", query, k=knn)

print(f"QUERY LABEL: {embs_test[idx]['text_label']}")
print(f"QUERY TEXT: {embs_test[idx]['Tweet text'][:200].replace(rn, nl)} [...]\n")
print("="*50)
print(f"Retrieved Documents:")
for score, label, text in zip(scores, samples['text_label'], samples['Tweet text']):
    print("="*50)
    print(f"TEXT:\n{text[:200].replace(rn, nl)} [...]")
    print(f"SCORE: {score:.2f}")
    print(f"LABEL: {label}")
