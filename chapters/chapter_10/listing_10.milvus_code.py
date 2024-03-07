from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from transformers import AutoTokenizer
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm
from uuid import uuid4


# Connect to Milvus
connections.connect("default", host="localhost", port="19530")


class PythonCodeIngestion:
    def __init__(
        self,
        collection,
        python_code=None,
        embedder=None,
        tokenizer=None,
        text_splitter=None,
        batch_limit=100,
    ):
        self.collection = collection
        self.python_code = python_code or load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca",
            split="train",
        )
        self.embedder = embedder or SentenceTransformer(
            "krlvi/sentence-t5-base-nlpl-code_search_net"
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            "Deci/DeciCoder-1b"
        )
        self.text_splitter = (
            text_splitter
            or RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=20,
                length_function=self.token_length,
                separators=["\n\n", "\n", " ", ""],
            )
        )
        self.batch_limit = batch_limit

    def token_length(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def get_metadata(self, page):
        return {
            "instruction": page["instruction"],
            "input": page["input"],
            "output": page["output"],
        }

    def split_texts_and_metadatas(self, page):
        basic_metadata = self.get_metadata(page)
        prompts = self.text_splitter.split_text(page["prompt"])
        metadatas = [
            {"chunk": j, "prompt": prompt, **basic_metadata}
            for j, prompt in enumerate(prompts)
        ]
        return prompts, metadatas

    def upload_batch(self, texts, metadatas):
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeddings = self.embedder.encode(texts)
        self.collection.insert([ids, embeddings, metadatas])

    def batch_upload(self):
        batch_texts = []
        batch_metadatas = []

        for page in tqdm(self.python_code):
            texts, metadatas = self.split_texts_and_metadatas(page)

            batch_texts.extend(texts)
            batch_metadatas.extend(metadatas)

            if len(batch_texts) >= self.batch_limit:
                self.upload_batch(batch_texts, batch_metadatas)
                batch_texts = []
                batch_metadatas = []

        if len(batch_texts) > 0:
            self.upload_batch(batch_texts, batch_metadatas)

        self.collection.flush()


if __name__ == "__main__":
    collection_name = "milvus_llm_example"
    dim = 768

    # Create collection if it doesn't exist
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(
            name="ids",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=36,
        ),
        FieldSchema(
            name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim
        ),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]

    schema = CollectionSchema(
        fields, f"{collection_name} is collection of python code prompts"
    )

    print(f"Create collection {collection_name}")
    collection = Collection(collection_name, schema)

    # Connect to collection and show size
    collection = Collection(collection_name)
    print(collection.num_entities)

    # Ingest data and show stats now that data is ingested
    python_code_ingestion = PythonCodeIngestion(collection)
    python_code_ingestion.batch_upload()
    print(collection.num_entities)

    # Build search index
    search_index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},  # number of clusters
    }
    collection.create_index("embeddings", search_index)

    # Before conducting a search, you need to load the data into memory.
    collection.load()

    # Make a query
    query = (
        "Construct a neural network model in Python to classify "
        "the MNIST data set correctly."
    )
    search_embedding = python_code_ingestion.embedder.encode(query)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},  # number of clusters to search
    }
    results = collection.search(
        [search_embedding],
        "embeddings",
        search_params,
        limit=3,
        output_fields=["metadata"],
    )
    for hits in results:
        for hit in hits:
            print(hit.distance)
            print(hit.entity.metadata["instruction"])
    # 0.7066953182220459
    # Create a neural network in Python to identify
    # hand-written digits from the MNIST dataset.
    # 0.7366453409194946
    # Create a question-answering system using Python
    # and Natural Language Processing.
    # 0.7389795184135437
    # Write a Python program to create a neural network model that can
    # classify handwritten digits (0-9) with at least 95% accuracy.
