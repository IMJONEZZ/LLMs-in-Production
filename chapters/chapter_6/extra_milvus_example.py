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


class WikiDataIngestion:
    def __init__(
        self,
        index,
        wikidata=None,
        embedder=None,
        tokenizer=None,
        text_splitter=None,
        batch_limit=100,
    ):
        self.index = index
        self.wikidata = wikidata or load_dataset(
            "wikipedia",
            "20220301.simple",
            split="train[:1000]",
            trust_remote_code=True,
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

    def get_wiki_metadata(self, page):
        return {
            "wiki-id": str(page["id"]),
            "source": page["url"],
            "title": page["title"],
        }

    def split_texts_and_metadatas(self, page):
        basic_metadata = self.get_wiki_metadata(page)
        texts = self.text_splitter.split_text(page["text"])
        metadatas = [
            {"chunk": j, "text": text, **basic_metadata}
            for j, text in enumerate(texts)
        ]
        return texts, metadatas

    def upload_batch(self, texts, metadatas):
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeddings = self.embedder.encode(texts)
        self.index.insert([ids, embeddings, metadatas])

    def batch_upload(self):
        batch_texts = []
        batch_metadatas = []

        for page in tqdm(self.wikidata):
            texts, metadatas = self.split_texts_and_metadatas(page)

            batch_texts.extend(texts)
            batch_metadatas.extend(metadatas)

            if len(batch_texts) >= self.batch_limit:
                self.upload_batch(batch_texts, batch_metadatas)
                batch_texts = []
                batch_metadatas = []

        if len(batch_texts) > 0:
            self.upload_batch(batch_texts, batch_metadatas)

        self.index.flush()


if __name__ == "__main__":
    index_name = "milvus_llm_example2"
    dim = 768

    # Create index if it doesn't exist
    has = utility.has_collection(index_name)
    if has:
        utility.drop_collection(index_name)

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
        fields, f"{index_name} is the simplest demo to introduce the APIs"
    )

    print(f"Create collection {index_name}")
    index = Collection(index_name, schema)

    # Connect to index and describe stats
    index = Collection(index_name)
    print(index.num_entities)

    # Ingest data and describe stats anew
    wiki_data_ingestion = WikiDataIngestion(index)
    wiki_data_ingestion.batch_upload()
    print(index.num_entities)

    # Build search index
    search_index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},  # number of clusters
    }
    index.create_index("embeddings", search_index)

    # Before conducting a search, you need to load the data into memory.
    index.load()

    # Make a query
    query = "Did Johannes Gutenberg invent the printing press?"
    search_embedding = wiki_data_ingestion.embedder.encode(query)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},  # number of clusters to search
    }
    results = index.search(
        [search_embedding],
        "embeddings",
        search_params,
        limit=3,
        output_fields=["metadata"],
    )
    for hits in results:
        for hit in hits:
            print(hit.distance)
            print(hit.entity.metadata["title"])
    # 0.7209961414337158
    # Paper
    # 0.8377139568328857
    # Book
    # 0.8619614839553833
    # Book
