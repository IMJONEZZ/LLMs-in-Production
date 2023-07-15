import os
import pinecone
import tiktoken
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

from tqdm.auto import tqdm
from uuid import uuid4

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# find API key in console at app.pinecone.io
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# find ENV (cloud region) next to API key in console
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


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
            "wikipedia", "20220301.simple", split="train[:10000]"
        )
        self.embedder = embedder or OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
        )
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
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
        tokens = self.tokenizer.encode(text, disallowed_special=())
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
        embeddings = self.embedder.embed_documents(texts)
        self.index.upsert(vectors=zip(ids, embeddings, metadatas))

    def batch_upload(self):
        batch_texts = []
        batch_metadatas = []

        for page in tqdm(self.wikidata):
            texts, metadatas = self.split_texts_and_metadatas(page)

            batch_texts.extend(texts)
            batch_metadatas.extend(metadatas)

            if len(texts) >= self.batch_limit:
                self.upload_batch(batch_texts, batch_metadatas)
                batch_texts = []
                batch_metadatas = []

        if len(texts) > 0:
            self.upload_batch(batch_texts, batch_metadatas)


if __name__ == "__main__":
    index_name = "Pincecone-LLM-Example"

    # Create index if it doesn't exist
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric="cosine",
            dimension=1536,  # 1536 dim of text-embedding-ada-002
        )

    # Connect to index and describe stats
    index = pinecone.GRPCIndex(index_name)
    index.describe_index_stats()

    # Ingest data and describe stats anew
    wiki_data_ingestion = WikiDataIngestion(index)
    wiki_data_ingestion.batch_upload()
    index.describe_index_stats()
