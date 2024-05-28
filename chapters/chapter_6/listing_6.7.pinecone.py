import os
import tiktoken
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm
from uuid import uuid4

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# find API key in console at app.pinecone.io
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# find ENV (cloud region) next to API key in console
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

pc = Pinecone(api_key=PINECONE_API_KEY)

# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


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

            if len(batch_texts) >= self.batch_limit:
                self.upload_batch(batch_texts, batch_metadatas)
                batch_texts = []
                batch_metadatas = []

        if len(batch_texts) > 0:
            self.upload_batch(batch_texts, batch_metadatas)


if __name__ == "__main__":
    index_name = "pincecone-llm-example"

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            metric="cosine",
            dimension=1536,  # 1536 dim of text-embedding-ada-002
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # Connect to index and describe stats
    index = pc.Index(index_name)
    print(index.describe_index_stats())

    # Use a generic embedder if an openai api key is not provided
    embedder = None
    if not OPENAI_API_KEY:
        embedder = SentenceTransformer(
            "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"
        )  # Also 1536 dim
        embedder.embed_documents = lambda *args, **kwargs: embedder.encode(
            *args, **kwargs
        ).tolist()

    # Ingest data and describe stats anew
    wiki_data_ingestion = WikiDataIngestion(index, embedder=embedder)
    wiki_data_ingestion.batch_upload()
    print(index.describe_index_stats())

    # Make a query
    query = "Did Johannes Gutenberg invent the printing press?"
    embeddings = wiki_data_ingestion.embedder.embed_documents(query)
    results = index.query(vector=embeddings, top_k=3, include_metadata=True)
    print(results)
    # {'matches': [{'id': '18e2ab0a-f627-436a-b1ab-ed89b5b29c3b',
    #     'metadata': {'chunk': 0.0,
    #                 'source': 'https://simple.wikipedia.org/wiki/Johannes%20Gutenberg',
    #                 'text': 'Johannes Gensfleisch zur Laden zum '
    #                         'Gutenberg (more commonly known as Johannes '
    #                         'Gutenberg) (1390s – 3 February 1468), was '
    #                         'a German metal-worker and inventor. He is '
    #                         'famous for his work in printing in the '
    #                         '1450s, and is specifically known for '
    #                         'inventing typography.\n'
    #                         '\n'
    #                         'Gutenberg was born in Mainz, Germany, as '
    #                         'the son of a merchant, Friele Gensfleisch '
    #                         "zur Laden. Gutenberg's father took the "
    #                         'surname "zum Gutenberg" after the name of '
    #                         'the place they now lived.\n'
    #                         '\n'
    #                         'Gutenberg invented a sort of metal alloy '
    #                         'for printing; inks; a way to fix type '
    #                         '(metal letters) very accurately; and a new '
    #                         'sort of printing press. He took the idea '
    #                         'for his printing press from the presses '
    #                         'wine-makers used. Many people say '
    #                         'Gutenberg invented printing with moveable '
    #                         'type, but it was already invented in China '
    #                         'before that: see printing.\n'
    #                         '\n'
    #                         'Before movable type, people used block '
    #                         'printing, where the printer prints a whole '
    #                         'page from one piece of metal or wood. With '
    #                         'movable type, the printer makes a letter '
    #                         '(A, B, C ...) from a piece of metal or '
    #                         'wood, and can use it again and again in '
    #                         "different words. Together, all Gutenberg's "
    #                         'inventions made printing fast. In '
    #                         'Renaissance Europe, the improved '
    #                         'information technology made an information '
    #                         'explosion – in a short time, people '
    #                         'printed many new books about many topics.',
    #                 'title': 'Johannes Gutenberg',
    #                 'wiki-id': '5675'},
    #     'score': 0.871572554,
    #     'values': []},
    #     {'id': '97f55ce5-a5f4-40d8-8886-1bb459a97984',
    #     'metadata': {'chunk': 3.0,
    #                 'source': 'https://simple.wikipedia.org/wiki/Pencil',
    #                 'text': 'American colonists imported pencils from '
    #                         'Europe until after the American '
    #                         'Revolution. Benjamin Franklin advertised '
    #                         'pencils for sale in his Pennsylvania '
    #                         'Gazette in 1729, and George Washington '
    #                         'used a three-inch pencil when he surveyed '
    #                         'the Ohio Territory in 1762. It is said '
    #                         'that William Munroe, a cabinetmaker in '
    #                         'Concord, Massachusetts, made the first '
    #                         'American wood pencils in 1812. This was '
    #                         'not the only pencil-making occurring in '
    #                         'Concord.  Henry David Thoreau discovered '
    #                         'how to make a good pencil out of inferior '
    #                         'graphite using clay as the binder; this '
    #                         "invention was prompted by his father's "
    #                         'pencil factory in Concord, which employed '
    #                         'graphite found in New Hampshire in 1821 by '
    #                         'Charles Dunbar.\n'
    #                         '\n'
    #                         'Eraser attached \n'
    #                         'On 30 March 1858, Hymen Lipman received '
    #                         'the first patent for attaching an eraser '
    #                         'to the end of a pencil. In 1862 Lipman '
    #                         'sold his patent to Joseph Reckendorfer for '
    #                         '$100,000, who went to sue the pencil '
    #                         'manufacturer Faber-Castell for '
    #                         'infringement. In 1875, the Supreme Court '
    #                         'of the United States ruled against '
    #                         'Reckendorfer declaring the patent '
    #                         'invalid.\n'
    #                         '\n'
    #                         'The metal band used to mate the eraser '
    #                         'with pencil is called a ferrule.\n'
    #                         '\n'
    #                         'Other attempts \n'
    #                         'The first attempt to manufacture graphite '
    #                         'sticks from powdered graphite was in '
    #                         'Nuremberg, Germany in 1662. They used a '
    #                         'mixture of graphite, sulfur and antimony. '
    #                         'Though usable, they were not as good as '
    #                         'the English pencils.',
    #                 'title': 'Pencil',
    #                 'wiki-id': '4063'},
    #     'score': 0.868597746,
    #     'values': []},
    #     {'id': 'de9a60b5-73b5-460e-806d-b8356946fc4e',
    #     'metadata': {'chunk': 0.0,
    #                 'source': 'https://simple.wikipedia.org/wiki/Printing%20press',
    #                 'text': 'The printing press is a machine for '
    #                         'printing. It makes many copies of '
    #                         'identical pages. The printing press today '
    #                         'is used to print books and newspapers. It '
    #                         'had a great influence on society, '
    #                         'especially western society. It was "one of '
    #                         'the most potent agents... of western '
    #                         'civilization in bringing together the '
    #                         'scattered ideas of thinkers".\n'
    #                         '\n'
    #                         'Woodcut printing has been done for several '
    #                         'centuries. That was whole pages cut into '
    #                         'wood, words and pictures. In the 15th '
    #                         'century Johannes Gutenberg improved the '
    #                         'process. He used separate alloy letters '
    #                         'screwed into a frame. This way a large set '
    #                         'of letters could make almost any page for '
    #                         'printing. This process was called '
    #                         'typesetting. Each letter was in a block of '
    #                         'metal, fixed in a frame. He could then '
    #                         'move paper and ink over it, much like a '
    #                         'stamp. This method was called letterpress. '
    #                         'The letters would leave ink on the paper '
    #                         'in the shape of the letters, creating text '
    #                         'or illustrations.\n'
    #                         '\n'
    #                         'Bigger and much faster presses were '
    #                         'invented in the industrial revolution. The '
    #                         'main improvements were made in the 19th '
    #                         'century. Two ideas altered the design of '
    #                         'the printing press entirely. First was the '
    #                         'use of steam power to run the machinery. '
    #                         'Second was the replacement of the printing '
    #                         'flatbed with the rotary motion of '
    #                         'cylinders. Both were done by the German '
    #                         'printer Friedrich Koenig between 1802 and '
    #                         '1818. Having moved to London in 1804, '
    #                         'Koenig got financial support for his '
    #                         'project in 1807.  In 1810, Koenig patented '
    #                         'a steam press "much like a hand press '
    #                         'connected to a steam engine".\n'
    #                         '\n'
    #                         'Soon other inventions were added, such as '
    #                         'the making of cheaper paper by using wood '
    #                         'pulp instead of rags. Later in the 19th '
    #                         'century came machines which speeded up '
    #                         'typesetting, which was previously done by '
    #                         'hand, letter by letter. A machine for hot '
    #                         'metal typesetting was designed by Linotype '
    #                         'Inc. It turned molten lead into type ready '
    #                         'for printing.',
    #                 'title': 'Printing press',
    #                 'wiki-id': '32435'},
    #     'score': 0.865480721,
    #     'values': []}],
    # 'namespace': ''}
