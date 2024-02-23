import os
import pinecone
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# find API key in console at app.pinecone.io
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Set up vectorstore
index_name = "pincecone-llm-example"
index = pinecone.Index(index_name)
embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
)
text_field = "text"
vectorstore = Pinecone(index, embedder.embed_query, text_field)

# Make a query
query = "Who was Johannes Gutenberg?"
vectorstore.similarity_search(
    query, k=3  # our search query  # return 3 most relevant docs
)

# Now lets use these results to enrich our LLM prompt
# Set up LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.0,
)

# Run query with vectorstore
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)
qa.run(query)

# Include wikipedia sources
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)
qa_with_sources(query)
