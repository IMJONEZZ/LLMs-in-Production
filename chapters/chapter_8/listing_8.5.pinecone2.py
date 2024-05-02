import os
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as lc_Pinecone
from pinecone import Pinecone

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# find API key in console at app.pinecone.io
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


pc = Pinecone(api_key=PINECONE_API_KEY)


# Set up vectorstore
index_name = "pincecone-llm-example"
index = pc.Index(index_name)
embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
)
text_field = "text"
vectorstore = lc_Pinecone(index, embedder.embed_query, text_field)

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
result = qa_with_sources(query)
print(result)

# {'question': 'Who was Johannes Gutenberg?', 'answer': ' The image shows a list of births and deaths for the month of April in the year 1771. The births include Richard Trevithick, Karl Philipp Fürst zu Schwarzenberg, Jean Rapp, Prince Ernest Augustus of Great Britain, Sir Walter Scott, Archduke Charles of Austria, Emperor Kokaku of Japan, and Frederick William, Duke of Brunswick. The deaths include Jean-Andoche Junot and Marie François Xavier Bichat. ', 'sources': ''}
