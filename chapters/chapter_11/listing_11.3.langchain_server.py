import os
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import numpy as np
from datasets import load_dataset
import tiktoken

os.environ["OPENAI_API_KEY"] = "Your API Key"
os.environ["OPENAI_API_BASE"] = (
    "http://0.0.0.0:1234/v1"  # Replace with your server's address and port
)
os.environ["OPENAI_API_HOST"] = (
    "http://0.0.0.0:1234"  # Replace with your host IP
)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Can be anything
    temperature=0.25,
    openai_api_base=os.environ["OPENAI_API_BASE"],  # Again
    openai_api_key=os.environ["OPENAI_API_KEY"],
    max_tokens=500,
    n=1,
)

# Embeddings for RAG
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Tokenization for checking context length quickly
tiktoker = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Change the prompt to be whatever you want
prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ###Instruction:
        You are an expert python developer.
        Given a question, some conversation history, and the closest code snippet we could find for the request, give your best suggestion for how to write the code needed to answer the User's question.

        ###Input:

        #Question: {question}
        
        #Conversation History: {conversation_history}
        
        Code Snippet:
        {code_snippet}


        ###Response:
        """

# Here's a rudimentary vectorDB, change it to anything we've covered
vectorDB = load_dataset(
    "csv", data_files="your dataset with embeddings.csv", split="train"
)
# If you haven't created a faiss or elasticsearch or usearch index, do it
try:
    vectorDB.load_faiss_index("embeddings", "my_index.faiss")
except Exception:
    print(
        "No faiss index, run vectorDB.add_faiss_index(column='embeddings') and vectorDB.save_faiss_index('embeddings', 'my_index.faiss')"
    )

# To keep track of chat history, do this however you need!
message_history = []

# Search the Vector DB
query = "How can I train an LLM from scratch?"
embedded = embedder.encode(query)
q = np.array(embedded, dtype=np.float32)
_, retrieved_example = vectorDB.get_nearest_examples("embeddings", q, k=1)

# Format the prompt
formatted_prompt = PromptTemplate(
    input_variables=["question", "conversation_history", "code_snippet"],
    template=prompt_template,
)
# Set up the actual LLM Chain
chain = LLMChain(llm=llm, prompt=formatted_prompt)

# Don't overload your context length
num_tokens = len(
    tiktoker.encode(
        f"{prompt_template},\n" + "\n".join(message_history) + query
    )
)
while num_tokens >= 4000:
    message_history.pop(0)
    num_tokens = len(
        tiktoker.encode(
            f"{prompt_template},\n" + "\n".join(message_history) + query
        )
    )

# Run RAG with your API
res = chain.run(
    {
        "question": query,
        "conversation_history": message_history,
        "code_snippet": "",
    }
)
message_history.append(f"User: {query}\nLlama: {res}")

# I'm just printing, do whatever you need to here
print(res)
