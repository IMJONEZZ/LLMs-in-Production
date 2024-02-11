import os
from langchain import OpenAI, PromptTemplate, LLMChain
from sentence_transformers import SentenceTransformer
import numpy as np
from datasets import load_dataset
import tiktoken

os.environ["OPENAI_API_KEY"] = "Your Server's API Key"
os.environ[
    "OPENAI_API_BASE"
] = "http://0.0.0.0:1234/v1"  # Replace with your server's address and port
os.environ[
    "OPENAI_API_HOST"
] = "http://0.0.0.0:1234"  # Replace with your host IP

llm = OpenAI(
    model_name="text-davinci-003",  # Can be anything
    temperature=0.25,
    openai_api_base="http://0.0.0.0:1234/v1",  # Again
    max_tokens=500,
    top_p=1,
    model_kwargs=dict(
        openai_key="your api key",
        top_k=1,
    ),
    presence_penalty=0.0,
    n=1,
    best_of=1,
    batch_size=20,
    logit_bias={},
    streaming=True,
)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tiktoker = tiktoken.encoding_for_model("gpt-3.5-turbo")


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

vectorDB = load_dataset(
    "csv", data_files="your dataset with embeddings.csv", split="train"
)
try:
    vectorDB.load_faiss_index("embeddings", "my_index.faiss")
except:
    print(
        "No faiss index, run vectorDB.add_faiss_index(column='embeddings') and vectorDB.save_faiss_index('embeddings', 'my_index.faiss')"
    )
message_history = []

query = "How can I train an LLM from scratch?"
embedded = embedder.encode(query)
q = np.array(embedded, dtype=np.float32)
_, retrieved_example = vectorDB.get_nearest_examples("embeddings", q, k=1)

formatted_prompt = PromptTemplate(
    input_variables=["question", "conversation_history", "code_snippet"],
    template=prompt_template,
)
chain = LLMChain(llm=llm, prompt=formatted_prompt)

num_tokens = len(
    tiktoker.encode(f"{prompt_template},\n" + message_history + query)
)
while num_tokens >= 4000:
    message_history.pop(0)
    num_tokens = len(
        tiktoker.encode(f"{prompt_template},\n" + message_history + query)
    )

res = chain.run(
    {
        "question": query,
        "conversation_history": message_history,
        "code_snippet": retrieved_example,
    }
)
message_history.append(f"User: {query}\nLlama: {res}")

print(res)
