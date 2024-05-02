import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


prompt = ChatPromptTemplate.from_template("Tell me a story about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

result = chain.invoke("the printing press")
print(result)

# Once upon a time, in a small town nestled in the countryside, there lived a young man named Johannes Gutenberg. Johannes was a brilliant inventor who had always been fascinated by the world of letters and words. He spent countless hours studying and experimenting with different types of ink and paper, trying to find the perfect combination to create a printing press that could produce books quickly and efficiently.

# One day, Johannes had a breakthrough. He discovered a new type of ink that was not only easy to work with but also produced incredibly sharp and clear prints. With this discovery, Johannes set to work building his printing press. He spent long hours in his small workshop, tinkering and testing until he finally created a machine that could print an entire page of text in just a few hours.
