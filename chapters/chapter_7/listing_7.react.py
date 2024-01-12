import os
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from dotenv import load_dotenv

load_dotenv()

# load API keys; you will need to obtain these if you haven't yet
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = OpenAI(model_name="text-davinci-003", temperature=0)
tools = load_tools(["google-serper", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True
)

agent.run(
    "Who is Olivia Wilde's boyfriend? \
    What is his current age raised to the 0.23 power?"
)

# > Entering new AgentExecutor chain...
# I need to find out who Olivia Wilde's boyfriend is and then
# calculate his age raised to the 0.23 power.
# Action: Search
# Action Input: "Olivia Wilde boyfriend"
# Observation: Olivia Wilde started dating Harry Styles after ending
# her years-long engagement to Jason Sudeikis — see their relationship
# timeline.
# Thought: I need to find out Harry Styles' age.
# Action: Search
# Action Input: "Harry Styles age"
# Observation: 29 years
# Thought: I need to calculate 29 raised to the 0.23 power.
# Action: Calculator
# Action Input: 29^0.23
# Observation: Answer: 2.169459462491557

# Thought: I now know the final answer.
# Final Answer: Harry Styles, Olivia Wilde's boyfriend, is 29 years old
# and his age raised to the 0.23 power is 2.169459462491557.

# > Finished chain.

# "Harry Styles, Olivia Wilde's boyfriend, is 29 years old and his age
# raised to the 0.23 power is 2.169459462491557."
