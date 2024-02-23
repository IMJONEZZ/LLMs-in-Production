from langchain.llms import LlamaCpp
from langchain.chains.conversation.memory import (
    ConversationBufferWindowMemory,
)
from langchain.agents import load_tools, initialize_agent, Tool
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import DuckDuckGoSearchRun, YouTubeSearchTool


# Load in the model
llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_0.gguf",
    n_gpu_layers=0,  # 1 if NEON, any number if CUBLAS else 0
    n_batch=512,
    n_ctx=32768,  # Context window for the model
    verbose=False,
)

# Define our own agent tools
search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Useful for when an internet search is needed",
)
youtube_tool = YouTubeSearchTool()
coding_tool = PythonREPLTool()

tools = load_tools(["llm-math"], llm=llm)
tools += [duckduckgo_tool, youtube_tool, coding_tool]

# Define our agent's memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output",
)

# Set up and initialize our custom agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Special tokens used by llama 2 chat
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Create the System Prompt
sys_msg = (
    "<s>"
    + B_SYS
    + """Assistant is a expert JSON builder designed to assist with a wide \
range of tasks.

Assistant is able to respond to the User and use tools using JSON strings \
that contain "action" and "action_input" parameters.

All of Assistant's communication is performed using this JSON format.

Assistant can also use tools by responding to the user with tool use \
instructions in the same "action" and "action_input" JSON format. Tools \
available to Assistant are:

- "Calculator": Useful for when you need to answer questions about math.
  - To use the calculator tool, Assistant should write like so:
    ```json
    {{"action": "Calculator",
      "action_input": "sqrt(4)"}}
    ```
- "DuckDuckGo Search": Useful for when an internet search is needed.
  - To use the duckduckgo search tool, Assistant should write like so:
    ```json
    {{"action": "DuckDuckGo Search",
      "action_input": "When was the Jonas Brothers' first concert"}}

Here are some previous conversations between the Assistant and User:

User: Hey how are you today?
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "I'm good thanks, how are you?"}}
```
User: I'm great, what is the square root of 4?
Assistant: ```json
{{"action": "Calculator",
 "action_input": "sqrt(4)"}}
```
User: 2.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 2!"}}
```
User: Thanks, when was the Jonas Brothers' first concert?
Assistant: ```json
{{"action": "DuckDuckGo Search",
 "action_input": "When was the Jonas Brothers' first concert"}}
```
User: 12.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "They had their first concert in 2005!"}}
```
User: Thanks could you tell me what 4 to the power of 2 is?
Assistant: ```json
{{"action": "Calculator",
 "action_input": "4**2"}}
```
User: 16.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 16!"}}
```

Here is the latest conversation between Assistant and User."""
    + E_SYS
)

# Add System Prompt to agent
new_prompt = agent.agent.create_prompt(system_message=sys_msg, tools=tools)
agent.agent.llm_chain.prompt = new_prompt

# Add Instruction to agent
instruction = (
    B_INST
    + " Respond to the following in JSON with 'action' and 'action_input' "
    "values " + E_INST
)
human_msg = instruction + "\nUser: {input}"
agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg

# Run with User Input
agent.run(
    "Tell me how old Justin Beiber was when the Patriots last won the "
    "Superbowl."
)
# Remember that for this we asked the model to respond in json

# > Entering new AgentExecutor chain…
# Assistant: {
# "action": "DuckDuckGo Search",
# "action\_input": "When did the New England Patriots last win the Super
# Bowl? Justin Bieber birthdate"
# }
# {
# "action": "Final Answer",
# "action\_input": "Justin Bieber was born on March 1, 1994. The Patriots
# last won the Super Bowl in February 2017."
# }
