from langchain.llms import LlamaCpp
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, initialize_agent, Tool, AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import DuckDuckGoSearchRun, YouTubeSearchTool
from langchain.document_loaders.csv_loader import CSVLoader

llm = LlamaCpp(
    model_path="./data/mistral-7b-instruct-v0.1.Q4_0.gguf",
    n_gpu_layers=0, # 1 if NEON, any number if CUBLAS else 0
    n_batch=512,
    n_ctx=32768, #Context window for the model
    verbose=False
)

# Here are some demonstrations of the tools

search = DuckDuckGoSearchRun()
hot_topic = search.run("Tiktoker finds proof of Fruit of the Loom cornucopia in the logo")

youtube_tool = YouTubeSearchTool()
fun_channel = youtube_tool.run("jaubrey", 3)

print(hot_topic, fun_channel)

# Let's create a couple out-of-the-box agents now

agent = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
)

agent.run("""Using Python_REPL please write a neural network in Pytorch.
          Use Python_REPL as the Action and your code as the Action Input.
          Use synthetic data from a normal distribution.
          Train for 1000 epochs and print every 100 epochs.
          Return a prediction for x = 5."""
        )

agent = create_csv_agent(
    llm,
    "./data/Slack_Dataset.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

agent.run(
    """Using python_repl_ast please tell me whether the user polite in their messages.
    Use python_repl_ast as the Action and the command as the Action input."""
)

# How to define your own agent tool

duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Useful for when an internet search is needed"
)

coding_tool = PythonREPLTool()
tools = load_tools(["llm-math"], llm=llm)
tools += [duckduckgo_tool, youtube_tool, coding_tool]

# How to define your agent's memory

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", 
    k=5, 
    return_messages=True, 
    output_key="output"
)

# Now let's set up and initialize a completely custom agent

agent = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent="chat-conversational-react-description", 
    verbose=True, 
    memory=memory,
    handle_parsing_errors=True
)

# special tokens used by llama 2 chat
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# create the system message
sys_msg = "<s>" + B_SYS + """Assistant is a expert JSON builder designed to assist with a wide range of tasks.

Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

All of Assistant's communication is performed using this JSON format.

Assistant can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. Tools available to Assistant are:

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

Here is the latest conversation between Assistant and User.""" + E_SYS
new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
agent.agent.llm_chain.prompt = new_prompt

instruction = B_INST + " Respond to the following in JSON with 'action' and 'action_input' values " + E_INST
human_msg = instruction + "\nUser: {input}"

agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg

agent.run("Tell me how old Justin Beiber was when the Patriots last won the Superbowl.")