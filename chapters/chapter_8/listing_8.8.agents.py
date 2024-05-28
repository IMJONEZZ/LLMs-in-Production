from langchain.llms import LlamaCpp
from langchain.agents import AgentType
from langchain_experimental.agents import create_csv_agent


# Load in the model
llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_0.gguf",
    n_gpu_layers=0,  # 1 if NEON, any number if CUBLAS else 0
    n_batch=512,
    n_ctx=32768,  # Context window for the model
    verbose=False,
)

# # An agent that will generate Python code and execute it
# agent = create_python_agent(
#     llm=llm,
#     tool=PythonREPLTool(),
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     handle_parsing_errors=True,
# )

# agent.run(
#     """Using Python_REPL please write a neural network in Pytorch.
#           Use Python_REPL as the Action and your code as the Action Input.
#           Use synthetic data from a normal distribution.
#           Train for 1000 epochs and print every 100 epochs.
#           Return a prediction for x = 5."""
# )
# > Entering new AgentExecutor chain…
# This is a task about creating and training a neural network. I should
# start by importing Pytorch and defining the synthetic data. Then I will
# define the neural network and train it for 1000 epochs. Finally, I will
# print the prediction for x = 5.
# Action: Python_REPL
# Action Input:
# ```python
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils import data
# from torchvision.datasets import make_classification
## define synthetic data from normal distribution
# data = make_classification(n_samples=1000, n_features=10, n_informative=5,
# random_state=42)
# X = data['features']
# y = data['targets']
# ```Python REPL can execute arbitrary code. Use with caution.
# Observation: ImportError("cannot import name 'make_classification' from
#'torchvision.datasets'
# Thought: It seems like the code is trying to use a function that does not
# exist, I will check the import statement.
# Action: Python_REPL
# Action Input:
# ```python
## fix import
# import torch.utils.data as data
# from torchvision.datasets import make_classification
# ```
# Observation: ImportError("cannot import name 'make_classification' from
#'torchvision.datasets'
# Thought: It still seems like the code is trying to use a function that
# does not exist, I will make sure that the function name matches the
# correct import statement.
# Action: Python_REPL
# Action Input:
# ```python
## fix import
# from torchvision.datasets.make_classification import make_classification
# data = make_classification(n_samples=1000, n_features=10, n_informative=5,
# random_state=42)
# X = data['features']
# y = data['targets']
# ```
# Observation: ImportError("cannot import name 'MakeClassification' from
#'torchvision.datasets'
# Thought:
# > Finished chain.


# An agent that will read a CSV and analyze it
agent = create_csv_agent(
    llm,
    "./data/Slack_Dataset.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
)

agent.run(
    """Using python_repl_ast please tell me whether the user polite in their
    messages. Use python_repl_ast as the Action and the command as the
    Action input."""
)
# > Entering new AgentExecutor chain…
# Action: python_repl_ast
# Action Input: df['text'].str.contains('thank you')
# Observation:
# 0  	False
# 1  	False
# 2  	False
# 3  	False
# 4  	False
#   	…
# 286	False
# 287	False
# 288	False
# 289	False
# 290	False
# Name: text, Length: 291, dtype: bool
# Thought: It seems the user was not polite in their messages.
# Final Answer: The user was not polite in their messages.
# > Finished chain.
