import time
from llama_cpp import Llama

llm = Llama(model_path="./models/wizardcoder-python-7b-v1.0.Q2_K.gguf")

start_time = time.time()
output = llm(
    "Q: Write python code to reverse a linked list. A: ",
    max_tokens=200,
    stop=["Q:"],
    echo=True,
)
end_time = time.time()

print(output["choices"])
# [
#     {'text': "Q: Write python code to reverse a linked list. A:
#         class Node(object):
#             def __init__(self, data=None):
#                 self.data = data
#                 self.next = None

#         def reverse_list(head):
#             prev = None
#             current = head
#             while current is not None:
#                 next = current.next
#                 current.next = prev
#                 prev = current
#                 current = next
#             return prev

#         # example usage
#         # initial list
#         head = Node('a')
#         head.next=Node('b')
#         head.next.next=Node('c')
#         head.next.next.next=Node('d')
#         print(head)
#         reverse_list(head) # call the function
#         print(head)

#         # expected output: d->c->b->a",
#     'index': 0,
#     'logprobs': None,
#     'finish_reason': 'stop'
#     }
# ]

print(f"Elapsed time: {end_time - start_time:.3f} seconds")
# Elapsed time: 239.457 seconds
