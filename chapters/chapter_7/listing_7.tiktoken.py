import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
print(encoding.encode("You're users chat message goes here."))
# [2675, 2351, 3932, 6369, 1984, 5900, 1618, 13]


def count_tokens(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


num_tokens = count_tokens("You're users chat message goes here.")
print(num_tokens)
# 8
