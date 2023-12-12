from guidance import models, gen, select

# Load a HuggingFace Transformers model
falcon = models.Transformers("tiiuae/falcon-rw-1b")

# Set a token limit that is an actual limit
lm = falcon + "Once upon a time, " + gen(max_tokens=10)
print(lm)  # Once upon a time, there was a little girl who was very shy.

# Set stopping tokens
lm = (
    falcon
    + "Write a sentence about the printing press. "
    + gen(stop=["\n", ".", "!"])
)
print(lm)  # Write a sentence about the printing press. \
# The printing press was invented by Johannes Gutenberg in 1450

# Combine mutliple limits
lm = falcon + "1, 2, 3," + gen(max_tokens=50, stop="11")
print(lm)
# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

# Generate a specifc response from a list
lm = falcon + "I like the color " + select(["cyan", "grey", "purple"])
print(lm)  # I like the color purple

# Use regular expressions to ensure the response matches a pattern
lm = falcon + "Generate an email: " + gen(regex="\w+@\w+.com")
print(lm)  # Generate an email: theoreticaly@gmail.com
