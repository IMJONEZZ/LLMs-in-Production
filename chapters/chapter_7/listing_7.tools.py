import guidance
from guidance import models, gen

# Load a HuggingFace Transformers model
falcon = models.Transformers("tiiuae/falcon-rw-1b")


@guidance
def add(lm, input1, input2):
    lm += f" = {int(input1) + int(input2)}"
    return lm


@guidance
def subtract(lm, input1, input2):
    lm += f" = {int(input1) - int(input2)}"
    return lm


@guidance
def multiply(lm, input1, input2):
    lm += f" = {float(input1) * float(input2)}"
    return lm


@guidance
def divide(lm, input1, input2):
    lm += f" = {float(input1) / float(input2)}"
    return lm


lm = (
    falcon
    + """\
1 + 2 = add(1, 2) = 3
4 - 5 = subtract(4, 5) = -1
5 * 6 = multiply(5, 6) = 30
7 / 8 = divide(7, 8) = 0.875
Generate more examples of add, subtract, multiply, and divide
"""
)
lm += gen(max_tokens=15, tools=[add, subtract, multiply, divide])
print(lm)
