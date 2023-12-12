import guidance
from guidance import models, select

# Load a HuggingFace Transformers model
falcon = models.Transformers("tiiuae/falcon-rw-1b")


# Create functions to easily implement grammars
@guidance(stateless=True)
def parts_of_speech(lm):
    return lm + select(["Noun", "Verb", "Adjective", "Adverb", ""])


lm = (
    falcon
    + "The child plays with a red ball. Ball in the previous sentence is a "
    + parts_of_speech()
)
print(lm)  # Noun


@guidance(stateless=True)
def pos_constraint(lm, sentence):
    words = sentence.split()
    for word in words:
        lm += word + ": " + parts_of_speech() + "\n"
    return lm


@guidance(stateless=True)
def pos_instruct(lm, sentence):
    lm += f"""
    Tag each word with their parts of speech.
    Example:
    Input: The child plays with a red ball.
    Output:
    The: 
    child: Noun
    plays: Verb
    with: 
    a: 
    red: Adjective
    ball.: Noun
    ---
    Input: {sentence}
    Output:
    """
    return lm


sentence = "Good software makes the complex appear to be simple"
lm = falcon + pos_instruct(sentence) + pos_constraint(sentence)
print(lm)
# Input: Good software makes the complex appear to be simple
# Output:
# Good:
# software:
# makes: Verb
# the:
# complex: Adjective
# appear: Adjective
# to:
# be: Verb
# simple: Adjective
