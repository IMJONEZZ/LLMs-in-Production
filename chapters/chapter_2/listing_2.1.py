from nltk.corpus.reader import PlaintextCorpusReader
from nltk.util import everygrams
from nltk.lm.preprocessing import (
    pad_both_ends,
    flatten,
    padded_everygram_pipeline,
)
from nltk.lm import MLE

try:
    import nltk

    nltk.data.find("tokenizers/punkt.zip")
except LookupError:
    nltk.download("punkt")

# Create a corpus from any number of plain .txt files
my_corpus = PlaintextCorpusReader("./", ".*\.txt")
file_ids = "./data/hamlet.txt"

for sent in my_corpus.sents(fileids=file_ids):
    print(sent)

# Pad each side of every line in the corpus with <s> and </s> to indicate the start and end of utterances
padded_trigrams = list(
    pad_both_ends(my_corpus.sents(fileids=file_ids)[1104], n=2)
)
list(everygrams(padded_trigrams, max_len=3))

list(
    flatten(
        pad_both_ends(sent, n=2)
        for sent in my_corpus.sents(fileids=file_ids)
    )
)

# Allow everygrams to create a training set and a vocab object from the data
train, vocab = padded_everygram_pipeline(
    3, my_corpus.sents(fileids=file_ids)
)

# Instantiate and train the model we’ll use for N-Grams, a Maximum Likelihood Estimator (MLE)
# This model will take the everygrams vocabulary, including the <UNK> token used for out-of-vocabulary
lm = MLE(3)
len(lm.vocab)

lm.fit(train, vocab)
print(lm.vocab)
len(lm.vocab)

# And finally, language can be generated with this model and conditioned with n-1 tokens preceding
lm.generate(6, ["to", "be"])

lm.vocab.lookup(my_corpus.sents(fileids=file_ids)[1104])

lm.vocab.lookup(["aliens", "from", "Mars"])

# Any set of tokens up to length=n can be counted easily to determine frequency
print(lm.counts)
lm.counts[["to"]]["be"]

# Any token can be given a probability of occurrence, and can be augmented with up to n-1 tokens to precede it
print(lm.score("be"))
print(lm.score("be", ["to"]))
print(lm.score("be", ["not", "to"]))

# This can be done as a log score as well to avoid very big and very small numbers
print(lm.logscore("be"))
print(lm.logscore("be", ["to"]))
print(lm.logscore("be", ["not", "to"]))

# Sets of tokens can be tested for entropy and perplexity as well
test = [("to", "be"), ("or", "not"), ("to", "be")]
print(lm.entropy(test))
print(lm.perplexity(test))
