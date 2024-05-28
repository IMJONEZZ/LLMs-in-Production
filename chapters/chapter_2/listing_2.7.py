import torch
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import nltk
import spacy

try:
    tokenizer = nltk.tokenize.RegexpTokenizer("\w+'?\w+|\w+'")
    tokenizer.tokenize("This is a test")
    stop_words = nltk.corpus.stopwords.words("english")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger", "ner"])
except Exception:
    nltk.download("stopwords")
    nltk.download("punkt")
    spacy.cli.download("en_core_web_lg")
    tokenizer = nltk.tokenize.RegexpTokenizer("\w+'?\w+|\w+'")
    tokenizer.tokenize("This is a test")
    stop_words = nltk.corpus.stopwords.words("english")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger", "ner"])

# Create our corpus for training and perform some classic NLP preprocessing
dataset = pd.read_csv("./data/twitter.csv")

text_data = list(
    map(lambda x: tokenizer.tokenize(x.lower()), dataset["text"])
)
text_data = [
    [token.lemma_ for word in text for token in nlp(word)]
    for text in text_data
]
label_data = list(map(lambda x: x, dataset["feeling"]))
assert len(text_data) == len(
    label_data
), f"{len(text_data)} does not equal {len(label_data)}"

EMBEDDING_DIM = 100
model = Word2Vec(
    text_data, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4
)
word_vectors = model.wv
print(f"Vocabulary Length: {len(model.wv)}")
del model

padding_value = len(word_vectors.index_to_key)
# Embeddings are needed to give semantic value to the inputs of an LSTM
embedding_weights = torch.Tensor(word_vectors.vectors)


class RNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        embedding_weights,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embedding_weights
        )
        self.rnn = torch.nn.RNN(embedding_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, text_lengths):
        embedded = self.embedding(x)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths
        )
        packed_output, hidden = self.rnn(packed_embedded)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output
        )
        return self.fc(hidden.squeeze(0))


INPUT_DIM = padding_value
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

rnn_model = RNN(
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, embedding_weights
)

rnn_optimizer = torch.optim.SGD(rnn_model.parameters(), lr=1e-3)
rnn_criterion = torch.nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        embedding_weights,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embedding_weights
        )
        self.rnn = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, text_lengths):
        embedded = self.embedding(x)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths
        )
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        )
        return self.fc(hidden.squeeze(0))


INPUT_DIM = padding_value
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

lstm_model = LSTM(
    INPUT_DIM,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    N_LAYERS,
    BIDIRECTIONAL,
    DROPOUT,
    embedding_weights,
)

lstm_optimizer = torch.optim.Adam(lstm_model.parameters())
lstm_criterion = torch.nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch["text"], batch["length"]).squeeze(1)
        loss = criterion(predictions, batch["label"])
        acc = binary_accuracy(predictions, batch["label"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch["text"], batch["length"]).squeeze(1)
            loss = criterion(predictions, batch["label"])
            acc = binary_accuracy(predictions, batch["label"])

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


batch_size = 2  # Usually should be a power of 2 because it's the easiest for computer memory.


def iterator(X, y):
    size = len(X)
    permutation = np.random.permutation(size)
    iterate = []
    for i in range(0, size, batch_size):
        indices = permutation[i : i + batch_size]
        batch = {}
        batch["text"] = [X[i] for i in indices]
        batch["label"] = [y[i] for i in indices]

        batch["text"], batch["label"] = zip(
            *sorted(
                zip(batch["text"], batch["label"]),
                key=lambda x: len(x[0]),
                reverse=True,
            )
        )
        batch["length"] = [len(utt) for utt in batch["text"]]
        batch["length"] = torch.IntTensor(batch["length"])
        batch["text"] = torch.nn.utils.rnn.pad_sequence(
            batch["text"], batch_first=True
        ).t()
        batch["label"] = torch.Tensor(batch["label"])

        batch["label"] = batch["label"].to(device)
        batch["length"] = batch["length"].to(device)
        batch["text"] = batch["text"].to(device)

        iterate.append(batch)

    return iterate


index_utt = [
    torch.tensor([word_vectors.key_to_index.get(word, 0) for word in text])
    for text in text_data
]

# You've got to determine some labels for whatever you're training on.
X_train, X_test, y_train, y_test = train_test_split(
    index_utt, label_data, test_size=0.2
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2
)

train_iterator = iterator(X_train, y_train)
validate_iterator = iterator(X_val, y_val)
test_iterator = iterator(X_test, y_test)

print(len(train_iterator), len(validate_iterator), len(test_iterator))


N_EPOCHS = 25

for model in [rnn_model, lstm_model]:
    print(
        "|-----------------------------------------------------------------------------------------|"
    )
    print(f"Training with {model.__class__.__name__}")
    if "RNN" in model.__class__.__name__:
        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train(
                rnn_model, train_iterator, rnn_optimizer, rnn_criterion
            )
            valid_loss, valid_acc = evaluate(
                rnn_model, validate_iterator, rnn_criterion
            )

            print(
                f"| Epoch: {epoch+1:02} | Train Loss: {train_loss: .3f} | Train Acc: {train_acc*100: .2f}% | Validation Loss: {valid_loss: .3f} | Validation Acc: {valid_acc*100: .2f}% |"
            )
    else:
        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train(
                lstm_model, train_iterator, lstm_optimizer, lstm_criterion
            )
            valid_loss, valid_acc = evaluate(
                lstm_model, validate_iterator, lstm_criterion
            )

            print(
                f"| Epoch: {epoch+1:02} | Train Loss: {train_loss: .3f} | Train Acc: {train_acc*100: .2f}% | Validation Loss: {valid_loss: .3f} | Validation Acc: {valid_acc*100: .2f}% |"
            )
