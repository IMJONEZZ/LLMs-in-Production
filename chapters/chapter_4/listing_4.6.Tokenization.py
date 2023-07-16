from pathlib import Path
import transformers
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer
import os
from tokenizers.processors import BertProcessing

# Initialize the txts to train from
paths = [str(x) for x in Path("./chapters/chapter_2/").glob("**/*.txt")]

# Train a BPE tokenizer
bpe_tokenizer = ByteLevelBPETokenizer()

bpe_tokenizer.train(
    files=paths,
    vocab_size=52_000,
    min_frequency=2,
    show_progress=True,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

token_dir = "./chapters/chapter_4/tokenizers/bytelevelbpe/"
if not os.path.exists(token_dir):
    os.makedirs(token_dir)
bpe_tokenizer.save_model("./chapters/chapter_4/tokenizers/bytelevelbpe/")

bpe_tokenizer = ByteLevelBPETokenizer(
    "./chapters/chapter_4/tokenizers/bytelevelbpe/vocab.json",
    "./chapters/chapter_4/tokenizers/bytelevelbpe/merges.txt",
)

print(
    bpe_tokenizer.encode(
        "This sentence is getting encoded by a tokenizer."
    ).tokens
)
print(
    bpe_tokenizer.encode("This sentence is getting encoded by a tokenizer.")
)

bpe_tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", bpe_tokenizer.token_to_id("</s>")),
    ("<s>", bpe_tokenizer.token_to_id("<s>")),
)
bpe_tokenizer.enable_truncation(max_length=512)


# Train a Sentencepiece Tokenizer
special_tokens = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<cls>",
    "<sep>",
    "<mask>",
]
sentencepiece_tokenizer = SentencePieceBPETokenizer()
sentencepiece_tokenizer.train(
    files=paths,
    vocab_size=4000,
    min_frequency=2,
    show_progress=True,
    special_tokens=special_tokens,
)
sentencepiece_tokenizer.save_model(
    "./chapters/chapter_4/tokenizers/sentencepiece/"
)
# convert
tokenizer = transformers.PreTrainedTokenizerFast(
    tokenizer_object=sentencepiece_tokenizer,
    model_max_length=512,
    special_tokens=special_tokens,
)
tokenizer.bos_token = "<s>"
tokenizer.bos_token_id = sentencepiece_tokenizer.token_to_id("<s>")
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = sentencepiece_tokenizer.token_to_id("<pad>")
tokenizer.eos_token = "</s>"
tokenizer.eos_token_id = sentencepiece_tokenizer.token_to_id("</s>")
tokenizer.unk_token = "<unk>"
tokenizer.unk_token_id = sentencepiece_tokenizer.token_to_id("<unk>")
tokenizer.cls_token = "<cls>"
tokenizer.cls_token_id = sentencepiece_tokenizer.token_to_id("<cls>")
tokenizer.sep_token = "<sep>"
tokenizer.sep_token_id = sentencepiece_tokenizer.token_to_id("<sep>")
tokenizer.mask_token = "<mask>"
tokenizer.mask_token_id = sentencepiece_tokenizer.token_to_id("<mask>")
# and save for later!
tokenizer.save_pretrained("./chapters/chapter_4/tokenizers/sentencepiece")

print(tokenizer.encode("This sentence is getting encoded by a tokenizer."))
