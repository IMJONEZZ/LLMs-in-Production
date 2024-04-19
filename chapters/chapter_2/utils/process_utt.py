import re
import string

from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_utt(utt):
    """
    Input:
        utt: a string containing a utt
    Output:
        utts_clean: a list of words containing the processed utt
    """
    stemmer = PorterStemmer()
    # stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    utt = re.sub(r"\$\w*", "", utt)
    # remove old style utt text "RT"
    utt = re.sub(r"^RT[\s]+", "", utt)
    # remove hyperlinks
    utt = re.sub(r"https?:\/\/.*[\r\n]*", "", utt)
    # remove hashtags
    # only removing the hash # sign from the word
    utt = re.sub(r"#", "", utt)
    # tokenize utts
    tokenizer = TweetTokenizer(
        preserve_case=False, strip_handles=True, reduce_len=True
    )
    utt_tokens = tokenizer.tokenize(utt)

    utts_clean = []
    for word in utt_tokens:
        if word not in string.punctuation:  # remove punctuation
            # utts_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            utts_clean.append(stem_word)

    return utts_clean
