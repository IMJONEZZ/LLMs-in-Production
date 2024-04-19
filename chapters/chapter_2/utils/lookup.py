def lookup(freqs, word, label):
    """
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    """
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]

    return n
