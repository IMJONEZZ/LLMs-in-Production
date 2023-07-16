from rouge_score import rouge_scorer

target = "The game 'The Legend of Zelda' follows the adventures of the \
    hero Link in the magical world of Hyrule."
prediction = "Link embarks on epic quests and battles evil forces to \
    save Princess Zelda and restore peace in the land of Hyrule."

# Example N-gram where N=1 and also using the longest common subsequence
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
scores = scorer.score(target, prediction)
print(scores)
# {'rouge1': Score(precision=0.28571428, recall=0.31578947, fmeasure=0.3),
# 'rougeL': Score(precision=0.238095238, recall=0.26315789, fmeasure=0.25)}
