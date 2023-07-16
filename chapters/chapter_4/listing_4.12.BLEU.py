import nltk.translate.bleu_score as bleu

target = [
    "The game 'The Legend of Zelda' follows the adventures of the \
    hero Link in the magical world of Hyrule.".split(),
    "Link goes on awesome quests and battles evil forces to \
    save Princess Zelda and restore peace to Hyrule.".split(),
]
prediction = "Link embarks on epic quests and battles evil forces to \
    save Princess Zelda and restore peace in the land of Hyrule.".split()


score = bleu.sentence_bleu(target, prediction)
print(score)  # 0.6187934993051339
