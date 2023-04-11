import numpy as np
from transformers import pipeline

def propose_best_word(context, temperature, tried_words):
    """
    Proposes one of the best words giving :
    - the context sentences with masks at some positions
    - a temperature parameter
    Updates the list of words that were already tried
    """

    # Loading mask filler
    camembert_fill_mask  = pipeline("fill-mask", model="camembert/camembert-base-wikipedia-4gb", tokenizer="camembert/camembert-base-wikipedia-4gb", top_k = 10)

    # Giving best word
    results = camembert_fill_mask(context)
    best_results = sorted([(mot['token_str'], mot['score']) for blank in results for mot in blank], key = lambda x: x[1], reverse = True)
    random_position = np.random.poisson(temperature)
    best_word = best_results[random_position][0]

    while best_word in tried_words:
        best_results.pop(random_position)
        random_position = np.random.poisson(temperature)
        best_word = best_results[random_position][0]

    # Updating tried words
    tried_words.append(best_word)

    return best_word, tried_words

def test_propose_best_word():
    context = "Le <mask> dans le <mask> rouge"
    temperature = 3
    tried_words = ['film', 'cinéma', 'Venise']
    print(propose_best_word(context, temperature, tried_words))

def update(titre, titre_masked, article, article_masked, suggestion):
    """
    After receiving a suggestion, updates the article
    """

    # Updates titre
    for i, (word, _) in enumerate(zip(titre.split(), titre_masked.split())):
        if word == suggestion:
            titre_masked_list = titre_masked.split()
            titre_masked_list[i] = suggestion
            titre_masked = " ".join(titre_masked_list)

    # Updates article
    for i, (word, _) in enumerate(zip(article.split(), article_masked.split())):
        if word == suggestion:
            article_masked_list = article_masked.split()
            article_masked_list[i] = suggestion
            article_masked = " ".join(article_masked_list)

    return titre, titre_masked, article, article_masked

def test_update():
    titre = "Le Rêve dans le pavillon rouge"
    article = "Le Rêve dans le pavillon rouge écrit en l'espace de dix ans par Cao Xueqin, est le dernier en date des quatre grands romans de la littérature classique chinoise, considéré par Mao Zedong comme l'une des fiertés de la Chine."
    titre_masked = "Le <mask> dans le <mask> <mask>"
    article_masked = "Le <mask> dans le <mask> <mask> <mask> en l'<mask> de <mask> <mask> par <mask> <mask>, est le <mask> en <mask> des <mask> <mask> <mask> de la <mask> <mask> <mask>, <mask> par <mask> <mask> comme l'une des <mask> de la <mask>."
    print(article_masked)
    suggestion = 'considéré'
    titre, titre_masked, article, article_masked = update(titre, titre_masked, article, article_masked, suggestion)
    print(article_masked)
