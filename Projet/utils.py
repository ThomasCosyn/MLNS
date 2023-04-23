import nltk
import numpy as np
from transformers import pipeline

def create_mask_filler():
    """
    Imports the LLM
    """

    camembert_fill_mask  = pipeline("fill-mask", model="camembert/camembert-base-wikipedia-4gb", tokenizer="camembert/camembert-base-wikipedia-4gb", top_k = 20)

    return camembert_fill_mask

def mask_sentence(sentence):
    """
    Transforms a text sentence into a list of masked and unmasked word tokens
    Also returns a dictionary containing the words of the article with their positions
    """

    # Loading included words
    with open('included_words.txt', 'r') as file:
        # Read the lines from the file
        lines = file.readlines()
        
        # Remove any leading or trailing whitespaces from each line
        words = [line.strip().strip('"\',') for line in lines]
    
    # Split with a word tokenizer
    splitted_sentence = nltk.word_tokenize(sentence, language = "french")

    # Re split to avoid words stuck to an apostrophe
    second_split = []
    for elem in splitted_sentence:
        elem_split = elem.split("'")
        if len(elem_split) > 1:
            second_split.append(elem_split[0])
            second_split.append("'")
            second_split.append(elem_split[1])
        else:
            second_split.append(elem_split[0])
    
    # Re split to separate words by hyphen
    third_split = []
    for elem in second_split:
        elem_split = elem.split("'")
        if len(elem_split) > 1:
            third_split.append(elem_split[0])
            third_split.append("'")
            third_split.append(elem_split[1])
        else:
            third_split.append(elem_split[0])

    # Build dictionary of words' positions
    words_dico = {}
    for i, word in enumerate(second_split):
        if word not in words_dico.keys():
            words_dico[word] = [i]
        else:
            words_dico[word].append(i) 

    # Build masked list
    sentence_list_mask = ["<mask>"] * len(second_split)
    for word in words_dico.keys():
        if word in words:
            for index in words_dico[word]:
                sentence_list_mask[index] = word

    return sentence_list_mask, words_dico

def propose_best_word(context, temperature, tried_words, llm_model):
    """
    Proposes one of the best words giving :
    - the context sentences with masks at some positions
    - a temperature parameter
    Updates the list of words that were already tried
    """

    # Giving best words
    results = llm_model(context)
    best_results = sorted([(mot['token_str'], mot['score']) for blank in results for mot in blank], key = lambda x: x[1], reverse = True)
    
    # Suppress doubles
    unique_dict = {}
    for item in best_results:
        key = item[0]
        value = item[1]
        if key not in unique_dict:
            unique_dict[key] = value
    best_results = [(key, value) for key, value in unique_dict.items()]

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

def render_list_as_text(liste):
    """
    Takes the masked list and renders it as text
    """
    return " ".join(liste)

def update(titre_dico, titre_masked, article_dico, article_masked, suggestion):
    """
    After receiving a suggestion, updates the article
    """

    good_answer = False

    # Update titre
    if suggestion in titre_dico.keys():
        for index in titre_dico[suggestion]:
            titre_masked[index] = suggestion
        print(render_list_as_text(titre_masked))
        good_answer = True

    # Update article
    if suggestion in article_dico.keys():
        for index in article_dico[suggestion]:
            article_masked[index] = suggestion
        print(render_list_as_text(article_masked))
        good_answer = True

    return titre_dico, titre_masked, article_dico, article_masked, good_answer

def test_update():
    titre = "Le Rêve dans le pavillon rouge"
    article = "Le Rêve dans le pavillon rouge écrit en l'espace de dix ans par Cao Xueqin, est le dernier en date des quatre grands romans de la littérature classique chinoise, considéré par Mao Zedong comme l'une des fiertés de la Chine."
    titre_masked = "Le <mask> dans le <mask> <mask>"
    article_masked = "Le <mask> dans le <mask> <mask> <mask> en l'<mask> de <mask> <mask> par <mask> <mask>, est le <mask> en <mask> des <mask> <mask> <mask> de la <mask> <mask> <mask>, <mask> par <mask> <mask> comme l'une des <mask> de la <mask>."
    print(article_masked)
    suggestion = 'considéré'
    titre, titre_masked, article, article_masked = update(titre, titre_masked, article, article_masked, suggestion)
    print(article_masked)
