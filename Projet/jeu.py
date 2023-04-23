import json
from transformers import pipeline
from utils import mask_sentence, propose_best_word, render_list_as_text, update

# Initialisation
article_name = 'billy_wilder'
temperature = 10
nb_essais = 20

with open(f'articles/{article_name}.json', 'r', encoding = 'utf8') as f:
    data = json.load(f)

titre = data['titre']
article = data['article']

titre_masked, titre_dico = mask_sentence(titre)
article_masked, article_dico = mask_sentence(article)

print(render_list_as_text(titre_masked))
print(render_list_as_text(article_masked))

nb_try = 0
nb_good_answer = 0

# Loading included words
with open('included_words.txt', 'r') as file:
    # Read the lines from the file
    lines = file.readlines()
    
    # Remove any leading or trailing whitespaces from each line
    tried_words = [line.strip().strip('"\',') for line in lines]

llm_model  = pipeline("fill-mask", model="camembert/camembert-base-wikipedia-4gb", tokenizer="camembert/camembert-base-wikipedia-4gb", top_k = 20)

for _ in range(nb_essais):

    # Listen to the suggested word
    suggestion, tried_words = propose_best_word(render_list_as_text(article_masked), temperature, tried_words, llm_model)
    print(suggestion)

    # Update article and title
    titre_dico, titre_masked, article_dico, article_masked, good_answer = update(titre_dico, titre_masked, article_dico, article_masked, suggestion)

    if titre == titre_masked:
        print("Gagné !")
        break

    nb_try += 1
    if good_answer: nb_good_answer+=1

print(render_list_as_text(titre_masked))
print(render_list_as_text(article_masked))

print("Nombre d'essais : {}".format(nb_try))
print(f"Accuracy : {nb_good_answer/nb_try}")