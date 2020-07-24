# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:33:13 2019

@author: guill
"""

# On importe les librairies
import nltk
import string
import csv

# On définit le chemin local
local_folder = "/Users/guill/OneDrive/Documents/Recherche/Python/Spyder/gutenberg_fr_analysis/"

# On charge les "stop-words"
with open(local_folder + "stopwords-fr.txt", "r", encoding="utf-8") as stop_word_file:
    stop_word_list = stop_word_file.read()
    stop_word_list = stop_word_list.split("\n")

# On boucle sur les années
for year in range(1780,1880,10):
    
    # On ouvre le document dans "raw_text"
    with open(local_folder + "raw_text/" + str(year) + ".txt", encoding="utf-8") as text_file:
        raw = text_file.read()
    
    # On change les "?, !, :" en points
    raw = raw.translate(str.maketrans("!?:","..."))
    # On sépare les phrases 
    sentence_list = raw.split(".")
    # On enlève le reste de la ponctuation
    sentence_list = [sent.translate(str.maketrans('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~', " "*len('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))) for sent in sentence_list]
    # On transforme chaque phrase en liste de mots
    sentence_token_list = [nltk.word_tokenize(sent.strip()) for sent in sentence_list]
    # On met tout en minuscules, on enlève les nombres, les "stop words" et on ne garde que les pharases de taille plus grande que 10.
    sentence_token_list = [[w.lower() for w in sent if w.isalpha() and not w.lower() in stop_word_list] for sent in sentence_token_list if len(sent) > 10]
    # On sauve nos résultats
    with open(local_folder + "preprocessed_text/" + str(year) + ".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sentence_token_list)