# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:01:39 2019

@author: guill
"""

# On charge les librairies
import csv
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Pour voir combien de "threads" sont disponibles
#import multiprocessing
#multiprocessing.cpu_count()

# On définit le dossier local
local_folder = "/Users/guill/OneDrive/Documents/Recherche/Python/Spyder/gutenberg_fr_analysis/"

# On ouvre la première année
with open(local_folder + "preprocessed_text/1780.csv", "r") as csvfile:
    text_data = list(csv.reader(csvfile))
    
# On fait le model
model = Word2Vec(   text_data, 
                    min_count=100, 
                    window=5,
                    size= 300,
                    sample=6e-5,
                    alpha=0.03,
                    negative=20,
                    workers=6,
                    sg = 1)

# On obtient le vocabulaire
words = list(model.wv.vocab.keys())
len(words)

# On obtient les fréquences
freq = [model.wv.vocab[word].count for word in words]

# On fait un dataframe words - frequency, on le classe par fréquence
word_freq_df = pd.DataFrame({"word": words, "freq": freq})
word_freq_df = word_freq_df.sort_values("freq", ascending=False)

# Pour vérifier qu'un mot appartient au vocabulaire
"seigneur" in words
"enfant" in words
"homme" in words
"femme" in words

# On regarde les voisins et les mots les plus éloigné
model.wv.most_similar(positive=["seigneur"])
model.wv.most_similar(negative=["seigneur"])
model.wv.most_similar(positive=["enfant"])
model.wv.most_similar(positive=["homme"])
model.wv.most_similar(positive=["femme"])

# Pour voir si un mot est plus proche d'un autre
model.wv.similarity("enfant","homme")
model.wv.similarity("enfant","femme")

# Faire des analogies
model.wv.most_similar(positive=["père", "femme"], negative=["homme"])
model.wv.most_similar(positive=["hommes","femme"], negative=["homme"])

###### Graphique des 50 mots les plus fréquent

# On prend les 50 mots les plus fréquents
freq_words = list(word_freq_df[0:49].word)

# On remplit la liste avec les vecteurs
vector_array = np.empty((0, 300), dtype='f')
for w in freq_words:
    vector_array = np.append(vector_array, model.wv.__getitem__([w]), axis=0)
vector_array_2 = model.wv.__getitem__(freq_words) # à comparer

# On réduit la dimensionalité
coords = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(vector_array)
x_coords = coords[:,0]
y_coords = coords[:,1]

# On plot
plt.scatter(x_coords, y_coords)
for label, x, y in zip(freq_words, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
plt.show()
    
###### Graphique des 50 mots les plus proche d'un mot

# Notre mot
word_to_study = "paysan"

# Les voisins
neighbour_words = [item[0] for item in model.wv.most_similar(word_to_study, topn=50)]
all_words = [word_to_study]
all_words.extend(neighbour_words)

# On remplit la liste avec les vecteurs
vector_array = np.empty((0, 300), dtype='f')
for w in all_words:
    vector_array = np.append(vector_array, model.wv.__getitem__([w]) - model.wv.__getitem__([word_to_study]), axis=0)

# On réduit la dimensionalité
coords = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(vector_array)
x_coords = coords[:,0]
y_coords = coords[:,1]

# On plot
plt.scatter(x_coords, y_coords)
for label, x, y in zip(all_words, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
plt.show()


