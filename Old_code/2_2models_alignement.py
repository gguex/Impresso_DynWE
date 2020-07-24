# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:15:34 2019

@author: guill
"""

# On importe les librairies
import csv
from gensim.models import Word2Vec
import numpy as np
from scipy.linalg import orthogonal_procrustes
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# On définit le dossier local
local_folder = "/Users/guill/OneDrive/Documents/Recherche/Python/Spyder/gutenberg_fr_analysis/"

# On ouvre le premier document
with open(local_folder + "preprocessed_text/1780.csv", "r") as csvfile:
    text_data = list(csv.reader(csvfile))
    
# On fait le modèle
model1 = Word2Vec(   text_data, 
                    min_count=100, 
                    window=5,
                    size= 300,
                    sample=6e-5,
                    alpha=0.03, 
                    negative=20,
                    workers=6,
                    sg = 1)

# On ouvre le deuxième document
with open(local_folder + "preprocessed_text/1790.csv", "r") as csvfile:
    text_data = list(csv.reader(csvfile))
    
# On fait le modèle
model2 = Word2Vec(   text_data, 
                    min_count=100, 
                    window=5,
                    size= 300,
                    sample=6e-5,
                    alpha=0.03, 
                    negative=20,
                    workers=6,
                    sg = 1)

# On copie les vocabulaires
vocab_m1 = model1.wv.vocab.keys()
vocab_m2 = model2.wv.vocab.keys()

# On sauve le vocabulaire commun 
common_vocab = list(vocab_m1&vocab_m2)

# On centre et standardises les points (à faire que dans les petits datasets)
#model1.wv.vectors = scale(model1.wv.vectors, axis=0)
#model2.wv.vectors = scale(model2.wv.vectors, axis=0)
# On fait l'alignement choix #1
M, _ = orthogonal_procrustes(model2.wv.__getitem__(common_vocab), model1.wv.__getitem__(common_vocab))
model2.wv.vectors = model2.wv.vectors.dot(M)
# On fait l'alignement choix #2
#V = model1.wv.__getitem__(common_vocab)
#M = (np.linalg.inv(np.transpose(V).dot(V)).dot(np.transpose(V))).dot(model2.wv.__getitem__(common_vocab))
#model1.wv.vectors = model1.wv.vectors.dot(M)

# On fait une dataframe words - frequency, on sort par le total
freq_m1 = [model1.wv.vocab[key].count for key in common_vocab]
freq_m2 = [model2.wv.vocab[key].count for key in common_vocab]
word_freq_df = pd.DataFrame({"word": common_vocab, "freq_m1": freq_m1, "freq_m2": freq_m2})
word_freq_df["TOTAL"] = word_freq_df.sum(axis=1)
word_freq_df = word_freq_df.sort_values("TOTAL", ascending=False)

# On plot les mots les plus fréquents
top50_words = list(word_freq_df[0:50].word)
high_dim_coords = np.concatenate((model1.wv.__getitem__(top50_words), model2.wv.__getitem__(top50_words)))
coords = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(high_dim_coords)
x_coords = coords[:,0]
y_coords = coords[:,1]
top50_words2 = top50_words[:]
top50_words2.extend(top50_words)

# Plotting
plt.scatter(x_coords, y_coords, c=list(np.append(np.repeat(1,50),np.repeat(2,50))))
for label, x, y in zip(top50_words2, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
for i in range(50):
    plt.plot((x_coords[i],x_coords[i+50]),(y_coords[i],y_coords[i+50]))
plt.show()







