# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:32:36 2019

@author: guill
"""

# On importe les librairies
from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# On définit le chemin local
local_folder = "/Users/guill/OneDrive/Documents/Recherche/Python/Spyder/gutenberg_fr_analysis/"

# On charge les modèles
models = []
years = [i for i in range(1780,1880,10)]
for i in range(len(years)):
    models.append(Word2Vec.load(local_folder + "aligned_models/" + str(years[i]) + ".model"))
   
# On récupère la dimension de l'embedding
_, model_dim = models[0].wv.vectors.shape

# On fait le vocabulaire commun
common_vocab = models[0].wv.vocab.keys()
for i in range(1,len(years)):
    new_voc = models[i].wv.vocab.keys()
    common_vocab = common_vocab & new_voc
common_vocab = list(common_vocab)

# On sauve le vocabulaire commun
#with open(local_folder + "your_file.txt", 'w') as f:
#    for item in common_vocab:
#        f.write("%s " % item)

# On obtient les fréquences
word_freq_df = pd.DataFrame({"word": common_vocab})
for i in range(len(years)):
    word_freq_df["freq_" + str(years[i])] = [models[i].wv.vocab[key].count for key in common_vocab]
word_freq_df["total"] = word_freq_df.sum(axis=1)
word_freq_df = word_freq_df.sort_values("total", ascending=False)
frequent_words = word_freq_df.loc[word_freq_df["total"] > 10000]["word"]

# On enregistre les distances du cosinus entre les embeddings 
word_dist_df = pd.DataFrame({"word": frequent_words})
for i in range(len(years)-1):
        word_init_pos = models[i-1].wv.__getitem__(frequent_words)
        word_final_pos = models[i].wv.__getitem__(frequent_words)
        dist_matrix = 1 - cosine_similarity(word_init_pos, word_final_pos)
        word_dist_df["delta_" + str(years[i])] = np.diagonal(dist_matrix)
word_dist_df["total"] = word_dist_df.sum(axis=1)        
word_dist_df = word_dist_df.sort_values("total", ascending=False)
word_dist_df[0:50].word

# Avec un mot, on trouve sa trajectoire et ses voisins
subject_word = "ouvrier"
no_neighbour = 10
df_word_neighbour = pd.DataFrame()
list_of_words = []
color_list = []
high_dim_coords = np.empty((0, model_dim), dtype='f')
for i in range(len(years)):
    list_of_words.append(subject_word)
    high_dim_coords =  np.append(high_dim_coords, models[i].wv.__getitem__([subject_word]), axis=0)
    nearest_word_list = models[i].wv.most_similar(subject_word)[:no_neighbour]
    neighbout_list = [element[0] for element in nearest_word_list]
    list_of_words.extend(neighbout_list)
    high_dim_coords = np.append(high_dim_coords, models[i].wv.__getitem__(neighbout_list), axis=0)
    df_word_neighbour[years[i]] = neighbout_list
    color_list = np.append(color_list, np.repeat(i, no_neighbour + 1))
pd.set_option('display.max_columns', 12)
print(df_word_neighbour)

# On plot sa trajectoire
coords = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(high_dim_coords)
x_coords = coords[:,0]
y_coords = coords[:,1]
plt.scatter(x_coords, y_coords, c=list(color_list))
for label, x, y in zip(list_of_words, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
for i in range(len(years)-1):
    plt.plot((x_coords[i*11],x_coords[i*11 + 11]),(y_coords[i*11],y_coords[i*11 + 11]))
plt.show()





