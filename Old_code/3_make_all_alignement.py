# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:29:51 2019

@author: guill
"""
# On charge les librairies
import csv
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
from scipy.linalg import orthogonal_procrustes
import numpy as np


# On définit le dossier local
local_folder = "/Users/guill/OneDrive/Documents/Recherche/Python/Spyder/gutenberg_fr_analysis/"

# On ouvre le premier document
with open(local_folder + "preprocessed_text/1780.csv", "r") as csvfile:
    text_data = list(csv.reader(csvfile))

# On fait le modèle (de référence)
pre_model = Word2Vec(text_data, 
                    min_count=100, 
                    window=4,
                    size= 300,
                    sample=1e-4,
                    alpha=0.02, 
                    negative=20,
                    workers=8,
                    sg = 1,
                    iter=10)

# On standardise les points (à faire que dans les petits datasets)
# pre_model.wv.vectors = scale(pre_model.wv.vectors, axis=0)
# On sauve le modèle
pre_model.save(local_folder + "aligned_models/1780.model")

# On boucle sur toutes les autres années
for year in range(1790,1880,10):
    
    # On ouvre le fichier
    with open(local_folder + "preprocessed_text/" + str(year) + ".csv", "r") as csvfile:
        text_data = list(csv.reader(csvfile))
    
    # On fait le modèle
    act_model = Word2Vec(text_data, 
                    min_count=100, 
                    window=4,
                    size= 300,
                    sample=1e-4,
                    alpha=0.02, 
                    negative=20,
                    workers=8,
                    sg = 1,
                    iter=10)
    
    # On crée le vocabulaire commun
    vocab_m1 = pre_model.wv.vocab.keys()
    vocab_m2 = act_model.wv.vocab.keys()
    common_vocab = list(vocab_m1&vocab_m2)
    
    # On standardise les points (à faire que dans les petits datasets)
    act_model.wv.vectors = scale(act_model.wv.vectors, axis=0)
    # On aligne sur le modèle précédent choix #1
    M, _ = orthogonal_procrustes(act_model.wv.__getitem__(common_vocab), pre_model.wv.__getitem__(common_vocab))
    act_model.wv.vectors = act_model.wv.vectors.dot(M)
    # On aligne sur le modèle précédent choix #2
    #V = act_model.wv.__getitem__(common_vocab)
    #M = (np.linalg.inv(np.transpose(V).dot(V)).dot(np.transpose(V))).dot(pre_model.wv.__getitem__(common_vocab))
    #act_model.wv.vectors = act_model.wv.vectors.dot(M)
    
    # On sauve le modèle aligné et on le met comme modèle de référence
    act_model.save(local_folder + "aligned_models/" + str(year) + ".model")
    pre_model = act_model
    

    
    