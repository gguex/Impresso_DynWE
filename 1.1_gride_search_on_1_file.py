# Test WE synchonique
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import numpy as np
import os
import datetime
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity

# Corpus file path
file_path = "/home/gguex/Documents/data/impresso/JDG_line_sent/JDG-1997_sent.txt"

# Name of output_file
output_file = "Gride_search_results.txt"

if not os.path.exists(output_file):
    with open(output_file, "w") as file:
        file.write("Création du fichier le {0}\n".format(datetime.datetime.now()))

with open(output_file, "a") as file:
    file.write("Gride search sur le fichier {0} le {1}\n".format(file_path, datetime.datetime.now()))
    file.write("win_size;vec_size;neg_sam;ns_exp;samp_size;" +
               "n_mc;n_rg;n_simlex;n_ws353;n_msimlex;n_all;n_cosine;" +
               "pval_mc;pval_rg;pval_simlex;pval_ws353;pval_msimlex;pval_all;sd_cosine;" +
               "cor_mc;cor_rg;cor_simlex;cor_ws353;cor_msimlex;cor_all;mean_cosine" +
               "\n")

# Words analogy datasets
fr_mc = pd.read_csv("/home/gguex/Documents/data/eval_datasets/fr_mc.csv")
fr_rg = pd.read_csv("/home/gguex/Documents/data/eval_datasets/fr_rg.csv")
fr_simlex = pd.read_csv("/home/gguex/Documents/data/eval_datasets/fr_simlex.csv")
fr_ws353 = pd.read_csv("/home/gguex/Documents/data/eval_datasets/fr_ws353.csv")
fr_msimlex = pd.read_csv("/home/gguex/Documents/data/eval_datasets/multi_simlex.csv")
fr_all = pd.read_csv("/home/gguex/Documents/data/eval_datasets/fr_all.csv")

# Load wiki model
wv_wiki = KeyedVectors.load("/home/gguex/Documents/data/pretrained_word_vectors/frwiki.model")

tested_windows_values = [3, 5, 7, 10, 12, 15]
tested_vectors_size = [300]
tested_negative_sample = [10, 15, 20, 25, 30]
tested_ns_exponent = [0.85, 0.75, 0.65]
tested_sample_size = [1e-3, 1e-4, 1e-5]

for vec_size in tested_vectors_size:
    for win_size in tested_windows_values:
        for neg_sam in tested_negative_sample:
            for ns_exp in tested_ns_exponent:
                for samp_size in tested_sample_size:
                    # Print
                    print("Test with {0} win_size, {1} vec_size, {2} neg_sam, {3} ns_exp, {4} samp_size: ".format(
                        win_size, vec_size,
                        neg_sam, ns_exp, samp_size))

                    # Model training
                    model = Word2Vec(corpus_file=file_path,
                                     min_count=100,
                                     window=win_size,
                                     size=vec_size,
                                     sample=samp_size,
                                     negative=neg_sam,
                                     alpha=0.025,
                                     ns_exponent=ns_exp,
                                     workers=8,
                                     sg=1)

                    # Get the vocabulary
                    words = list(model.wv.vocab.keys())

                    # Select rows which word1 and word2 are in the vocabulary, compute similarity and correlation

                    # MC
                    presence_mc = fr_mc[fr_mc["word1"].isin(words) & fr_mc["word2"].isin(words)]
                    n_mc = presence_mc.shape[0]
                    sim_mc = [model.wv.similarity(row["word1"], row["word2"])
                              for _, row in presence_mc.iterrows()]
                    cor_mc, pval_mc = stats.spearmanr(presence_mc["score"], np.array(sim_mc))

                    # RG
                    presence_rg = fr_rg[fr_rg["word1"].isin(words) & fr_rg["word2"].isin(words)]
                    n_rg = presence_rg.shape[0]
                    sim_rg = [model.wv.similarity(row["word1"], row["word2"])
                              for _, row in presence_rg.iterrows()]
                    cor_rg, pval_rg = stats.spearmanr(presence_rg["score"], np.array(sim_rg))

                    # SIMLEX
                    presence_simlex = fr_simlex[fr_simlex["word1"].isin(words) & fr_simlex["word2"].isin(words)]
                    n_simlex = presence_simlex.shape[0]
                    sim_simlex = [model.wv.similarity(row["word1"], row["word2"])
                                  for _, row in presence_simlex.iterrows()]
                    cor_simlex, pval_simlex = stats.spearmanr(presence_simlex["score"], np.array(sim_simlex))

                    # WS353
                    presence_ws353 = fr_ws353[fr_ws353["word1"].isin(words) & fr_ws353["word2"].isin(words)]
                    n_ws353 = presence_ws353.shape[0]
                    sim_ws353 = [model.wv.similarity(row["word1"], row["word2"])
                                 for _, row in presence_ws353.iterrows()]
                    cor_ws353, pval_ws353 = stats.spearmanr(presence_ws353["score"], np.array(sim_ws353))

                    # MSIMLEX
                    presence_msimlex = fr_msimlex[
                        fr_msimlex["word1"].isin(words) & fr_msimlex["word2"].isin(words)]
                    n_msimlex = presence_msimlex.shape[0]
                    sim_msimlex = [model.wv.similarity(row["word1"], row["word2"])
                                   for _, row in presence_msimlex.iterrows()]
                    cor_msimlex, pval_msimlex = stats.spearmanr(presence_msimlex["score"], np.array(sim_msimlex))

                    # ALL
                    presence_all = fr_all[fr_all["word1"].isin(words) & fr_all["word2"].isin(words)]
                    n_all = presence_all.shape[0]
                    sim_all = [model.wv.similarity(row["word1"], row["word2"])
                              for _, row in presence_all.iterrows()]
                    cor_all, pval_all = stats.spearmanr(presence_all["score"], np.array(sim_all))

                    # Compute the similarity with the wiki model
                    # Get the vocab for each model
                    vocab_wiki = set(wv_wiki.vocab.keys())
                    vocab_model = set(model.wv.vocab.keys())
                    # Find the common vocabulary
                    common_vocab = list(vocab_wiki & vocab_model)

                    # Make the orthogonal_procrustes alignment
                    wiki_matrix = wv_wiki.__getitem__(common_vocab)
                    model_matrix = model.wv.__getitem__(common_vocab)
                    M, _ = orthogonal_procrustes(model_matrix, wiki_matrix)
                    model_matrix_aligned = model.wv.__getitem__(common_vocab).dot(M)

                    # Compute the cosine
                    cos_mat = cosine_similarity(wiki_matrix, model_matrix_aligned)
                    vec_cosine = cos_mat.diagonal()
                    mean_cosine = vec_cosine.mean()
                    sd_cosine = vec_cosine.std()
                    n_cosine = len(vec_cosine)

                    # Print result
                    print("MC: correlation = {0}, p-value = {1} (n_test = {2})".format(cor_mc, pval_mc, n_mc))
                    print("RG: correlation = {0}, p-value = {1} (n_test = {2})".format(cor_rg, pval_rg, n_rg))
                    print("SIMLEX: correlation = {0}, p-value = {1} (n_test = {2})".format(cor_simlex,
                                                                                             pval_simlex, n_simlex))
                    print("WS353: correlation = {0}, p-value = {1} (n_test = {2})".format(cor_ws353,
                                                                                            pval_ws353, n_ws353))
                    print("MSIMLEX: correlation = {0}, p-value = {1} (n_test = {2})".format(cor_msimlex,
                                                                                              pval_msimlex, n_msimlex))
                    print("ALL: correlation = {0}, p-value = {1} (n_test = {2})".format(cor_all,
                                                                                            pval_all, n_all))
                    print("MEAN_COSINE = {0} (n = {1}, std = {2})".format(mean_cosine, n_cosine, sd_cosine))

                    # Write to file
                    with open(output_file, "a") as file:
                        file.write(";".join(map(str, [win_size, vec_size, neg_sam, ns_exp, samp_size,
                                                      n_mc, n_rg, n_simlex, n_ws353, n_msimlex, n_all, n_cosine,
                                                      pval_mc, pval_rg, pval_simlex, pval_ws353, pval_msimlex, pval_all,
                                                      sd_cosine,
                                                      cor_mc, cor_rg, cor_simlex, cor_ws353, cor_msimlex, cor_all,
                                                      mean_cosine])))
                        file.write("\n")
