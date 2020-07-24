import neighboring_words as nw
import pandas as pd
from scipy import stats
import numpy as np

# Words analogy datasets
en_mc = pd.read_csv("/home/gguex/Documents/data/eval_datasets/en_mc.csv")
en_rg = pd.read_csv("/home/gguex/Documents/data/eval_datasets/en_rg.csv")
en_simlex = pd.read_csv("/home/gguex/Documents/data/eval_datasets/en_simlex.csv")
en_ws353 = pd.read_csv("/home/gguex/Documents/data/eval_datasets/en_ws353.csv")
en_msimlex = pd.read_csv("/home/gguex/Documents/data/eval_datasets/multi_simlex_en.csv")
en_all = pd.read_csv("/home/gguex/Documents/data/eval_datasets/en_all.csv")

def new_cosine(word1, word2):
    try :
       cosine_value = nw.COSINE(word1, word2)
    except ZeroDivisionError:
       cosine_value = None
    return(cosine_value)

# MC
sim_mc = [new_cosine(row["word1"], row["word2"]) for _, row in en_mc.iterrows()]
value_ok_mc = [elements != None for elements in sim_mc]
n_mc = sum(value_ok_mc)
cor_mc, pval_mc = stats.spearmanr(np.array(en_mc["score"])[value_ok_mc], np.array(sim_mc)[value_ok_mc])

# RG
sim_rg = [new_cosine(row["word1"], row["word2"]) for _, row in en_rg.iterrows()]
value_ok_rg = [elements != None for elements in sim_rg]
n_rg = sum(value_ok_rg)
cor_rg, pval_rg = stats.spearmanr(en_rg["score"][value_ok_rg], np.array(sim_rg)[value_ok_rg])

# SIMLEX
sim_simlex = [new_cosine(row["word1"], row["word2"]) for _, row in en_simlex.iterrows()]
value_ok_simlex = [elements != None for elements in sim_simlex]
n_simlex = sum(value_ok_simlex)
cor_simlex, pval_simlex = stats.spearmanr(en_simlex["score"][value_ok_simlex], np.array(sim_simlex)[value_ok_simlex])

# WS353
sim_ws353 = [new_cosine(row["word1"], row["word2"]) for _, row in en_ws353.iterrows()]
value_ok_ws353 = [elements != None for elements in sim_ws353]
n_ws353 = sum(value_ok_ws353)
cor_ws353, pval_ws353 = stats.spearmanr(en_ws353["score"][value_ok_ws353], np.array(sim_ws353)[value_ok_ws353])

# MSIMLEX
sim_msimlex = [new_cosine(row["word1"], row["word2"]) for _, row in en_msimlex.iterrows()]
value_ok_msimlex = [elements != None for elements in sim_msimlex]
n_msimlex = sum(value_ok_msimlex)
cor_msimlex, pval_msimlex = stats.spearmanr(
    en_msimlex["score"][value_ok_msimlex], np.array(sim_msimlex)[value_ok_msimlex])

# ALL
sim_all = [new_cosine(row["word1"], row["word2"]) for _, row in en_all.iterrows()]
value_ok_all = [elements != None for elements in sim_all]
n_all = sum(value_ok_all)
cor_all, pval_all = stats.spearmanr(en_all["score"][value_ok_all], np.array(sim_all)[value_ok_all])


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