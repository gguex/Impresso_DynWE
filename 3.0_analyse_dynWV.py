import dynamicwe as dwe

# -------------------------------------
# --- Parameters
# -------------------------------------

dyn_we = dwe.load("/home/gguex/Documents/data/impresso/embeddings/w2v/wv_5year_ref")

common_word_freq = dyn_we.word_freq_series(dyn_we.common_vocab)
common_word_freq["sum"] = common_word_freq.sum(axis=1)
top1000_word = list(common_word_freq.nlargest(1000, "sum").index)


#dyn_we.word_vector_series(["fluor"])

#dyn_we.cosine_sim_series(["chocolat", "rêve", "lgksn"])

#dyn_we.plot_word_neighbors("famille", 1000)

all_changes = 1 - dyn_we.cosine_autosim_series(top1000_word)
p_value_df = dwe.meanshift_pvalue_series(all_changes, 1000)

all_changes["sum"] = all_changes.sum(axis=1)

neighbour = dyn_we.neighbors_series("famille", n_top=50)