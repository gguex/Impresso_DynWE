import dynamicwe as dwe

# -------------------------------------
# --- Parameters
# -------------------------------------

dyn_we = dwe.load("/home/gguex/Documents/data/impresso/embeddings/w2v/wv_test_1")

dyn_we.word_freq_series(["amour", "haine", "chocolat"])

dyn_we.word_vector_series(["dgsgs"])

dyn_we.cosine_sim_series(["chocolat", "rêve", "lgksn"])

dyn_we.cosine_autosim_series(["chocolat", "rêve", "lgksn"], start=9, step=-1)
