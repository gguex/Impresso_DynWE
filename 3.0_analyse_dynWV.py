import dynamicwe as dwe

# -------------------------------------
# --- Parameters
# -------------------------------------

dyn_we = dwe.load("/home/gguex/Documents/data/impresso/embeddings/w2v/wv_test_2")

dyn_we.word_freq_series(["amour", "haine", "chocolat"])

dyn_we.word_vector_series(["fluor"])

dyn_we.cosine_sim_series(["chocolat", "rêve", "lgksn"])

all_changes = dyn_we.cosine_autosim_series(dyn_we.common_vocab, step=1)

neighbour = dyn_we.neighbors_series("trésor", n_top=20)