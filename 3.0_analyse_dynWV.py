from dynamicwe import DynamicWordEmbedding

# -------------------------------------
# --- Parameters
# -------------------------------------

dyn_we = DynamicWordEmbedding()
dyn_we.load("/home/gguex/Documents/data/impresso/embeddings/w2v/wv_test_1")

dyn_we.get_words_freq(["amour", "haine", "chocolat"])