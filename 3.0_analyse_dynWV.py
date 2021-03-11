import dynamicwe as dwe

# -------------------------------------
# --- Parameters
# -------------------------------------

dyn_we = dwe.load("/home/gguex/Documents/data/impresso/embeddings/w2v/wv_test_1")

dyn_we.word_freq_series(["amour", "haine", "chocolat"])

dyn_we.word_vector_series(["dgsgs"])

dyn_we.cosine_sim_series(["chocolat", "rêve", "lgksn"])

dyn_we.cosine_autosim_series(["chocolat", "rêve", "lgksn"], start=9, step=-1)

neighbou = dyn_we.neighbors_series("chocolat", n_top=20)

i = 0
self = dyn_we
word_list = ["roi", "homme", "femme"]
mult_factors=[1, -1, 1]
top_n = 10

index_present = [self.vocab_list[i].index(word) for word in word_list if word in self.vocab_list[i]]

word_vector = self.embedding_list[i][index_present].T.dot(mult_factors)
word_norm = np.sqrt(np.sum(word_vector ** 2))
word_vector_list = self.embedding_list[i]
word_norm_list = np.sqrt(np.sum(word_vector_list ** 2, axis=1))
cosine_list = self.embedding_list[i].dot(word_vector) / (word_norm_list * word_norm)
top_id = np.flip(np.argsort(cosine_list)[-top_n:])
top_cosine = cosine_list[top_id]
top_word = np.array(self.vocab_list[i])[top_id]