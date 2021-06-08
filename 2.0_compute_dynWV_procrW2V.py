import multiprocessing as mp
from dynamicwe import DynamicWordEmbedding

# -------------------------------------
# --- Computations
# -------------------------------------

# Init model
dyn_we = DynamicWordEmbedding()
# Build model
dyn_we.build_with_aligned_w2v("/home/gguex/Documents/data/impresso/by_year",
                              ref_name="1997.txt",
                              min_count=100,
                              window=5,
                              size=300,
                              sample=1e-5,
                              negative=5,
                              alpha=0.025,
                              ns_exponent=0.75,
                              workers=mp.cpu_count(),
                              sg=1)
# Save model
dyn_we.save("/home/gguex/Documents/data/impresso/embeddings/w2v/wv_test_2", overwrite=True)
