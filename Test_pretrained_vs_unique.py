from gensim.models import Word2Vec
# from gensim.test.utils import datapath
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity

# Corpus file path
file_path = "/home/gguex/Documents/data/impresso/JDG_line_sent/JDG-1997_sent.txt"

model = Word2Vec(corpus_file=file_path,
                 min_count=100,
                 window=5,
                 size=300,
                 sample=1e-5,
                 negative=15,
                 alpha=0.025,
                 ns_exponent=0.75,
                 workers=8,
                 sg=1)

model2 = Word2Vec(corpus_file=file_path,
                 min_count=100,
                 window=15,
                 size=300,
                 sample=1e-8,
                 negative=5,
                 alpha=0.5,
                 ns_exponent=1,
                 workers=8,
                 sg=1)

# Get the vocab for each model
vocab_model = set(model.wv.vocab.keys())
vocab_model2 = set(model2.wv.vocab.keys())


# Find the common vocabulary
common_vocab = list(vocab_model2 & vocab_model)

# Make the orthogonal_procrustes alignment
model_matrix = model.wv.__getitem__(common_vocab)
model2_matrix = model2.wv.__getitem__(common_vocab)
M, _ = orthogonal_procrustes(model2_matrix, model_matrix)
model2_matrix_aligned = model2.wv.__getitem__(common_vocab).dot(M)

# Compute the cosine
cos_mat = cosine_similarity(model_matrix, model2_matrix_aligned)
vec_cosine = cos_mat.diagonal()
mean_cosine = vec_cosine.mean()
sd_cosine = vec_cosine.std()

print("Mean cosine = {0}, Std cosine = {1}".format(mean_cosine, sd_cosine))
