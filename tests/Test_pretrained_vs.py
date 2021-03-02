from gensim.models import KeyedVectors, Word2Vec
# from gensim.test.utils import datapath
from scipy.linalg import orthogonal_procrustes
import os
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# Corpus file path
file_path = "/home/gguex/Documents/data/impresso/JDG_line_sent/JDG-1997_sent.txt"

# Output file path
output_file = "Results_word_models_compare.txt"

# wv_from_text = KeyedVectors.load_word2vec_format(
#    datapath("/home/gguex/Documents/data/pretrained_word_vectors/frwiki_20180420_300d.txt"), binary=False)
# wv_from_text.save("/home/gguex/Documents/data/pretrained_word_vectors/frwiki.model")

if not os.path.exists(output_file):
    with open(output_file, "w") as file:
        file.write("Création du fichier le {0}\n".format(datetime.datetime.now()))

with open(output_file, "a") as file:
    file.write("Gride search sur le fichier {0} le {1}\n".format(file_path, datetime.datetime.now()))
    file.write("win_size;vec_size;neg_sam;ns_exp;samp_size;mean_cos;std_cos\n")

# Load wiki model
wv_wiki = KeyedVectors.load("/home/gguex/Documents/data/pretrained_word_vectors/frwiki.model")

tested_windows_values = [2, 5, 7, 10, 12, 15]
tested_vectors_size = [300]
tested_negative_sample = [10, 15, 20, 25]
tested_ns_exponent = [0.75]
tested_sample_size = [1e-5]

for vec_size in tested_vectors_size:
    for win_size in tested_windows_values:
        for neg_sam in tested_negative_sample:
            for ns_exp in tested_ns_exponent:
                for samp_size in tested_sample_size:
                    # Print
                    print("Test with {0} win_size, {1} vec_size, {2} neg_sam, {3} ns_exp, {4} samp_size: ".format(
                        win_size, vec_size,
                        neg_sam, ns_exp, samp_size))

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

                    print("Mean cosine = {0}, Std cosine = {1}".format(mean_cosine, sd_cosine))
                    # Write to file
                    with open(output_file, "a") as file:
                        file.write(";".join(map(str, [win_size, vec_size, neg_sam, ns_exp, samp_size,
                                                      mean_cosine, sd_cosine])))
                        file.write("\n")
