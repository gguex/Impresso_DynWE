import os
from datetime import date
import pandas as pd
from gensim.models import Word2Vec
import multiprocessing as mp
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes

# -------------------------------------
# --- Parameters
# -------------------------------------

# Date range
year_range = list(range(1989, 1999))
# Reference year
ref_year = 1997
# Data folder path
input_folder_path = "/home/gguex/Documents/data/impresso/by_year"
# Output folder path
output_folder_path = "/home/gguex/Documents/data/impresso/embeddings/w2v"
# Model folder name
folder_name = "wv_test_1"

# W2V hyper-parameters
window_range = 10
vector_size = 300
negative_sample_size = 10
ns_exponent = 0.75
sample_size = 1e-5

# Number of cpus to use
n_cpu = mp.cpu_count()

# -------------------------------------
# --- Process
# -------------------------------------

# Create a folder for results
result_folder = f"{output_folder_path}/{folder_name}"
try:
    os.mkdir(result_folder)
except OSError as os_error:
    print(f"Directory creation failed: {os_error}")
else:
    print(f"Successfully created the directory {result_folder}")

# Write metadata
with open(f"{result_folder}/metadata.txt", "w") as meta_file:
    meta_file.write(f"W2V orthogonal procrustes model (Histwords)\n"
                    f"{date.today()}\n"
                    f"year_range = {year_range}\n"
                    f"ref_year = {ref_year}\n"
                    f"window_range = {window_range}\n"
                    f"vector_size = {vector_size}\n"
                    f"negative_sample_size = {negative_sample_size}\n"
                    f"ns_exponent = {ns_exponent}\n"
                    f"sample_size = {sample_size}\n")

# Remove year if in list
if ref_year in year_range:
    year_range.remove(ref_year)

# Make the reference embedding
ref_model = Word2Vec(corpus_file=f"{input_folder_path}/{ref_year}.txt",
                     min_count=100,
                     window=window_range,
                     size=vector_size,
                     sample=sample_size,
                     negative=negative_sample_size,
                     alpha=0.025,
                     ns_exponent=ns_exponent,
                     workers=n_cpu,
                     sg=1)

# Get ref vocab and vectors
ref_vocab = list(ref_model.wv.vocab.keys())
ref_vectors = ref_model.wv.__getitem__(ref_vocab)

# Save ref model
ref_df = pd.DataFrame(ref_vectors, index=ref_vocab)
ref_df.to_csv(f"{result_folder}/{ref_year}_vectors.csv", header=False)

# Loop on years
for year in tqdm(year_range):
    # Make year model
    year_model = Word2Vec(corpus_file=f"{input_folder_path}/{year}.txt",
                          min_count=100,
                          window=window_range,
                          size=vector_size,
                          sample=sample_size,
                          negative=negative_sample_size,
                          alpha=0.025,
                          ns_exponent=ns_exponent,
                          workers=n_cpu,
                          sg=1)

    # Get year vocab and vectors
    year_vocab = list(year_model.wv.vocab.keys())
    year_vectors = year_model.wv.__getitem__(year_vocab)

    # Align the year on ref
    common_vocab = list(set(ref_vocab) & set(year_vocab))
    restrained_ref_vectors = ref_model.wv.__getitem__(common_vocab)
    restrained_year_vectors = year_model.wv.__getitem__(common_vocab)
    M, _ = orthogonal_procrustes(restrained_year_vectors, restrained_ref_vectors)
    aligned_year_vectors = year_vectors.dot(M)

    # Save year model
    year_df = pd.DataFrame(aligned_year_vectors, index=year_vocab)
    year_df.to_csv(f"{result_folder}/{year}_vectors.csv", header=False)
