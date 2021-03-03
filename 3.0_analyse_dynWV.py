import pandas as pd
import numpy as np

# -------------------------------------
# --- Parameters
# -------------------------------------

# Year list
year_list = list(range(1989, 1999))

# Dyn word embeddings path
word_embeddings_dir_path = "/home/gguex/Documents/data/impresso/embeddings/w2v/wv_test_1"

# -------------------------------------
# --- Process
# -------------------------------------

# Loading models
df_list = []
for i, year in enumerate(year_list):
    # Loading df
    we_df = pd.read_csv(f"{word_embeddings_dir_path}/{year}_vectors.csv", header=None)
    # Saving df
    df_list.append(we_df)
    # Creating common vocab
    vocab = list(we_df[0])
    if i == 0:
        common_vocab = set(vocab)
    else:
        common_vocab = common_vocab & set(vocab)
common_vocab = list(common_vocab)



