import dynamicwe as dwe
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# -------------------------------------
# --- Parameters
# -------------------------------------

dyn_we = dwe.load("/home/gguex/Documents/data/impresso/embeddings/w2v/wv_test_1")

# Words analogy datasets
eval_data_list = ["fr_mc.csv", "fr_rg.csv", "fr_simlex.csv", "fr_ws353.csv", "multi_simlex.csv", "fr_all.csv"]

# Build dataset dict
eval_data_dict = {}
for eval_data in eval_data_list:
    eval_data_dict[eval_data] = pd.read_csv(f"eval_datasets/{eval_data}")

# Loop on embeddings
result_array = []
for i, embedding in enumerate(dyn_we.embedding_list):
    # Loop on eval datasets
    result_row = []
    for key in eval_data_dict:
        # Get the eval df
        eval_df = eval_data_dict[key]
        # Restrain the eval df to the word which appear in embedding
        presence_df = eval_df[eval_df["word1"].isin(dyn_we.vocab_list[i]) & eval_df["word2"].isin(dyn_we.vocab_list[i])]
        # Compute the similarity in embedding
        sim_list = [cosine_similarity([embedding[dyn_we.vocab_list[i].index(row["word1"])],
                                       embedding[dyn_we.vocab_list[i].index(row["word2"])]])[0, 1]
                    for _, row in presence_df.iterrows()]
        # Compute the correlation
        cor_spear, _ = stats.spearmanr(presence_df["score"], np.array(sim_list))
        # Store cor results
        result_row.append(cor_spear)
        # Store the number of lines tested
        result_row.append(presence_df.shape[0])
    result_array.append(result_row)

result_df = pd.DataFrame(result_array, index=dyn_we.embedding_name_list,
                         columns=np.repeat(list(eval_data_dict.keys()), 2))