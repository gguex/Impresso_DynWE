import pandas as pd
import re
import nltk
from tqdm import tqdm

# -- Input and output file path names

data_to_test_path = "/home/gguex/Documents/data/impresso/JDG/JDG-1997.jsonl.bz2"
#output_file_path = "/home/gguex/Documents/data/impresso/JDG_text_only_new/JDG-1997.txt"
output_file_path = "/home/gguex/Documents/data/impresso/JDG_text_only_new/JDG-1997_ar.txt"

# -- Constructing tokenizer

sent_tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')

# -- Processing one file

# Opening json
df = pd.read_json(data_to_test_path, lines=True)
# Keeping ads and articles, text only
#processed_df = df[df["tp"].isin(["ad", "ar"])]["ft"].dropna()
# Keeping only articles text
processed_df = df[df["tp"].isin(["ar"])]["ft"].dropna()

# Opening output file
with open(output_file_path, "w") as output_file:
    for article in tqdm(processed_df):
        # Transforming article in a list of sentences
        sent_list = sent_tokenizer.tokenize(article)
        for sentence in sent_list:
            # Keeping only alphas and "-"
            sentence_treated = re.sub(r'[^a-zA-ZÀ-ÿ\- ]', ' ', sentence)
            # Removing "-" which aren't between two words
            sentence_treated = re.sub(r'\- ', ' ', sentence_treated)
            sentence_treated = re.sub(r' \-', ' ', sentence_treated)
            sentence_treated = re.sub(r'\-$', ' ', sentence_treated)
            sentence_treated = re.sub(r'^\-', ' ', sentence_treated)
            # Removing extra spaces and setting to lower
            sentence_treated = ' '.join(sentence_treated.lower().split())
            # If non-empty, write it
            if len(sentence_treated) > 0:
                output_file.write(sentence_treated + "\n")
