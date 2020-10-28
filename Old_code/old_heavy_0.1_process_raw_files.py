import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
from nltk import RegexpTokenizer

# -- Constructing tokenizer

tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')

# -- Defining forbidden words

forbidden_words = ["q", "w", "e", "r", "t", "z", "u", "i", "o", "p", "s", "d", "f", "g", "h", "j", "k", "l", "x", "c",
                   "v", "b", "n", "m"]

# -- Importing lexical dataset and Lexical treatments

lexique = pd.read_csv("/home/gguex/Documents/data/lexique_fr/Lexique383.tsv", sep="\t")

# Adding frequencies
lexique["freqtot"] = lexique["freqfilms2"] + lexique["freqlivres"]
# Keeping only words and frequency columns
lexique = lexique[["ortho", "freqtot"]]
# Aggregating same word type
lexique.groupby("ortho").sum()
# Removing forbidden words
lexique = lexique.drop(lexique[lexique.ortho.isin(forbidden_words)].index)
# Sorting by decreasing frequencies
lexique = lexique.sort_values("freqtot", ascending=False, ignore_index=True)
# Rewriting index
lexique.reset_index()
# Computing "cost"
lexique["cost"] = lexique["freqtot"] / sum(lexique["freqtot"])
# Building dictionary
words = list(lexique["ortho"])

# Building wordcost dict and max length of word
wordcost = dict((row["ortho"], np.log((i + 1) * np.log(len(words)))) for i, row in lexique.iterrows())
maxword = max(len(str(x)) for x in words)


# -- Infer spaces function for non-splitted words

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(it):
        candidates = enumerate(reversed(cost[max(0, it - maxword):it]))
        return min((ci + wordcost.get(s[it - ki - 1:it], 9e999), ki + 1) for ki, ci in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1, len(s) + 1):
        c, k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i > 0:
        c, k = best_match(i)
        assert c == cost[i]
        out.append(s[i - k:i])
        i -= k

    return " ".join(reversed(out))


# -- Processing example

# Names of in_folder and out_folder (JDG)
in_folder_path = "/home/gguex/Documents/data/impresso/JDG/"
out_folder_path = "/home/gguex/Documents/data/impresso/JDG_text_only_new/"

# Running on all files
for _, _, files in os.walk(in_folder_path):
    files.sort()
    for file in files:
        # Opening json
        df = pd.read_json(in_folder_path + file, lines=True)
        # Keeping ads and articles, text only
        processed_df = df[df["tp"].isin(["ad", "ar"])]["ft"].dropna()
        # Opening output file
        with open(out_folder_path + file[:len(file) - 10] + ".txt", "w") as output_file:
            for article in tqdm(processed_df):
                # Transforming all end of sentences into "."
                article = re.sub(r'[!?:]', '..', article)
                # Keeping only alphas, "-", "'", and "."
                article = re.sub(r'[^a-zA-ZÀ-ÿ\-\'. ]', ' ', article)
                # Tokenizing
                token_list = tokenizer.tokenize(article)
                count_written = 0
                for token in token_list:
                    # If "." end of line
                    if token == "." and count_written > 0:
                        output_file.write("\n")
                        count_written = 0
                    # If first letter is upper and next letter is lower, keep it (proper nouns)
                    elif token[0:1].isupper() and token[1:2].islower():
                        output_file.write(token.lower() + " ")
                        count_written += 1
                    # If it's alpha or contains "'" and "-", proceed
                    elif token.isalpha() or ("'" in token) or ("-" in token):
                        token_lower = token.lower()
                        # If it's dictionary, keep it
                        if token_lower in words:
                            output_file.write(token_lower + " ")
                            count_written += 1
                        # Else try to split it
                        else:
                            token_lower = re.sub("-", "", token_lower)
                            splited_token = infer_spaces(token_lower)
                            splited_token_list = tokenizer.tokenize(splited_token)
                            test_word_ok = [tested_token in words for tested_token in splited_token_list]
                            # If all word are in dictionary, keep them
                            if np.sum(test_word_ok) == len(test_word_ok):
                                output_file.write(splited_token + " ")
                                count_written += len(test_word_ok)
                # End of article
                if count_written > 0:
                    output_file.write("\n")
                    count_written = 0
