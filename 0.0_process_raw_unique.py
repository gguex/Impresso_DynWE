import pandas as pd
import numpy as np
import nltk
import re
from tqdm import tqdm

### Files name

data_to_test_path = "/home/gguex/Documents/data/impresso/JDG/JDG-1997.jsonl.bz2"
output_file_path = "JDG-1997_new.txt"

### Importing lexique dataset

# Defining forbidden words
forbidden_words = ["q", "w", "e", "r", "t", "z", "u", "i", "o", "p", "s", "d", "f", "g", "h", "j", "k", "l", "x", "c",
                   "v", "b", "n", "m"]

# Importing lexique with word frequencies
lexique = pd.read_csv("/home/gguex/Documents/data/lexique_fr/Lexique383.tsv", sep="\t")

# lexical treatments
lexique["freqtot"] = lexique["freqfilms2"] + lexique["freqlivres"]
lexique = lexique[["ortho", "freqtot"]]
lexique.groupby("ortho").sum()
lexique = lexique.drop(lexique[lexique.ortho.isin(forbidden_words)].index)
lexique = lexique.sort_values("freqtot", ascending=False, ignore_index=True)
lexique.reset_index()
lexique["cost"] = lexique["freqtot"] / sum(lexique["freqtot"])
words = list(lexique["ortho"])

wordcost = dict((row["ortho"], np.log((i + 1) * np.log(len(words)))) for i, row in lexique.iterrows())
maxword = max(len(str(x)) for x in words)

#### Infer spaces function for non-splitted words

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i - maxword):i]))
        return min((c + wordcost.get(s[i - k - 1:i], 9e999), k + 1) for k, c in candidates)

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

### Testing

df = pd.read_json(data_to_test_path, lines=True)
processed_df = df[df["tp"].isin(["ad", "ar"])]["ft"].dropna()

with open(output_file_path, "w") as output_file:
    for article in tqdm(processed_df):
        article = re.sub("'", "' ", article)
        article = re.sub(r'[^a-zA-ZÀ-ÿ\-\' ]', ' ', article)
        article = re.sub(r'[!?:]', '.', article)
        token_list = nltk.word_tokenize(article)
        for token in token_list:
            if token == ".":
                output_file.write("\n")
                #print("\n")
            elif token[0:1].isupper() and token[1:2].islower():
                output_file.write(token.lower() + " ")
                #print(token.lower(), end=" ")
            elif token.isalpha():
                token_lower = token.lower()
                if token_lower in words:
                    output_file.write(token_lower + " ")
                    #print(token_lower, end=" ")
                else:
                    token_lower = re.sub("-", "", token_lower)
                    splited_token = infer_spaces(token_lower)
                    splited_token_list = nltk.word_tokenize(splited_token)
                    test_word_ok = [tested_token in words for tested_token in splited_token_list]
                    if np.sum(test_word_ok) == len(test_word_ok):
                        output_file.write(splited_token + " ")
                        #print(splited_token, end=" ")
        output_file.write("\n")