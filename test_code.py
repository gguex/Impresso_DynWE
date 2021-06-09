import os
from nltk import word_tokenize, FreqDist
import numpy as np

corpora_folder = "/home/gguex/Documents/data/impresso/by_year"
min_count = 100
window = 4

file_list = os.listdir(corpora_folder)
file_list.sort()
corpus = file_list[0]

# Load file
with open(f"{corpora_folder}/{corpus}") as corpus_file:
    corpus_text = corpus_file.read()

# Get all tokens
tokens = word_tokenize(corpus_text)
# Get word frequencies
freq_dic = FreqDist(tokens)

# Build voc with enough tokens
remaining_voc = [item[0] for item in freq_dic.items() if item[1] >= min_count]

# Init the count table
pair_list = []

# split the corpus in sentence, loop on them
sent_list = corpus_text.split("\n")
for sent in sent_list:
    print(sent)
    # The list of word with enough count
    word_list = [word for word in sent.split(" ") if word in remaining_voc]
    # Loop on id of the word
    for id_tg_word, target_word in enumerate(word_list):

        # Get the neighbour ids
        id_list = [id_tg_word + win_size for win_size in range(-window, window + 1)
                   if (0 <= id_tg_word + win_size < len(word_list)) and win_size != 0]

        # Get the id of the target word
        invoc_id_target = remaining_voc.index(target_word)
        # Get the id of the context words, update table
        for id in id_list:
            invoc_id_context = remaining_voc.index(word_list[id])
            pair_list.append((invoc_id_target, invoc_id_context))