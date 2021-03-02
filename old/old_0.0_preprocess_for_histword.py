# Test WE synchonique

import csv
import nltk
import os

in_folder_name = "/home/gguex/Documents/data/impresso/JDG_text_only/"
out_folder_name = "/home/gguex/Documents/data/impresso/JDG_line_sent/"

for _, _, files in os.walk(in_folder_name):
    files.sort()
    for file in files:
        # Open file
        with open(in_folder_name + file, "r") as csvfile:
            text_data = list(csv.reader(csvfile, delimiter='μ'))

        # Make a list of sentence
        text_sentence_list = [item for sublist in text_data for item in sublist]

        # Make a list of list of token
        sentence_token_list = [nltk.word_tokenize(sent.strip()) for sent in text_sentence_list]

        # Remove non-alphanum and lower all words
        sentence_token_list = [[w.lower() for w in sent if w.isalpha()] for sent in sentence_token_list]

        # Write the file
        with open(out_folder_name + file[:len(file) - 4] + "_sent.txt", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=" ")
            writer.writerows(sentence_token_list)

in_folder_name = "/home/gguex/Documents/data/impresso/GDL_text_only/"
out_folder_name = "/home/gguex/Documents/data/impresso/GDL_line_sent/"

for _, _, files in os.walk(in_folder_name):
    files.sort()
    for file in files:
        # Open file
        with open(in_folder_name + file, "r") as csvfile:
            text_data = list(csv.reader(csvfile, delimiter='μ'))

        # Make a list of sentence
        text_sentence_list = [item for sublist in text_data for item in sublist]

        # Make a list of list of token
        sentence_token_list = [nltk.word_tokenize(sent.strip()) for sent in text_sentence_list]

        # Remove non-alphanum and lower all words
        sentence_token_list = [[w.lower() for w in sent if w.isalpha()] for sent in sentence_token_list]

        # Write the file
        with open(out_folder_name + file[:len(file) - 4] + "_sent.txt", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=" ")
            writer.writerows(sentence_token_list)
