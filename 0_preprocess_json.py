import pandas as pd
import re
import nltk
from tqdm import tqdm
import multiprocessing as mp

# -------------------------------------
# --- Parameters
# -------------------------------------

# Date range
year_range = list(range(1989, 1999))
# Input folder path
input_folder_path = "/home/gguex/Documents/data/impresso/GDL"
# Output folder path
output_folder_path = "/home/gguex/Documents/data/impresso/GDL_text_only"
# List of files to process
input_file_list = [f"GDL-{year}.jsonl.bz2" for year in year_range]
# Keep advertisement parameter
keep_ad = False

# Number of cpus to use
n_cpu = mp.cpu_count()

# -------------------------------------
# --- Process
# -------------------------------------

def process_article(article):
    # Tokenizer
    sent_tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')
    # Transforming article in a list of sentences
    sent_list = sent_tokenizer.tokenize(article)
    # Outputted string
    string_output = ""
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
        if len(sentence) > 0:
            string_output += f"{sentence_treated}\n"

    return string_output

for input_file in tqdm(input_file_list):

    # Opening json
    df = pd.read_json(f"{input_folder_path}/{input_file}", lines=True)

    # Keeping ads or not
    if keep_ad:
        processed_df = df[df["tp"].isin(["ad", "ar"])]["ft"].dropna()
    else:
        processed_df = df[df["tp"].isin(["ar"])]["ft"].dropna()

    # Compute article strings
    with mp.Pool(processes=n_cpu) as pool:
        process_article_list = pool.map(process_article, processed_df)

    # Write text
    with open(f"{output_folder_path}/{input_file[:-10]}.txt", "w") as output_file:
        for article_string in process_article_list:
            if len(article_string) > 0:
                output_file.write(article_string + "\n")
