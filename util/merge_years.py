from nltk import word_tokenize, FreqDist, sent_tokenize
from tqdm import tqdm

# -------------------------------------
# --- Parameters
# -------------------------------------

# Year bins
year_bin_list = [list(range(1950, 1955)),
                 list(range(1955, 1960)),
                 list(range(1960, 1965)),
                 list(range(1965, 1970)),
                 list(range(1970, 1975)),
                 list(range(1975, 1980)),
                 list(range(1980, 1985)),
                 list(range(1985, 1990)),
                 list(range(1990, 1995)),
                 list(range(1995, 1999))]
year_bin_list = [[1989], [1990], [1991], [1992], [1993], [1994], [1995], [1996], [1997], [1998]]
# folder JDG
JDG_folder = "/home/gguex/Documents/data/impresso/JDG_text_only/"
# folder GDL
GDL_folder = "/home/gguex/Documents/data/impresso/GDL_text_only/"
# Output folder
output_folder = "/home/gguex/Documents/data/impresso/by_year_new/"
# Threshold for words
word_threshold = 500

# -------------------------------------
# --- Computation
# -------------------------------------

if word_threshold is None:
    for year_bin in tqdm(year_bin_list):
        out_date = year_bin[0]
        with open(f"{output_folder}{out_date}.txt", 'w') as outfile:
            for in_date in year_bin:
                with open(f"{JDG_folder}JDG-{in_date}.txt") as infile:
                    for line in infile:
                        outfile.write(line)
                with open(f"{GDL_folder}GDL-{in_date}.txt") as infile:
                    for line in infile:
                        outfile.write(line)
else:
    for year_bin in tqdm(year_bin_list):
        out_date = year_bin[0]
        with open(f"{output_folder}{out_date}.txt", 'w') as outfile:
            for in_date in year_bin:
                with open(f"{JDG_folder}JDG-{in_date}.txt") as JDG_infile, \
                        open(f"{GDL_folder}GDL-{in_date}.txt") as GDL_infile:
                    JDG_text = JDG_infile.read()
                    GDL_text = GDL_infile.read()
                    tokens = word_tokenize(JDG_text + " " + GDL_text)
                    freq_dic = FreqDist(tokens)
                    for line in JDG_text.split("\n"):
                        line_tokens = line.split()
                        kept_tokens = [token for token in line_tokens if freq_dic[token] >= word_threshold]
                        if len(kept_tokens) > 0:
                            outfile.write(" ".join(kept_tokens))
                            outfile.write("\n")
                    for line in GDL_text.split("\n"):
                        line_tokens = line.split()
                        kept_tokens = [token for token in line_tokens if freq_dic[token] >= word_threshold]
                        if len(kept_tokens) > 0:
                            outfile.write(" ".join(kept_tokens))
                            outfile.write("\n")