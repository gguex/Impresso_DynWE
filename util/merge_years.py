# -------------------------------------
# --- Parameters
# -------------------------------------

# Date range
date_range = list(range(1989, 1999))
# folder JDG
JDG_folder = "/home/gguex/Documents/data/impresso/JDG_text_only/"
# folder GDL
GDL_folder = "/home/gguex/Documents/data/impresso/GDL_text_only/"
# Output folder
output_folder = "/home/gguex/Documents/data/impresso/by_year/"

# -------------------------------------
# --- Computation
# -------------------------------------

for date in date_range:
    filenames = ['file1.txt', 'file2.txt', ...]
    with open(f"{output_folder}{date}.txt", 'w') as outfile:
        with open(f"{JDG_folder}JDG-{date}.txt") as infile:
            for line in infile:
                outfile.write(line)
        with open(f"{GDL_folder}GDL-{date}.txt") as infile:
            for line in infile:
                outfile.write(line)
