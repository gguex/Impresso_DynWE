# -------------------------------------
# --- Parameters
# -------------------------------------

# Date range
date_range = list(range(1989, 1999))
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
# folder JDG
JDG_folder = "/home/gguex/Documents/data/impresso/JDG_text_only/"
# folder GDL
GDL_folder = "/home/gguex/Documents/data/impresso/GDL_text_only/"
# Output folder
output_folder = "/home/gguex/Documents/data/impresso/by_5year/"

# -------------------------------------
# --- Computation
# -------------------------------------

for year_bin in year_bin_list:
    out_date = year_bin[0]
    with open(f"{output_folder}{out_date}.txt", 'w') as outfile:
        for in_date in year_bin:
            with open(f"{JDG_folder}JDG-{in_date}.txt") as infile:
                for line in infile:
                    outfile.write(line)
            with open(f"{GDL_folder}GDL-{in_date}.txt") as infile:
                for line in infile:
                    outfile.write(line)
