#### Creation of datasets

# Open the "golden standard" files
fr_mc = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/fr-mc.dataset", sep=";").iloc[:, 1:-1]
fr_rg = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/fr-rg.dataset", sep=";").iloc[:, 1:-1]
fr_simlex = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/fr-simlex.dataset", sep=";").iloc[:, 1:-1]
fr_ws353 = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/fr-ws353.dataset", sep=";").iloc[:, 1:-1]

# Lowercases
fr_mc["word1"] = fr_mc["word1"].str.lower()
fr_mc["word2"] = fr_mc["word2"].str.lower()
fr_rg["word1"] = fr_rg["word1"].str.lower()
fr_rg["word2"] = fr_rg["word2"].str.lower()
fr_simlex["word1"] = fr_simlex["word1"].str.lower()
fr_simlex["word2"] = fr_simlex["word2"].str.lower()
fr_ws353["word1"] = fr_ws353["word1"].str.lower()
fr_ws353["word2"] = fr_ws353["word2"].str.lower()
fr_mc["score"] = fr_mc["score"] * 10 / 4

# Save datasets
fr_mc.to_csv("/home/gguex/Documents/data/eval_datasets/fr_mc.csv", index=False)
fr_rg.to_csv("/home/gguex/Documents/data/eval_datasets/fr_rg.csv", index=False)
fr_simlex.to_csv("/home/gguex/Documents/data/eval_datasets/fr_simlex.csv", index=False)
fr_ws353.to_csv("/home/gguex/Documents/data/eval_datasets/fr_ws353.csv", index=False)

# Multisimlex dataset

multi_simlex = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/FRA.csv").iloc[:, 1:]
multi_simlex["word1"] = multi_simlex["Word 1"]
multi_simlex["word2"] = multi_simlex["Word 2"]
multi_simlex["score"] = multi_simlex.mean(axis=1)
multi_simlex["score"] = multi_simlex["score"] * 10 / 6
multi_simlex[["word1", "word2", "score"]].to_csv("/home/gguex/Documents/data/eval_datasets/multi_simlex.csv",
                                                 index=False)

# Merge all of them in one
fr_all = pd.concat([fr_simlex, fr_ws353, fr_rg, fr_mc, multi_simlex[["word1", "word2", "score"]]])
fr_all = fr_all.drop_duplicates(subset=["word1", "word2"])
fr_all.to_csv("/home/gguex/Documents/data/eval_datasets/fr_all.csv", index=False)


#### Creation of datasets : ENG

# Open the "golden standard" files
en_mc = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/en-mc.dataset", sep=";").iloc[:, 1:-1]
en_rg = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/en-rg.dataset", sep=";").iloc[:, 1:-1]
en_simlex = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/en-simlex.dataset", sep=";").iloc[:, 1:-1]
en_ws353 = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/en-ws353.dataset", sep=";").iloc[:, 1:-1]

# Lowercases
en_mc["word1"] = en_mc["word1"].str.lower()
en_mc["word2"] = en_mc["word2"].str.lower()
en_rg["word1"] = en_rg["word1"].str.lower()
en_rg["word2"] = en_rg["word2"].str.lower()
en_simlex["word1"] = en_simlex["word1"].str.lower()
en_simlex["word2"] = en_simlex["word2"].str.lower()
en_ws353["word1"] = en_ws353["word1"].str.lower()
en_ws353["word2"] = en_ws353["word2"].str.lower()
en_mc["score"] = en_mc["score"] * 10 / 4

# Save datasets
en_mc.to_csv("/home/gguex/Documents/data/eval_datasets/en_mc.csv", index=False)
en_rg.to_csv("/home/gguex/Documents/data/eval_datasets/en_rg.csv", index=False)
en_simlex.to_csv("/home/gguex/Documents/data/eval_datasets/en_simlex.csv", index=False)
en_ws353.to_csv("/home/gguex/Documents/data/eval_datasets/en_ws353.csv", index=False)

# Multisimlex dataset

multi_simlex_en = pd.read_csv("/home/gguex/Documents/data/eval_datasets_raw/ENG.csv").iloc[:, 1:]
multi_simlex_en["word1"] = multi_simlex_en["Word 1"]
multi_simlex_en["word2"] = multi_simlex_en["Word 2"]
multi_simlex_en["score"] = multi_simlex_en.mean(axis=1)
multi_simlex_en["score"] = multi_simlex_en["score"] * 10 / 6
multi_simlex_en[["word1", "word2", "score"]].to_csv("/home/gguex/Documents/data/eval_datasets/multi_simlex_en.csv",
                                                 index=False)

# Merge all of them in one
en_all = pd.concat([en_simlex, en_ws353, en_rg, en_mc, multi_simlex_en[["word1", "word2", "score"]]])
en_all = en_all.drop_duplicates(subset=["word1", "word2"])
en_all.to_csv("/home/gguex/Documents/data/eval_datasets/en_all.csv", index=False)
