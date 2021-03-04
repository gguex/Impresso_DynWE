import os
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
from datetime import datetime
import pandas as pd
import json


class DynamicWordEmbedding:

    def __init__(self):
        self.embedding_list = []
        self.embedding_name_list = []
        self.ref_name = None
        self.vocab_list = []
        self.common_vocab = []
        self.freq_list = []
        self.metadata = {}

    def build_with_aligned_w2v(self, corpora_folder, corpus_list=None, embedding_name_list=None, ref_name=None,
                               min_count=100, window=10, size=300, sample=1e-5, negative=10, alpha=0.025,
                               ns_exponent=0.75, workers=6, sg=1):

        # Folder and corpus list error management
        if not os.path.isdir(corpora_folder):
            raise FileNotFoundError(f"No such directory: '{corpora_folder}'")
        file_list = os.listdir(corpora_folder)
        file_list.sort()
        if len(file_list) == 0:
            raise FileNotFoundError(f"'{corpora_folder}' is empty")
        if corpus_list is None:
            corpus_list = file_list
        else:
            if corpus_list.__class__.__name__ != "list":
                corpus_list = [corpus_list]
            missing_corpus_list = [corpus for corpus in corpus_list if corpus not in file_list]
            if len(missing_corpus_list) > 0:
                raise FileNotFoundError(f"{missing_corpus_list} corpora are missing in {corpora_folder}")

        # Embedding name list error management
        if embedding_name_list is None:
            embedding_name_list = corpus_list
        else:
            if embedding_name_list.__class__.__name__ != "list":
                embedding_name_list = [embedding_name_list]
            if len(embedding_name_list) != len(corpus_list):
                raise ValueError("'embedding_name_list' must be same size as 'corpus_list'")
        self.embedding_name_list = embedding_name_list

        # Reference name error management
        if ref_name is None:
            ref_name = embedding_name_list[-1]
        elif ref_name not in embedding_name_list:
            raise ValueError("'ref_name' must be present in 'embedding_name_list'")
        self.ref_name = ref_name

        # Get reference corpus index
        ref_index = embedding_name_list.index(ref_name)

        # Make the reference embedding
        ref_model = Word2Vec(corpus_file=f"{corpora_folder}/{corpus_list[ref_index]}",
                             min_count=min_count,
                             window=window,
                             size=size,
                             sample=sample,
                             negative=negative,
                             alpha=alpha,
                             ns_exponent=ns_exponent,
                             workers=workers,
                             sg=sg)

        # Get ref vocab and vectors
        ref_vocab = list(ref_model.wv.vocab.keys())
        ref_vectors = ref_model.wv.__getitem__(ref_vocab)
        ref_freq = [ref_model.wv.vocab.get(word).count for word in ref_vocab]
        self.common_vocab = ref_vocab

        # Loop on corpus_list
        for i, corpus in tqdm(enumerate(corpus_list)):

            # Save reference if "i" is ref_index
            if i == ref_index:
                self.embedding_list.append(ref_vectors)
                self.vocab_list.append(ref_vocab)
                self.freq_list.append(ref_freq)
            else:
                corpus_model = Word2Vec(corpus_file=f"{corpora_folder}/{corpus}",
                                        min_count=min_count,
                                        window=window,
                                        size=size,
                                        sample=sample,
                                        negative=negative,
                                        alpha=alpha,
                                        ns_exponent=ns_exponent,
                                        workers=workers,
                                        sg=sg)

                # Get year vocab and vectors
                corpus_vocab = list(corpus_model.wv.vocab.keys())
                corpus_vectors = corpus_model.wv.__getitem__(corpus_vocab)
                corpus_freq = [corpus_model.wv.vocab.get(word).count for word in corpus_vocab]

                # Align the year on ref
                corpus_common_vocab = list(set(ref_vocab) & set(corpus_vocab))
                restrained_ref_vectors = ref_model.wv.__getitem__(corpus_common_vocab)
                restrained_year_vectors = corpus_model.wv.__getitem__(corpus_common_vocab)
                mat, _ = orthogonal_procrustes(restrained_year_vectors, restrained_ref_vectors)

                # Save elements
                self.embedding_list.append(corpus_vectors.dot(mat))
                self.vocab_list.append(corpus_vocab)
                self.freq_list.append(corpus_freq)
                self.common_vocab = list(set(self.common_vocab) & set(corpus_vocab))

            # Create metadata
            self.metadata = {"type": "aligned_w2v", "date": str(datetime.now()), "input_folder": corpora_folder,
                             "corpus_list": corpus_list, "ref_name": self.ref_name, "min_count": min_count,
                             "window": window, "size": size, "sample": sample, "negative": negative, "alpha": alpha,
                             "ns_exponent": ns_exponent, "sg": sg}

    def save(self, output_folder, overwrite=False):

        # Error management
        if os.path.isdir(output_folder):
            if not overwrite:
                raise OSError(f"{output_folder} already exists (overwrite is False)")
        else:
            os.mkdir(output_folder)

        # Save embeddings
        for i, embedding_name in enumerate(self.embedding_name_list):
            embedding_df = pd.DataFrame(self.embedding_list[i], index=self.vocab_list[i])
            embedding_df = pd.concat([pd.DataFrame(self.freq_list[i], index=self.vocab_list[i]), embedding_df], axis=1)
            embedding_df.to_csv(f"{output_folder}/{embedding_name}.csv", header=False)

        # Save metadata
        with open(f"{output_folder}/metadata.json", "w") as meta_file:
            json.dump(self.metadata, meta_file)

    def load(self, input_folder):

        # Error management
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"No such directory: '{input_folder}'")
        file_list = os.listdir(input_folder)
        file_list.sort()

        # Load metadata
        if "metadata.json" not in file_list:
            raise Warning(f"Metadata file is missing in '{input_folder}'")
        else:
            with open(f"{input_folder}/metadata.json") as meta_file:
                self.metadata = json.load(meta_file)
            file_list.remove("metadata.json")
            if "ref_name" in self.metadata.keys():
                self.ref_name = self.metadata["ref_name"]

        if len(file_list) == 0:
            raise FileNotFoundError(f"'{input_folder}' is empty")

        # Load embeddings data
        self.embedding_name_list = []
        self.vocab_list = []
        self.freq_list = []
        self.embedding_list = []
        for i, file in enumerate(file_list):
            embedding_df = pd.read_csv(f"{input_folder}/{file}", header=None)
            # Save name
            self.embedding_name_list.append(file[:-4])
            # Save vocab
            embedding_vocab = list(embedding_df[0])
            self.vocab_list.append(embedding_vocab)
            # Save common vocab
            if i == 0:
                self.common_vocab = embedding_vocab
            else:
                self.common_vocab = list(set(self.common_vocab) & set(embedding_vocab))
            # Save freq_list
            self.freq_list.append(list(embedding_df[1]))
            # Save vectors
            self.embedding_list.append(np.array(embedding_df.drop([0, 1], axis=1)))


