import os
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pandas as pd
import json
import itertools


class DynamicWordEmbedding:

    def __init__(self):
        self.embedding_list = []
        self.embedding_name_list = []
        self.ref_name = None
        self.vocab_list = []
        self.common_vocab = []
        self.freq_list = []
        self.metadata = {}

    def build_with_aligned_w2v_ref(self, corpora_folder, corpus_list=None, embedding_name_list=None, ref_name=None,
                                   min_count=100, window=5, size=300, sample=1e-5, negative=10, alpha=0.025,
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
            self.metadata = {"type": "aligned_w2v_ref", "date": str(datetime.now()), "input_folder": corpora_folder,
                             "corpus_list": corpus_list, "ref_name": self.ref_name, "min_count": min_count,
                             "window": window, "size": size, "sample": sample, "negative": negative, "alpha": alpha,
                             "ns_exponent": ns_exponent, "sg": sg}

    def build_with_aligned_w2v(self, corpora_folder, corpus_list=None, embedding_name_list=None,
                               min_count=100, window=5, size=300, sample=1e-5, negative=10, alpha=0.025,
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

        previous_model = None
        # Make all embeddings
        for i, corpus in tqdm(enumerate(corpus_list)):
            if i == 0:
                previous_model = Word2Vec(corpus_file=f"{corpora_folder}/{corpus}",
                                          min_count=min_count,
                                          window=window,
                                          size=size,
                                          sample=sample,
                                          negative=negative,
                                          alpha=alpha,
                                          ns_exponent=ns_exponent,
                                          workers=workers,
                                          sg=sg)
                self.common_vocab = list(previous_model.wv.vocab.keys())
            else:
                current_model = Word2Vec(corpus_file=f"{corpora_folder}/{corpus}",
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
                corpus_vocab = list(previous_model.wv.vocab.keys())
                current_vocab = list(current_model.wv.vocab.keys())
                corpus_vectors = previous_model.wv.__getitem__(corpus_vocab)
                corpus_freq = [previous_model.wv.vocab.get(word).count for word in corpus_vocab]

                # Align the year on ref
                corpus_common_vocab = list(set(current_vocab) & set(corpus_vocab))
                restrained_current_vectors = current_model.wv.__getitem__(corpus_common_vocab)
                restrained_previous_vectors = previous_model.wv.__getitem__(corpus_common_vocab)
                mat, _ = orthogonal_procrustes(restrained_previous_vectors, restrained_current_vectors)

                # Save elements
                self.embedding_list.append(corpus_vectors.dot(mat))
                self.vocab_list.append(corpus_vocab)
                self.freq_list.append(corpus_freq)
                self.common_vocab = list(set(self.common_vocab) & set(corpus_vocab))

                # Update previous model
                previous_model = current_model

        # Last update
        corpus_vocab = list(previous_model.wv.vocab.keys())
        corpus_vectors = previous_model.wv.__getitem__(corpus_vocab)
        corpus_freq = [previous_model.wv.vocab.get(word).count for word in corpus_vocab]
        self.embedding_list.append(corpus_vectors)
        self.vocab_list.append(corpus_vocab)
        self.freq_list.append(corpus_freq)
        self.common_vocab = list(set(self.common_vocab) & set(corpus_vocab))

        # Create metadata
        self.metadata = {"type": "aligned_w2v", "date": str(datetime.now()), "input_folder": corpora_folder,
                         "corpus_list": corpus_list, "ref_name": self.ref_name, "min_count": min_count,
                         "window": window, "size": size, "sample": sample, "negative": negative, "alpha": alpha,
                         "ns_exponent": ns_exponent, "sg": sg}

    def build_with_svd_ppmi(self, corpora_folder, corpus_list=None, embedding_name_list=None, min_count=100, window=10):
        to_do = True

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

    def word_freq_series(self, word_list, relative_freq=False):

        # Create df for output
        output_df = pd.DataFrame(columns=self.embedding_name_list)
        # If word_list is not a list, change it
        if word_list.__class__.__name__ != "list":
            word_list = [word_list]
        # Loop on word list
        for i, word in enumerate(word_list):
            frequency_dict = {}
            # Loop on embeddings
            for j, vocab in enumerate(self.vocab_list):
                if word in vocab:
                    word_frequency = self.freq_list[j][vocab.index(word)]
                    if not relative_freq:
                        frequency_dict[self.embedding_name_list[j]] = word_frequency
                    else:
                        frequency_dict[self.embedding_name_list[j]] = word_frequency / sum(self.freq_list[j])
                else:
                    frequency_dict[self.embedding_name_list[j]] = 0
            # Add to df
            row_to_add = pd.Series(frequency_dict, name=word)
            output_df = output_df.append(row_to_add)

        # Return df
        return output_df

    def word_vector_series(self, word_list):

        # If word_list is not a list, change it
        if word_list.__class__.__name__ != "list":
            word_list = [word_list]

        # The output dictonary
        output_dict = {}
        # Loop on embedding_name
        for i, embedding_name in enumerate(self.embedding_name_list):
            # Get the dimension of the embedding
            _, embedding_dim = self.embedding_list[i].shape
            # Create a NaN DataFrame
            embedding_df = pd.DataFrame(np.full([len(word_list), embedding_dim], np.nan), index=word_list)
            # Loop on word in the list
            for word in word_list:
                # If it exists, change the value
                if word in self.vocab_list[i]:
                    word_id = self.vocab_list[i].index(word)
                    embedding_df.loc[word, :] = self.embedding_list[i][word_id, :]
            # Add the the output dictionary
            output_dict[embedding_name] = embedding_df

        return output_dict

    def cosine_sim_series(self, word_list):

        # If word_list is not a list, change it
        if word_list.__class__.__name__ != "list":
            word_list = [word_list]

        # If 1 word return empty list, if 2 words return list, else return dict
        if len(word_list) == 1:
            return []
        else:
            cosine_list = []
            cosine_dict = {}
            # Loop on embeddings
            for i, embedding in enumerate(self.embedding_list):
                # Make a empty sim matrix
                sim_matrix = np.full([len(word_list), len(word_list)], np.nan)
                # Indices of word_list in vocab_list
                index_present = [self.vocab_list[i].index(word) for word in word_list if word in self.vocab_list[i]]
                # If none, return empty list
                if len(index_present) == 0:
                    cosine_list.append(np.nan)
                    cosine_dict[self.embedding_name_list[i]] = \
                        pd.DataFrame(sim_matrix, columns=word_list, index=word_list)
                    continue
                # Indices of vocab_list in word_list
                index_in_sim_present = np.array([j for j, word in enumerate(word_list) if word in self.vocab_list[i]])
                # Compute the cosine sim matrix of present words
                cosine_sim_present = cosine_similarity(embedding[index_present])
                # Fill the value for present words
                sim_matrix[index_in_sim_present[:, np.newaxis], index_in_sim_present] = cosine_sim_present
                # If only 2 words, keep only 1 value, otherwise keep the whole matrix
                if len(word_list) == 2:
                    cosine_list.append(sim_matrix[0, 1])
                else:
                    cosine_df = pd.DataFrame(sim_matrix, columns=word_list, index=word_list)
                    cosine_dict[self.embedding_name_list[i]] = cosine_df

            # Return the dict or the list
            if len(word_list) == 2:
                return cosine_list
            else:
                return cosine_dict

    def cosine_autosim_series(self, word_list, start=0, step=1):

        # Get the word vectors from word list
        word_vectors_dict = self.word_vector_series(word_list)
        # The length of series
        dict_length = len(word_vectors_dict)

        # Forward or backward pass
        if step > 0:
            index_list = list(range(start, dict_length, int(np.ceil(step))))
        elif step < 0:
            index_list = list(range(start, -1, int(np.floor(step))))
        else:
            UserWarning("The value of 'step' can't be 0, setting it to 1")
            index_list = list(range(start, dict_length, 1))

        # Loop on the series
        output_list = []
        column_name_list = []
        old_df = []
        old_name = ""
        for i, ind in enumerate(index_list):
            # If first iteration, create old values
            if i == 0:
                old_name = self.embedding_name_list[ind]
                old_df = word_vectors_dict[old_name]
            else:
                # Get new values
                new_name = self.embedding_name_list[ind]
                new_df = word_vectors_dict[new_name]
                auto_cosine_list = []
                # Loop on words
                for word in word_list:
                    old_vec = old_df.loc[word]
                    new_vec = new_df.loc[word]
                    # Get cosine if vector exists
                    if not (old_vec.isnull().values.any() or new_vec.isnull().values.any()):
                        auto_cosine_list.append(cosine_similarity(np.array([old_vec, new_vec]))[0, 1])
                    else:
                        auto_cosine_list.append(np.nan)
                # Store in list
                output_list.append(auto_cosine_list)
                # Update name list
                column_name_list.append(f"{old_name} -> {new_name}")
                # Update old value
                old_name = new_name
                old_df = new_df

        return pd.DataFrame(np.array(output_list).T, index=word_list, columns=column_name_list)

    def neighbors_series(self, word_list, n_top=10, mult_factors=None):

        # If word_list is not a list, change it
        if word_list.__class__.__name__ != "list":
            word_list = [word_list]

        # If mult_factors is None, set it to the mean
        if mult_factors is None:
            mult_factors = [[1 / len(word_list)] * len(word_list)] * len(self.embedding_name_list)
        # If mult_factors is not a list of list, duplicate the list
        elif mult_factors[0].__class__.__name__ != "list":
            mult_factors = [mult_factors] * len(self.embedding_name_list)

        topn_word_dict_list = []
        # Loop on embeddings
        for i, embedding in enumerate(self.embedding_list):

            # Indices of word_list in vocab_list
            index_present = [self.vocab_list[i].index(word) for word in word_list if word in self.vocab_list[i]]
            # If not all words are present, output None and Nan
            if len(index_present) < len(word_list):
                topn_word_dict_list.append({})
            if len(index_present) == len(word_list):
                # The mean vector
                word_vector = embedding[index_present].T.dot(mult_factors[i])
                # The norm of the vector
                word_norm = np.sqrt(np.sum(word_vector ** 2))
                # The list of all norms
                word_norm_list = np.sqrt(np.sum(embedding ** 2, axis=1))
                # Compute the cosine
                cosine_list = embedding.dot(word_vector) / (word_norm_list * word_norm)
                # Get the top n indices
                top_id_list = np.flip(np.argsort(cosine_list)[-n_top:])
                # Get the top cosine similarities list
                top_cosine_list = cosine_list[top_id_list]
                # Ge the top neighboring words list
                top_neighbor_list = np.array(self.vocab_list[i])[top_id_list]
                topn_word_dict_list.append(dict(zip(top_neighbor_list, top_cosine_list)))

        # Get common_voc
        common_voc = set(itertools.chain.from_iterable([cos_dict.keys() for cos_dict in topn_word_dict_list]))
        # Create df
        output_df = pd.DataFrame(index=common_voc)
        for i, embed_name in enumerate(self.embedding_name_list):
            output_df = pd.concat([output_df, pd.Series(topn_word_dict_list[i], name=embed_name)], axis=1)
        # Return df
        return output_df


def load(input_folder):
    output_dyn_emb = DynamicWordEmbedding()
    output_dyn_emb.load(input_folder)
    return output_dyn_emb


def build_with_aligned_w2v(corpora_folder, corpus_list=None, embedding_name_list=None, min_count=100,
                           window=10, size=300, sample=1e-5, negative=10, alpha=0.025, ns_exponent=0.75, workers=6,
                           sg=1):
    output_dyn_emb = DynamicWordEmbedding()
    output_dyn_emb.build_with_aligned_w2v(corpora_folder, corpus_list=corpus_list,
                                          embedding_name_list=embedding_name_list,
                                          min_count=min_count, window=window, size=size, sample=sample,
                                          negative=negative, alpha=alpha, ns_exponent=ns_exponent, workers=workers,
                                          sg=sg)
    return output_dyn_emb
