from keras.preprocessing import text

# Corpus path
corpus_path = "/home/gguex/Documents/data/impresso/mini/mini.txt"

# Opening file
with open(corpus_path) as corpus_file:
    corpus_text = corpus_file.read()
    corpus_sent = corpus_text.split("\n")

# Keras tokenizer
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(corpus_sent)

# Get translators
word2id = tokenizer.word_index
id2word = {v: k for k, v in word2id.items()}

# Save vocab_size and set embedding size
vocab_size = len(word2id) + 1
embed_size = 100

# Transform the corpus into id
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in corpus_sent]

