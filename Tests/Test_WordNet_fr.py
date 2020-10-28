from nltk.corpus import wordnet as wn

dog_synset = wn.synset("dog.n.01")

dog_lemmas = dog_synset.lemmas()
dog_lemmas[1].usage_domains()

get_related_synsets(dog_synset)

related_lemmas(dog_lemmas[1])

# gets all related synsets to 'word'
set(reduce(lambda x, y: x + get_related_synsets(y), wn.synsets("dog"), []))

# FRA

wn.synsets("chien", lang='fra')
chien_lemmas = wn.lemmas("chien", lang="fra")
chien_lemmas[1].usage_domains()

len(wordnet.all_lemma_names(pos='n', lang='fra'))