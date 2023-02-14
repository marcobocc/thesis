import random
import string


class SearchEngineConfig:
    DOT_PRODUCT = 'DOT_PRODUCT'
    COSINE = 'COSINE'
    EUCLIDEAN = 'EUCLIDEAN'

    def __init__(self,
                 use_stemming,
                 remove_stopwords,
                 expand_synonyms,
                 cutoff,
                 similarity,
                 config_name=""):
        self.config_name = config_name if len(config_name) else ''.join(random.choice(string.ascii_lowercase) for _ in range(32))
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.expand_synonyms = expand_synonyms
        self.cutoff = cutoff
        self.similarity = similarity

    def to_dict(self):
        return vars(self)
