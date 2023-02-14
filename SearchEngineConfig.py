class SearchEngineConfig:
    DOT_PRODUCT = 'DOT_PRODUCT'
    COSINE = 'COSINE'
    EUCLIDEAN = 'EUCLIDEAN'

    def __init__(self,
                 config_name,
                 use_stemming,
                 remove_stopwords,
                 expand_synonyms,
                 cutoff,
                 similarity):
        self.config_name = config_name
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.expand_synonyms = expand_synonyms
        self.cutoff = cutoff
        self.similarity = similarity

    def to_dict(self):
        return vars(self)