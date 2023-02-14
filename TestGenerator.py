import re
import random

from nltk.corpus import stopwords
from TribunaleDataLoader import TribunaleDataLoader

[
    {
        "query": "quanto costa rinunciare eredita",
        "documents": [
            "Revoca della rinuncia all’eredita'",
            "Rinuncia all’eredita'",
            "FISSAZIONE DEI TERMINI PER ACCETTARE L’EREDITA",
            "ACCETTAZIONE DELL’EREDITA CON BENEFICIO DI INVENTARIO",
            "Esecutori testamentari: rinuncia",
            "Esecutori testamentari: accettazione",
            "Apposizione dei sigilli all’eredita'",
            "RIMOZIONE DEI SIGILLI ALL’EREDITA'",
            "AUTORIZZAZIONE A VENDERE BENI DI EREDITA' ACCETTATE CON BENEFICIO",
            "Inventario dell’eredita'",
            "Eredita giacente",
            "Atti di straordinaria amministrazione a favore di minori e/o genitori"
        ]
    }
]


class TestGenerator:
    def __init__(self):
        self.tribunaleDataLoader = TribunaleDataLoader()
        self.italian_stopwords = stopwords.words("italian")

    def generate(self, num_samples, num_documents_to_select, num_keywords_per_document):
        test_samples = [
            self._generate_sample(num_documents_to_select, num_keywords_per_document)
            for _ in range(num_samples)
        ]
        return test_samples

    def _generate_sample(self, num_documents_to_select, num_keywords_per_document):
        documents = random.sample(self.tribunaleDataLoader.documents, num_documents_to_select)
        num_words_to_select_per_document = num_keywords_per_document
        keywords = self._pick_random_keywords_from_each_document(documents, num_words_to_select_per_document)
        query = " ".join(keywords)
        documentTitles = [document["identifier"].upper() for document in documents]
        test_sample = {
            "query": query,
            "documents": documentTitles
        }
        return test_sample

    def _pick_random_keywords_from_each_document(self, documents, num_words_to_select_per_document):
        keywords = [self._pick_random_keyword_from_single_document(document, num_words_to_select_per_document) for document in documents]
        # Assumption: keep the keywords that belong to the same document close together
        return keywords

    def _pick_random_keyword_from_single_document(self, document, num_words_to_select):
        words = list(set(re.findall(r'\w+', document["contents"]) + re.findall(r'\w+', document["identifier"])))
        filtered = [w.lower() for w in words if len(w) > 2]  # Do not sample words with 2 characters
        # Assumption: the order of the keywords does not matter in a single document
        selected_words = []
        while True:  # Do not build a query composed only of stopwords
            selected_words = random.sample(filtered, num_words_to_select)
            query = " ".join(selected_words)
            num_stopwords = len([w for w in selected_words if w in self.italian_stopwords])
            if num_stopwords < len(selected_words):
                break
        return query
