from cmath import exp
import json
import re
import math
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

class NewSearchEngine:
    def __init__(self, documents):
        self.italian_stopwords = stopwords.words("italian")
        self.stemmer = nltk.stem.snowball.ItalianStemmer()
        self.documents = documents
        self.index = self._build_index(documents)

    def search(self, query, include_synonyms=False, include_stopwords=False):
        sorted_search_results = self._search_with_score(query, include_synonyms, include_stopwords)
        return sorted_search_results

    def _build_index(self, documents):
        index = {}
        for document in documents:
            document_words = [word.lower() for word in re.findall(r'\w+', document["contents"]) + re.findall(r'\w+', document["identifier"])]
            for word in document_words:
                stemmed_word = self.stemmer.stem(word)
                if stemmed_word not in index:
                    index[stemmed_word] = {}
                documentName = document["identifier"]
                if documentName not in index[stemmed_word]:
                    index[stemmed_word][documentName] = {
                        "count" : 0,
                        "tf" : 0
                    }
                index[stemmed_word][documentName]["count"] += 1
                index[stemmed_word][documentName]["tf"] = index[stemmed_word][documentName]["count"] / len(document_words)
        return index

    def _get_idf(self, index, term):
        stemmed_term = self.stemmer.stem(term)
        df = len(index[stemmed_term] if stemmed_term in index else 0.0)
        num_documents = len(self.documents)
        idf = math.log((num_documents + 1) / (float(df) + 1.0)) # Smooth TF-IDF to prevent division by 0
        return idf

    def _get_tf(self, index, term, documentIdentifier):
        stemmed_term = self.stemmer.stem(term)
        if stemmed_term not in index:
            return 0
        if documentIdentifier not in index[stemmed_term]:
            return 0
        return index[stemmed_term][documentIdentifier]["tf"]

    def _get_tfidf(self, index, term, documentIdentifier):
        return self._get_tf(index, term, documentIdentifier) * self._get_idf(index, term)

    def _find_documents(self, keywords, index):
        search_results = {}
        for word in keywords:
                stemmed_word = self.stemmer.stem(word)
                if stemmed_word in index:
                    for documentIdentifier in index[stemmed_word]:
                        if documentIdentifier not in search_results:
                            search_results[documentIdentifier] = []
                        search_results[documentIdentifier].append(word)
        return search_results

    def _rank_documents(self, search_results, index):
        ranked_search_results = {}
        for documentIdentifier in search_results:
            ranked_search_results[documentIdentifier] = []
            for keyword in search_results[documentIdentifier]:
                tfidf = self._get_tfidf(index, keyword, documentIdentifier)
                ranked_search_results[documentIdentifier].append({
                    "keyword": keyword,
                    "tfidf" : tfidf
                })
        return ranked_search_results

    def _sort_documents_by_rank(self, ranked_search_results):
        sorted_documents = []
        for documentIdentifier in ranked_search_results:
            keywords = ranked_search_results[documentIdentifier]
            tfidfs = []
            for entry in keywords:
                tfidfs.append(entry["tfidf"])
            final_score = 0
            for tfidf in tfidfs:
                final_score += math.pow(tfidf, 2)
            final_score = math.sqrt(final_score)
            sorted_documents.append({
                    "document" : documentIdentifier.upper(),
                    "score" : final_score
                })
        sorted_documents.sort(key=lambda document: document["score"], reverse=True)
        return sorted_documents

    def _expand_keywords(self, keywords):
        expanded_keywords = [keywords]
        for word in keywords:
                synonyms = wn.synonyms(word, lang="ita")
                expanded_keywords += synonyms
        return [item for sublist in expanded_keywords for item in sublist]

    def _remove_stopwords(self, keywords):
        keywords_no_stopwords = []
        for word in keywords:
            if word not in self.italian_stopwords:
                keywords_no_stopwords.append(word)
        return keywords_no_stopwords

    def _search_with_score(self, query, include_synonyms=False, include_stopwords=False):
        original_keywords = [word.lower() for word in re.findall(r'\w+', query)]

        next_keywords = self._remove_stopwords(original_keywords) if not include_stopwords else original_keywords
        next_keywords = self._expand_keywords(next_keywords) if include_synonyms else next_keywords

        search_results = self._find_documents(next_keywords, self.index)
        ranked_search_results = self._rank_documents(search_results, self.index)
        sorted_search_results = self._sort_documents_by_rank(ranked_search_results)
        return sorted_search_results

