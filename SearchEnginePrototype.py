from typing import List

import re
import math
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from TribunaleDataLoader import TribunaleDataLoader
from SearchEngineConfig import SearchEngineConfig


class SearchEnginePrototype:
    def __init__(self, config: SearchEngineConfig):
        self.config = config

        self._database = TribunaleDataLoader()
        self._italian_stopwords = stopwords.words("italian")
        self._stemmer = nltk.stem.snowball.ItalianStemmer()

        self.documents = self._database.documents
        self.index = self._build_index(self.documents)

    def search(self, query):
        sorted_search_results = self._search_with_score(query)
        return sorted_search_results

    def _stem(self, word):
        if self.config.use_stemming:
            return self._stemmer.stem(word)
        return word

    def _expand_synonyms(self, keywords: List[str]):
        if self.config.expand_synonyms:
            expanded_keywords = [keywords]
            for word in keywords:
                synonyms = wn.synonyms(word, lang="ita")
                expanded_keywords += synonyms
            return [item for sublist in expanded_keywords for item in sublist]
        return keywords

    def _remove_stopwords(self, keywords: List[str]):
        if self.config.remove_stopwords:
            keywords_no_stopwords = []
            for word in keywords:
                if word not in self._italian_stopwords:
                    keywords_no_stopwords.append(word)
            return keywords_no_stopwords
        return keywords

    def _compute_similarity(self, query_scores, document_scores):
        if self.config.similarity == SearchEngineConfig.COSINE:
            return self._get_cosine_similarity(query_scores, document_scores)
        elif self.config.similarity == SearchEngineConfig.DOT_PRODUCT:
            return self._get_dot_product_similarity(query_scores, document_scores) # noqa
        elif self.config.similarity == SearchEngineConfig.EUCLIDEAN:
            return self._get_euclidean_similarity(query_scores, document_scores) # noqa
        raise Exception("Similarity measure does not exist")

    def _build_index(self, documents):
        index = {}
        for document in documents:
            title = re.findall(r'\w+', document["identifier"])
            body = re.findall(r'\w+', document["contents"])
            stemmed_title = set([self._stem(t.lower()) for t in title])
            document_words = [word.lower() for word in body + title]
            for word in document_words:
                stemmed_word = self._stem(word)
                if stemmed_word not in index:
                    index[stemmed_word] = {}
                documentName = document["identifier"]
                if documentName not in index[stemmed_word]:
                    index[stemmed_word][documentName] = {
                        "count": 0,
                        "tf": 0,
                        "in_title": True if stemmed_word in stemmed_title else False, # noqa
                        "in_tags": False
                    }
                index[stemmed_word][documentName]["count"] += 1
                index[stemmed_word][documentName]["tf"] = index[stemmed_word][documentName]["count"] / len(document_words) # noqa
        return index

    def _get_idf(self, term):
        stemmed_term = self._stem(term)
        df = len(self.index[stemmed_term]) if stemmed_term in self.index else 0.0 # noqa
        num_documents = len(self.documents)
        idf = math.log((num_documents) / (float(df) + 1.0)) # noqa
        return idf

    def _get_tf(self, term, documentIdentifier):
        stemmed_term = self._stem(term)
        if stemmed_term not in self.index:
            return 0
        if documentIdentifier not in self.index[stemmed_term]:
            return 0
        return self.index[stemmed_term][documentIdentifier]["tf"]

    def _get_tfidf(self, term, documentIdentifier):
        return self._get_tf(term, documentIdentifier) * self._get_idf(term)

    def _find_documents(self, keywords):
        search_results = {}
        for word in keywords:
            stemmed_word = self._stem(word)
            if stemmed_word in self.index:
                for documentIdentifier in self.index[stemmed_word]:
                    if documentIdentifier not in search_results:
                        search_results[documentIdentifier] = []
                    search_results[documentIdentifier].append(word)
        return search_results

    def _get_scored_documents(self, search_results, query_keywords):
        scored_documents = {}
        for documentIdentifier in search_results:
            scored_documents[documentIdentifier] = {}
            for keyword in set(query_keywords):
                tfidf = self._get_tfidf(keyword, documentIdentifier)
                scored_documents[documentIdentifier][keyword] = tfidf
        return scored_documents

    def _get_query_score(self, query_keywords):
        stemmed_query = [self._stem(term) for term in query_keywords]
        query_scores = {}
        for x in set(query_keywords):
            tf = stemmed_query.count(self._stem(x)) / len(query_keywords)
            idf = self._get_idf(x)
            query_scores[x] = tf * idf
        return query_scores

    def _get_cosine_similarity(self, query_scores, document_scores):
        keywords = [kw for kw in query_scores]
        if len(query_scores) != len(document_scores):
            raise Exception("Vectors are different size")
        q = []
        d = []
        for kw in keywords:
            q.append(query_scores[kw])
            d.append(document_scores[kw])
        dotproduct = 0
        q_square = 0
        d_square = 0
        for i in range(len(q)):
            dotproduct += (q[i] * d[i])
            q_square += q[i] * q[i]
            d_square += d[i] * d[i]
        cosine_similarity = dotproduct / (math.sqrt(q_square) * math.sqrt(d_square)) # noqa
        return cosine_similarity

    def _get_dot_product_similarity(self, query_scores, document_scores):
        keywords = [kw for kw in query_scores]
        if len(query_scores) != len(document_scores):
            raise Exception("Vectors are different size")
        q = []
        d = []
        for kw in keywords:
            q.append(query_scores[kw])
            d.append(document_scores[kw])
        dotproduct = 0
        q_square = 0
        d_square = 0
        for i in range(len(q)):
            dotproduct += (q[i] * d[i])
            q_square += q[i] * q[i]
            d_square += d[i] * d[i]
        cosine_similarity = dotproduct / (math.sqrt(q_square) * math.sqrt(d_square)) # noqa
        return cosine_similarity

    def _get_euclidean_similarity(self, query_scores, document_scores):
        keywords = [kw for kw in query_scores]
        if len(query_scores) != len(document_scores):
            raise Exception("Vectors are different size")
        q = []
        d = []
        for kw in keywords:
            q.append(query_scores[kw])
            d.append(document_scores[kw])
        euclidean_distance = 0
        for i in range(len(q)):
            euclidean_distance += (q[i] - d[i]) * (q[i] - d[i])
        euclidean_distance = euclidean_distance
        euclidean_similarity = 1 / (1 + euclidean_distance)
        return euclidean_similarity

    def _sort_documents_by_rank(self, search_results, query_keywords):
        scored_documents = self._get_scored_documents(search_results, query_keywords) # noqa
        query_scores = self._get_query_score(query_keywords)
        ranked_documents = []
        for documentIdentifier in scored_documents:
            document_scores = scored_documents[documentIdentifier]
            similarity = self._compute_similarity(query_scores, document_scores)  # noqa
            ranked_documents.append({
                    "document": documentIdentifier.upper(),
                    "score": similarity,
                    "keyword_scores": document_scores
                })
        ranked_documents.sort(key=lambda document: document["score"], reverse=True)  # noqa
        if ranked_documents:
            max_value = ranked_documents[0]["score"]
            thresholded = [d for d in ranked_documents if d["score"] >= max_value * self.config.cutoff]  # noqa
            return thresholded
        return ranked_documents

    def _search_with_score(self, query):
        original_keywords = [word.lower() for word in re.findall(r'\w+', query)] # noqa

        next_keywords = self._remove_stopwords(original_keywords)
        next_keywords = self._expand_synonyms(next_keywords)
        query_keywords = next_keywords

        search_results = self._find_documents(query_keywords)
        sorted_search_results = self._sort_documents_by_rank(search_results, query_keywords) # noqa
        return sorted_search_results


config = SearchEngineConfig(
    config_name="test",
    use_stemming=True,
    remove_stopwords=True,
    expand_synonyms=False,
    cutoff=0.0,
    similarity=SearchEngineConfig.EUCLIDEAN
)

searchEngine = SearchEnginePrototype(config)
result = searchEngine.search("unioni civili")
