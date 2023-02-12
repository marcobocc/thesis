from cmath import exp
import re
import math
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from tribunale.TribunaleDataLoader import TribunaleDataLoader
import json

class NewSearchEngine:
    def __init__(self, documents):
        self.italian_stopwords = stopwords.words("italian")
        self.stemmer = nltk.stem.snowball.ItalianStemmer()
        self.documents = documents
        self.index = self._build_index(documents)

    def search(self, query, include_synonyms=False, include_stopwords=False, score_threshold=0.0):
        sorted_search_results = self._search_with_score(query, include_synonyms, include_stopwords, score_threshold)
        return sorted_search_results

    def _build_index(self, documents):
        index = {}
        for document in documents:
            title = re.findall(r'\w+', document["identifier"])
            body = re.findall(r'\w+', document["contents"])
            stemmed_title = set([self.stemmer.stem(t.lower()) for t in title])
            document_words = [word.lower() for word in body + title]
            for word in document_words:
                stemmed_word = self.stemmer.stem(word)
                if stemmed_word not in index:
                    index[stemmed_word] = {}
                documentName = document["identifier"]
                if documentName not in index[stemmed_word]:
                    index[stemmed_word][documentName] = {
                        "count" : 0,
                        "tf" : 0,
                        "in_title" : True if stemmed_word in stemmed_title else False,
                        "in_tags" : False
                    }
                index[stemmed_word][documentName]["count"] += 1
                index[stemmed_word][documentName]["tf"] = index[stemmed_word][documentName]["count"] / len(document_words)
        return index

    def _get_idf(self, term):
        stemmed_term = self.stemmer.stem(term)
        df = len(self.index[stemmed_term]) if stemmed_term in self.index else 0.0
        num_documents = len(self.documents)
        idf = math.log((num_documents + 1) / (float(df) + 1.0)) # Smooth TF-IDF to prevent division by 0
        return idf

    def _get_tf(self, term, documentIdentifier):
        stemmed_term = self.stemmer.stem(term)
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
                stemmed_word = self.stemmer.stem(word)
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
        stemmed_query = [self.stemmer.stem(term) for term in query_keywords]
        query_scores = {}
        for x in set(query_keywords):
            tf = stemmed_query.count(self.stemmer.stem(x)) / len(query_keywords)
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
        cosine_similarity = dotproduct
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
        euclidean_similarity = 1 / (math.exp(euclidean_distance))
        return euclidean_similarity

    def _sort_documents_by_rank(self, search_results, query_keywords, score_threshold=0.0):
        scored_documents = self._get_scored_documents(search_results, query_keywords)
        query_scores = self._get_query_score(query_keywords)
        ranked_documents = []
        for documentIdentifier in scored_documents:
            document_scores = scored_documents[documentIdentifier]
            cosine_similarity = self._get_cosine_similarity(query_scores, document_scores)
            ranked_documents.append({
                    "document" : documentIdentifier.upper(),
                    "score" : cosine_similarity,
                    "keyword_scores" : document_scores
                })
        ranked_documents.sort(key=lambda document: document["score"], reverse=True)
        if ranked_documents:
            max_value = ranked_documents[0]["score"]
            thresholded = [d for d in ranked_documents if d["score"] > max_value * score_threshold]
            return thresholded
        return ranked_documents

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

    def _search_with_score(self, query, include_synonyms=False, include_stopwords=False, score_threshold=0.0):
        original_keywords = [word.lower() for word in re.findall(r'\w+', query)]

        next_keywords = self._remove_stopwords(original_keywords) if not include_stopwords else original_keywords
        next_keywords = self._expand_keywords(next_keywords) if include_synonyms else next_keywords
        query_keywords = next_keywords

        search_results = self._find_documents(query_keywords)
        sorted_search_results = self._sort_documents_by_rank(search_results, query_keywords, score_threshold=score_threshold)
        return sorted_search_results

#engine = NewSearchEngine(TribunaleDataLoader().documents)
#results = engine.search("quanto costa rinunciare eredita", score_threshold=0.3)
#print(json.dumps(results, indent=4))