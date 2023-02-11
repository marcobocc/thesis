from tribunale.TribunaleSearchEngine import *
from tribunale.TribunaleDataLoader import *
from NewSearchEngine import NewSearchEngine
import re
import random

class TestSuite:
    def __init__(self):
        self.tribunaleSearchEngine = TribunaleSearchEngine()
        self.tribunaleDataLoader = TribunaleDataLoader()
        self.newSearchEngine = NewSearchEngine(self.tribunaleDataLoader.documents)

    def _search_in_new_engine(self, query):
        return {
            "excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False),
            "excl_synonyms_incl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=True),
            "incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False),
            "incl_synonyms_incl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=True)
        }

    def _search_in_tribunale(self, query):
        return self.tribunaleSearchEngine.search_in_tribunale(query)

    def _evaluate_search_results(self, search_results, relevantDocumentTitles):
        num_relevant_documents_found = 0
        num_relevant_documents_in_top_positions = 0
        num_relevant_documents_in_top_5 = 0
        num_relevant_documents_in_top_10 = 0
        num_relevant_documents_in_top_15 = 0

        total_number_of_documents_returned = len(search_results)
        total_number_of_relevant_documents = len(relevantDocumentTitles)

        for relevantDocumentTitle in relevantDocumentTitles:
            position_in_search_results = None
            document_found = False
            try:
                position_in_search_results = search_results.index(relevantDocumentTitle.upper())
            except ValueError:
                pass
            finally:
                document_found = position_in_search_results is not None
                if document_found:
                    num_relevant_documents_found += 1
                    if position_in_search_results < total_number_of_relevant_documents:
                        num_relevant_documents_in_top_positions += 1
                    if position_in_search_results < 15:
                        num_relevant_documents_in_top_15 += 1
                        if position_in_search_results < 10:
                            num_relevant_documents_in_top_10 += 1
                            if position_in_search_results < 5:
                                num_relevant_documents_in_top_5 += 1

        recall = num_relevant_documents_found / total_number_of_relevant_documents
        precision = num_relevant_documents_found / total_number_of_documents_returned if total_number_of_documents_returned else 1
        f1_score = 2 * recall * precision / (precision + recall) if (precision + recall) else 0

        success_in_top_positions = 0
        success_in_top_5 = 0
        success_in_top_10 = 0
        success_in_top_15 = 0

        if num_relevant_documents_found:
            success_in_top_positions = num_relevant_documents_in_top_positions / num_relevant_documents_found
            success_in_top_5 = num_relevant_documents_in_top_5 / num_relevant_documents_found
            success_in_top_10 = num_relevant_documents_in_top_10 / num_relevant_documents_found
            success_in_top_15 = num_relevant_documents_in_top_15 / num_relevant_documents_found

        statistics = {
            "recall" : recall,
            "precision" : precision,
            "f1_score" : f1_score,
            "num_relevant_in_top_positions" : num_relevant_documents_in_top_positions,
            "num_relevant_in_top_5" : num_relevant_documents_in_top_5,
            "num_relevant_in_top_10" : num_relevant_documents_in_top_10,
            "num_relevant_in_top_15" : num_relevant_documents_in_top_15,
            "success_in_top_positions_perc" : success_in_top_positions,
            "success_in_top_5_perc" : success_in_top_5 * 100,
            "success_in_top_10_perc" : success_in_top_10 * 100,
            "success_in_top_15_perc" : success_in_top_15 * 100
        }

        return statistics

    def _search_and_compare(self, query, documentTitles):
        searchResultsFromTribunale = self._search_in_tribunale(query)
        searchResultsFromNewEngine_AllConfigs = self._search_in_new_engine(query)
        test_results_tribunale = {
            "current_search_engine" : {
                "query" : query,
                "ground-truth" : documentTitles,
                "search_results" : searchResultsFromTribunale,
                "statistics" : self._evaluate_search_results(searchResultsFromTribunale, documentTitles)
            }
        }
        test_results_new_engine_all_configs = {
            "new_search_engine[{}]".format(config) : {
                "query" : query,
                "ground-truth" : documentTitles,
                "search_results" : searchResultsFromNewEngine_AllConfigs[config],
                "statistics" : self._evaluate_search_results(searchResultsFromNewEngine_AllConfigs[config], documentTitles)
            }
        for config in searchResultsFromNewEngine_AllConfigs
        }
        return test_results_tribunale | test_results_new_engine_all_configs

    def _generate_random_query_from_document(self, document, num_words_to_select):
        words = list(set(re.findall(r'\w+', document["contents"]) + re.findall(r'\w+', document["identifier"])))
        filtered = [w for w in words if len(w) >= 2] # Do not sample words with 1 character
        # Assumption: the order of the keywords does not matter in a single document
        query = " ".join(random.sample(filtered, num_words_to_select))
        return query

    def _generate_random_queries_from_many_documents(self, documents, num_words_to_select_per_document):
        queries = [self._generate_random_query_from_document(document, num_words_to_select_per_document) for document in documents]
        # Assumption: keep the keywords that belong to the same document close together
        return queries

    def searchingForOneKeywordFromOneDocument(self):
        document = random.sample(self.tribunaleDataLoader.documents, 3)[0]
        query = self._generate_random_query_from_document(document, num_words_to_select=1)
        print("SEARCH QUERY:", query)
        searchResultsFromTribunale = self._search_and_compare(query, documentTitles=[document["identifier"].upper()])
        print(json.dumps(searchResultsFromTribunale, indent=4))

    def searchingForOneKeywordFromManyDocuments(self, num_documents_to_select):
        documents = random.sample(self.tribunaleDataLoader.documents, num_documents_to_select)
        queries = self._generate_random_queries_from_many_documents(documents, num_words_to_select_per_document=1)
        print("SEARCH QUERIES:", queries)
        query = " ".join(queries)
        documentTitles = [document["identifier"].upper() for document in documents]
        searchResultsFromTribunale = self._search_and_compare(query, documentTitles=documentTitles)
        print(json.dumps(searchResultsFromTribunale, indent=4))


testSuite = TestSuite()
testSuite.searchingForOneKeywordFromManyDocuments(num_documents_to_select=3)