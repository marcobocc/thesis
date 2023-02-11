from tribunale.TribunaleSearchEngine import *
from tribunale.TribunaleDataLoader import *
from NewSearchEngine import NewSearchEngine
from Report import Report
import re
import random
import pandas as pd
import matplotlib as mpl
from datetime import datetime
from nltk.corpus import stopwords
import os

class bcolors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    INFO = '\x1b[0m'

class TestSuite:
    def __init__(self):
        self.tribunaleSearchEngine = TribunaleSearchEngine()
        self.tribunaleDataLoader = TribunaleDataLoader()
        self.newSearchEngine = NewSearchEngine(self.tribunaleDataLoader.documents)
        self.italian_stopwords = stopwords.words("italian")

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
        num_relevant_documents_in_top_1 = 0
        num_relevant_documents_in_top_2 = 0
        num_relevant_documents_in_top_3 = 0
        num_relevant_documents_in_top_5 = 0
        num_relevant_documents_in_top_8 = 0
        num_relevant_documents_in_top_10 = 0
        num_relevant_documents_in_top_15 = 0

        total_number_of_documents_returned = len(search_results)
        total_number_of_relevant_documents = len(relevantDocumentTitles)

        search_results_doc_titles = [search_result["document"] for search_result in search_results]

        for relevantDocumentTitle in relevantDocumentTitles:
            position_in_search_results = None
            document_found = False
            try:
                position_in_search_results = search_results_doc_titles.index(relevantDocumentTitle.upper())
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
                            if position_in_search_results < 8:
                                num_relevant_documents_in_top_8 += 1
                                if position_in_search_results < 5:
                                    num_relevant_documents_in_top_5 += 1
                                    if position_in_search_results < 3:
                                        num_relevant_documents_in_top_3 += 1
                                        if position_in_search_results < 2:
                                            num_relevant_documents_in_top_2 += 1
                                            if position_in_search_results < 1:
                                                num_relevant_documents_in_top_1 += 1

        recall = num_relevant_documents_found / total_number_of_relevant_documents
        precision = num_relevant_documents_found / total_number_of_documents_returned if total_number_of_documents_returned else 0
        f1_score = 2 * recall * precision / (precision + recall) if (precision + recall) else 0

        perc_of_relevants_in_top_positions = 0
        perc_of_relevants_in_top_1 = 0
        perc_of_relevants_in_top_2 = 0
        perc_of_relevants_in_top_3 = 0
        perc_of_relevants_in_top_5 = 0
        perc_of_relevants_in_top_8 = 0
        perc_of_relevants_in_top_10 = 0
        perc_of_relevants_in_top_15 = 0

        if num_relevant_documents_found:
            # Out of all relevant documents retrieved, what is the % of them that occupies the top K positions
            perc_of_relevants_in_top_positions = num_relevant_documents_in_top_positions / num_relevant_documents_found * recall
            perc_of_relevants_in_top_1 = num_relevant_documents_in_top_1 / num_relevant_documents_found * recall
            perc_of_relevants_in_top_2 = num_relevant_documents_in_top_2 / num_relevant_documents_found * recall
            perc_of_relevants_in_top_3 = num_relevant_documents_in_top_3 / num_relevant_documents_found * recall
            perc_of_relevants_in_top_5 = num_relevant_documents_in_top_5 / num_relevant_documents_found * recall
            perc_of_relevants_in_top_8 = num_relevant_documents_in_top_8 / num_relevant_documents_found * recall
            perc_of_relevants_in_top_10 = num_relevant_documents_in_top_10 / num_relevant_documents_found * recall
            perc_of_relevants_in_top_15 = num_relevant_documents_in_top_15 / num_relevant_documents_found * recall

        # Percentage of top K positions that is filled with a relevant search result
        perc_of_top_positions_occupied_by_relevants = num_relevant_documents_in_top_positions / total_number_of_relevant_documents
        perc_of_top_1_occupied_by_relevants = num_relevant_documents_in_top_1 / 1
        perc_of_top_2_occupied_by_relevants = num_relevant_documents_in_top_2 / 2
        perc_of_top_3_occupied_by_relevants = num_relevant_documents_in_top_3 / 3
        perc_of_top_5_occupied_by_relevants = num_relevant_documents_in_top_5 / 5
        perc_of_top_8_occupied_by_relevants = num_relevant_documents_in_top_8 / 8
        perc_of_top_10_occupied_by_relevants = num_relevant_documents_in_top_10 / 10
        perc_of_top_15_occupied_by_relevants = num_relevant_documents_in_top_15 / 15

        statistics = {
            "recall" : recall,
            "precision" : precision,
            "f1_score" : f1_score,
            "num_relevant_found" : num_relevant_documents_found,
            "num_documents_found" : total_number_of_documents_returned,
            "actual_relevants" : total_number_of_relevant_documents,
            "num_relevant_in_top_positions" : num_relevant_documents_in_top_positions,
            "num_relevant_in_top_1" : num_relevant_documents_in_top_1,
            "num_relevant_in_top_2" : num_relevant_documents_in_top_2,
            "num_relevant_in_top_3" : num_relevant_documents_in_top_3,
            "num_relevant_in_top_5" : num_relevant_documents_in_top_5,
            "num_relevant_in_top_8" : num_relevant_documents_in_top_8,
            "num_relevant_in_top_10" : num_relevant_documents_in_top_10,
            "num_relevant_in_top_15" : num_relevant_documents_in_top_15,
            "perc_of_relevants_in_top_positions" : perc_of_relevants_in_top_positions,
            "perc_of_relevants_in_top_1" : perc_of_relevants_in_top_1,
            "perc_of_relevants_in_top_2" : perc_of_relevants_in_top_2,
            "perc_of_relevants_in_top_3" : perc_of_relevants_in_top_3,
            "perc_of_relevants_in_top_5" : perc_of_relevants_in_top_5,
            "perc_of_relevants_in_top_8" : perc_of_relevants_in_top_8,
            "perc_of_relevants_in_top_10" : perc_of_relevants_in_top_10,
            "perc_of_relevants_in_top_15" : perc_of_relevants_in_top_15,
            "perc_of_top_positions_occupied_by_relevants" : perc_of_top_positions_occupied_by_relevants,
            "perc_of_top_1_occupied_by_relevants" : perc_of_top_1_occupied_by_relevants,
            "perc_of_top_2_occupied_by_relevants" : perc_of_top_2_occupied_by_relevants,
            "perc_of_top_3_occupied_by_relevants" : perc_of_top_3_occupied_by_relevants,
            "perc_of_top_5_occupied_by_relevants" : perc_of_top_5_occupied_by_relevants,
            "perc_of_top_8_occupied_by_relevants" : perc_of_top_8_occupied_by_relevants,
            "perc_of_top_10_occupied_by_relevants" : perc_of_top_10_occupied_by_relevants,
            "perc_of_top_15_occupied_by_relevants" : perc_of_top_15_occupied_by_relevants,
            "scores" : [search_result["score"] for search_result in search_results]
        }

        return statistics

    def _search_and_compare(self, query, documentTitles):
        searchResultsFromTribunale = self._search_in_tribunale(query)
        searchResultsFromNewEngine_AllConfigs = self._search_in_new_engine(query)
        test_results_tribunale = [
            {
                "search_engine" : "current_search_engine",
                "query" : query,
                "ground-truth" : documentTitles,
                "search_results" : [search_result["document"] for search_result in searchResultsFromTribunale],
                "statistics" : self._evaluate_search_results(searchResultsFromTribunale, documentTitles)
            }
        ]
        test_results_new_engine_all_configs = [
            {
                "search_engine" : "new ({})".format(config),
                "query" : query,
                "ground-truth" : documentTitles,
                "search_results" : [search_result["document"] for search_result in searchResultsFromNewEngine_AllConfigs[config]],
                "statistics" : self._evaluate_search_results(searchResultsFromNewEngine_AllConfigs[config], documentTitles)
            }
        for config in searchResultsFromNewEngine_AllConfigs
        ]
        return test_results_tribunale + test_results_new_engine_all_configs

    def _generate_random_query_from_document(self, document, num_words_to_select):
        words = list(set(re.findall(r'\w+', document["contents"]) + re.findall(r'\w+', document["identifier"])))
        filtered = [w.lower() for w in words if len(w) > 2] # Do not sample words with 2 characters
        # Assumption: the order of the keywords does not matter in a single document
        selected_words = []
        while True: # Do not build a query composed only of stopwords
            selected_words = random.sample(filtered, num_words_to_select)
            query = " ".join(selected_words)
            num_stopwords = len([w for w in selected_words if w in self.italian_stopwords])
            if num_stopwords < len(selected_words):
                break
        return query

    def _generate_random_queries_from_many_documents(self, documents, num_words_to_select_per_document):
        queries = [self._generate_random_query_from_document(document, num_words_to_select_per_document) for document in documents]
        # Assumption: keep the keywords that belong to the same document close together
        return queries

    def _convert_test_results(self, test_results):
        dataframe = pd.concat([pd.Series(test_result).to_frame().T for test_result in test_results])
        statsDataframe = pd.json_normalize(dataframe["statistics"]).reset_index()
        final_dataframe = pd.concat([dataframe.drop(["statistics"], axis=1).reset_index(), statsDataframe], axis=1)
        return final_dataframe

    def _summarize_test_results(self, df):
        current_search_engine = df.loc[df["search_engine"] ==  "current_search_engine"]
        new_search_engine = df.loc[df["search_engine"] !=  "current_search_engine"]
        excl_synonyms_excl_stopwords = new_search_engine.loc[new_search_engine["search_engine"] == "new (excl_synonyms_excl_stopwords)"]
        incl_synonyms_excl_stopwords = new_search_engine.loc[new_search_engine["search_engine"] == "new (incl_synonyms_excl_stopwords)"]
        excl_synonyms_incl_stopwords = new_search_engine.loc[new_search_engine["search_engine"] == "new (excl_synonyms_incl_stopwords)"]
        incl_synonyms_incl_stopwords = new_search_engine.loc[new_search_engine["search_engine"] == "new (incl_synonyms_incl_stopwords)"]
        summary_dfs = [current_search_engine, excl_synonyms_excl_stopwords, incl_synonyms_excl_stopwords, excl_synonyms_incl_stopwords, incl_synonyms_incl_stopwords]
        return summary_dfs

    def searchOneKeywordPerDocument(self, num_documents_to_select):
        documents = random.sample(self.tribunaleDataLoader.documents, num_documents_to_select)
        num_words_to_select_per_document = 1
        queries = self._generate_random_queries_from_many_documents(documents, num_words_to_select_per_document)
        query = " ".join(queries)
        documentTitles = [document["identifier"].upper() for document in documents]
        testResults = self._search_and_compare(query, documentTitles=documentTitles)
        # print(json.dumps(testResults))
        df = self._convert_test_results(testResults)
        return df

    def searchMultipleKeywordsPerDocument(self, num_documents_to_select):
        documents = random.sample(self.tribunaleDataLoader.documents, num_documents_to_select)
        num_words_to_select_per_document = random.sample(range(2, 4), 1)[0]
        queries = self._generate_random_queries_from_many_documents(documents, num_words_to_select_per_document)
        query = " ".join(queries)
        documentTitles = [document["identifier"].upper() for document in documents]
        testResults = self._search_and_compare(query, documentTitles=documentTitles)
        # print(json.dumps(testResults))
        df = self._convert_test_results(testResults)
        return df

    def _run_test(self, folder, test_name, test_number, num_tests, iterations, function, **kwargs):
        all_dfs = []
        successful_iterations = 0
        output_file = folder + "/" + test_name + ".csv"
        print(bcolors.INFO + "-" * 100)
        print(bcolors.INFO + "Starting tests [{}]".format(output_file))
        print(bcolors.INFO + "-" * 100)
        while successful_iterations < iterations:
            try:
                df = function(**kwargs)
                all_dfs.append(df)
                successful_iterations = successful_iterations + 1
                if not (successful_iterations % 2):
                    print(bcolors.GREEN + "({}/{}) Tests run: {} (out of {})".format(test_number, num_tests, successful_iterations, iterations))
            except Exception as e:
                print(bcolors.YELLOW + "Repeating test [reason: {}]".format(str(e)))
        final_df = pd.concat(all_dfs)
        summary = pd.concat(self._summarize_test_results(final_df))
        summary.to_csv(output_file)
        return summary

    def run_suite(self, tests, iterations, test_dir):
        num_tests = len(tests)
        for i in range(len(tests)):
            max_keywords = tests[i]["max_keywords"]
            num_documents = tests[i]["num_documents"]
            test_name = "max_{}_words_{}_documents".format(max_keywords, num_documents)
            function = self.searchOneKeywordPerDocument if max_keywords == 1 else self.searchMultipleKeywordsPerDocument
            self._run_test(test_dir, test_name, i+1, num_tests, iterations, function, num_documents_to_select=num_documents)


test_suite_name = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
base_dir = "tests"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

test_dir = base_dir + "/" + test_suite_name
os.mkdir(test_dir)

iterations=500
tests = [
    {
        "max_keywords": 1,
        "num_documents": 1
    },
    {
        "max_keywords": 2,
        "num_documents": 1
    },
    {
        "max_keywords": 3,
        "num_documents": 1
    },
    {
        "max_keywords": 1,
        "num_documents": 2
    },
    {
        "max_keywords": 2,
        "num_documents": 2
    },
    {
        "max_keywords": 3,
        "num_documents": 3
    },
    {
        "max_keywords": 2,
        "num_documents": 5
    },
]
testSuite = TestSuite()
testSuite.run_suite(tests, iterations, test_dir)
report = Report(test_suite_name)