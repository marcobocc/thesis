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
            "0.9_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.9),
            "0.8_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.8),
            "0.7_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.7),
            "0.6_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.6),
            "0.5_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.5),
            "0.4_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.4),
            "0.3_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.3),
            "0.2_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.2),
            "0.1_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.1),
            "0.05_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.05),
            "0.01_cutoff_excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False, score_threshold=0.01),

            "0.9_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.9),
            "0.8_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.8),
            "0.7_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.7),
            "0.6_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.6),
            "0.5_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.5),
            "0.4_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.4),
            "0.3_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.3),
            "0.2_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.2),
            "0.1_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.1),
            "0.05_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.05),
            "0.01_cutoff_incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False, score_threshold=0.01),

            "excl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=False),
            "excl_synonyms_incl_stopwords" : self.newSearchEngine.search(query, include_synonyms=False, include_stopwords=True),
            "incl_synonyms_excl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=False),
            "incl_synonyms_incl_stopwords" : self.newSearchEngine.search(query, include_synonyms=True, include_stopwords=True),

        }

    def _search_in_tribunale(self, query):
        return self.tribunaleSearchEngine.search_in_tribunale(query)

    def _evaluate_search_results(self, search_results, relevantDocumentTitles):
        num_relevant_documents_found = 0
        total_number_of_documents_returned = len(search_results)
        total_number_of_relevant_documents = len(relevantDocumentTitles)

        total_num_of_documents_database = len(self.tribunaleDataLoader.documents)
        total_number_of_irrelevant_documents_database = total_num_of_documents_database - total_number_of_relevant_documents

        max_k = 15
        num_relevants_in_top_k = [0 for i in range(max_k)]

        # Preprocessing
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
                if (document_found):
                    num_relevant_documents_found += 1
                    for i in range(position_in_search_results, max_k):
                        num_relevants_in_top_k[i] += 1

        # Total recall
        recall = num_relevant_documents_found / total_number_of_relevant_documents

        # Total precision
        precision = 0
        if (total_number_of_relevant_documents == 0 and total_number_of_documents_returned == 0):
            precision = 1
        elif (total_number_of_relevant_documents > 0 and total_number_of_documents_returned == 0):
            precision = 0
        else:
            precision = num_relevant_documents_found / total_number_of_documents_returned if total_number_of_documents_returned else 0

        # F1 Score
        f1_score = 2 * recall * precision / (precision + recall) if (precision + recall) else 0

        # Fallout
        irrelevant_documents_found = total_number_of_documents_returned - num_relevant_documents_found
        fallout = irrelevant_documents_found / total_number_of_irrelevant_documents_database

        # How many of the top K positions are filled with relevant documents (K capped at total_number_of_documents_returned)
        precision_at_top_k = [0 for i in range(max_k)]
        for i in range(max_k):
            if (total_number_of_documents_returned):
                precision_at_top_k[i] = num_relevants_in_top_k[i] / min(i+1, total_number_of_documents_returned)
            else:
                precision_at_top_k[i] = precision

        # Create dictionary of results
        statistics = {
            "recall" : recall,
        }

        labels_precision_at_top_k = [
            "precision_at_top_{}".format(k+1)
            for k in range(max_k)
        ]
        for i in range(max_k):
            label = labels_precision_at_top_k[i]
            statistics[label] = precision_at_top_k[i]

        extra_stats = {
            "f1_score" : f1_score,
            "precision" : precision,
            "fallout" : fallout,
            "num_relevant_found" : num_relevant_documents_found,
            "num_documents_found" : total_number_of_documents_returned,
            "actual_relevants" : total_number_of_relevant_documents,
            "num_irrelevant_found" : irrelevant_documents_found,
            "actual_irrelevants" : total_number_of_irrelevant_documents_database
        }

        labels_num_relevants_at_top_k = [
            "num_relevants_at_top_{}".format(k+1)
            for k in range(max_k)
        ]
        for i in range(max_k):
            label = labels_num_relevants_at_top_k[i]
            extra_stats[label] = num_relevants_in_top_k[i]

        statistics.update(extra_stats)
        return statistics

    def _search_and_compare(self, query, documentTitles):
        searchResultsFromTribunale = self._search_in_tribunale(query)
        searchResultsFromNewEngine_AllConfigs = self._search_in_new_engine(query)
        test_results_tribunale = [
            {
                "search_engine" : "current_search_engine",
                "query" : query,
                "search_results" : [search_result["document"] for search_result in searchResultsFromTribunale],
                "ground-truth" : documentTitles,
                "statistics" : self._evaluate_search_results(searchResultsFromTribunale, documentTitles)
            }
        ]
        test_results_new_engine_all_configs = [
            {
                "search_engine" : "new ({})".format(config),
                "query" : query,
                "search_results" : [search_result["document"] for search_result in searchResultsFromNewEngine_AllConfigs[config]],
                "ground-truth" : documentTitles,
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
        df.set_index("search_engine")
        return df

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

    def searchQueryFromGroundTruth(self, query, gt_documents):
        actual_documentTitles = [document["identifier"].upper() for document in self.tribunaleDataLoader.documents]
        gt_documentTitles = [document.upper() for document in gt_documents]
        for gt_documentTitle in gt_documentTitles:
            if gt_documentTitle not in actual_documentTitles:
                raise Exception("Ground truth contains document {} that is not present in the database".format(gt_documentTitle))
        testResults = self._search_and_compare(query, documentTitles=gt_documentTitles)
        # print(json.dumps(testResults))
        df = self._convert_test_results(testResults)
        return df

    def _run_groundtruth_suite(self, test_dir, gt_tests):
        num_tests = len(gt_tests)
        test_name = "gt_dataset"
        all_dfs = []
        successful_iterations = 0
        output_file = test_dir + "/" + test_name + ".csv"
        print(bcolors.INFO + "-" * 100)
        print(bcolors.INFO + "Starting tests [{}]".format(output_file))
        print(bcolors.INFO + "-" * 100)
        iterations = num_tests
        while successful_iterations < iterations:
            query = gt_tests[successful_iterations]["query"]
            gt_documents = gt_tests[successful_iterations]["documents"]
            try:
                df = self.searchQueryFromGroundTruth(query, gt_documents)
                all_dfs.append(df)
                successful_iterations = successful_iterations + 1
                if not (successful_iterations % 1):
                    print(bcolors.GREEN + "Tests run: {} (out of {})".format(successful_iterations, iterations))
            except Exception as e:
                print(bcolors.YELLOW + "Repeating test [reason: {}]".format(str(e)))
        final_df = pd.concat(all_dfs)
        summary = self._summarize_test_results(final_df)
        summary.to_csv(output_file)
        return summary

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
                if not (successful_iterations % 1):
                    print(bcolors.GREEN + "({}/{}) Tests run: {} (out of {})".format(test_number, num_tests, successful_iterations, iterations))
            except Exception as e:
                print(bcolors.YELLOW + "Repeating test [reason: {}]".format(str(e)))
        final_df = pd.concat(all_dfs)
        summary = self._summarize_test_results(final_df)
        summary.to_csv(output_file)
        return summary

    def _run_validation_suite(self, tests, iterations, test_dir):
        num_tests = len(tests)
        for i in range(len(tests)):
            max_keywords = tests[i]["max_keywords"]
            num_documents = tests[i]["num_documents"]
            test_name = "max_{}_words_{}_documents".format(max_keywords, num_documents)
            function = self.searchOneKeywordPerDocument if max_keywords == 1 else self.searchMultipleKeywordsPerDocument
            self._run_test(test_dir, test_name, i+1, num_tests, iterations, function, num_documents_to_select=num_documents)

    def run_validation_suite(self, test_suite_name):
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
                "num_documents": 4
            },
        ]
        self._run_validation_suite(tests, iterations, test_dir)
        Report(test_suite_name)

    def run_groundtruth_suite(self, test_suite_name):
        base_dir = "tests"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        test_dir = base_dir + "/" + test_suite_name
        os.mkdir(test_dir)
        f = open('gt.json')
        gt_tests = json.load(f)
        f.close()
        self._run_groundtruth_suite(test_dir, gt_tests)
        Report(test_suite_name)

test_date = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
test_name = "euclidean"

test_suite_name = test_date + "_" + test_name
testSuite = TestSuite()
#testSuite.run_validation_suite(test_suite_name)
testSuite.run_groundtruth_suite(test_suite_name)