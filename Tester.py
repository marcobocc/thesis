from SearchEngineURP import SearchEngineURP
from TribunaleDataLoader import TribunaleDataLoader
from SearchEnginePrototype import SearchEnginePrototype
from SearchEngineConfig import SearchEngineConfig
from Report import Report
from TestGenerator import TestGenerator

import os
import json
import pandas as pd

from datetime import datetime
import itertools


class bcolors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    INFO = '\x1b[0m'


class Tester:
    def __init__(self):
        self.searchEngineURP = SearchEngineURP()
        self.tribunaleDataLoader = TribunaleDataLoader()

        print("Initializing the search engines...")
        self.configs = self.generate_search_engine_configs()
        self.searchEngines = [SearchEnginePrototype(config) for config in self.configs]

    def generate_search_engine_configs(self):
        choices = {
            "use_stemming": [True, False],
            "remove_stopwords": [True, False],
            "expand_synonyms": [True, False],
            "cutoff": [0.0, 0.3, 0.5, 0.7, 0.9],
            "similarity": [SearchEngineConfig.COSINE, SearchEngineConfig.DOT_PRODUCT, SearchEngineConfig.EUCLIDEAN]
        }
        combinations = [dict(zip(choices.keys(), instance)) for instance in itertools.product(*choices.values())]
        configs = [
            SearchEngineConfig(
                remove_stopwords=c["remove_stopwords"],
                use_stemming=c["use_stemming"],
                expand_synonyms=c["expand_synonyms"],
                cutoff=c["cutoff"],
                similarity=c["similarity"]
            )
            for c in combinations
        ]
        return configs

    def _search_in_all(self, query, documentTitles):
        query_results_from_URP_with_scores = self.searchEngineURP.search(query)
        query_results_URP = [
            {
                "search_engine": self.searchEngineURP.config.config_name,
                "query": query,
                "search_results": [search_result["document"] for search_result in query_results_from_URP_with_scores],
                "ground-truth": documentTitles,
                "statistics": self._evaluate_search_results(query_results_from_URP_with_scores, documentTitles),
                "config": self.searchEngineURP.config.to_dict()
            }
        ]
        query_results_from_prototypes_with_scores = [s.search(query) for s in self.searchEngines]
        query_results_prototypes = [
            {
                "search_engine": self.searchEngines[i].config.config_name,
                "query": query,
                "search_results": [search_result["document"] for search_result in query_results_from_prototypes_with_scores[i]],
                "ground-truth": documentTitles,
                "statistics": self._evaluate_search_results(query_results_from_prototypes_with_scores[i], documentTitles),
                "config": self.searchEngines[i].config.to_dict(),
            }
            for i in range(len(query_results_from_prototypes_with_scores))
        ]
        return query_results_URP + query_results_prototypes

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
            "recall": recall,
        }

        labels_precision_at_top_k = [
            "precision_at_top_{}".format(k+1)
            for k in range(max_k)
        ]
        for i in range(max_k):
            label = labels_precision_at_top_k[i]
            statistics[label] = precision_at_top_k[i]

        extra_stats = {
            "f1_score": f1_score,
            "precision": precision,
            "fallout": fallout,
            "num_relevant_found": num_relevant_documents_found,
            "num_documents_found": total_number_of_documents_returned,
            "actual_relevants": total_number_of_relevant_documents,
            "num_irrelevant_found": irrelevant_documents_found,
            "actual_irrelevants": total_number_of_irrelevant_documents_database
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

    def _convert_test_results(self, test_results):
        dataframe = pd.concat([pd.Series(test_result).to_frame().T for test_result in test_results])
        statsDataframe = pd.json_normalize(dataframe["statistics"])
        final_dataframe = pd.concat([dataframe.drop(["statistics"], axis=1).reset_index(), statsDataframe], axis=1)
        return final_dataframe

    def _summarize_test_results(self, df):
        return df.reset_index(drop=True)

    def searchQueryFromGroundTruth(self, query, gt_documents):
        actual_documentTitles = [document["identifier"].upper() for document in self.tribunaleDataLoader.documents]
        gt_documentTitles = [document.upper() for document in gt_documents]
        for gt_documentTitle in gt_documentTitles:
            if gt_documentTitle not in actual_documentTitles:
                raise Exception("Ground truth contains document {} that is not present in the database".format(gt_documentTitle))
        testResults = self._search_in_all(query, documentTitles=gt_documentTitles)
        # print(json.dumps(testResults))
        df = self._convert_test_results(testResults)
        return df

    def _run_tests(self, test_dir, tests):
        num_tests = len(tests)
        test_name = "gt_dataset"
        all_dfs = []
        successful_iterations = 0
        output_file = test_dir + "/" + test_name + ".csv"
        print(bcolors.INFO + "-" * 100)
        print(bcolors.INFO + "Starting tests [{}]".format(output_file))
        print(bcolors.INFO + "-" * 100)
        iterations = num_tests
        while successful_iterations < iterations:
            query = tests[successful_iterations]["query"]
            gt_documents = tests[successful_iterations]["documents"]
            try:
                df = self.searchQueryFromGroundTruth(query, gt_documents)
                all_dfs.append(df)
                successful_iterations = successful_iterations + 1
                if not (successful_iterations % 1):
                    print(bcolors.GREEN + "{}/{}: Successful".format(successful_iterations, iterations))
            except Exception as e:
                print(bcolors.YELLOW + "{}/{}: Repeating test [reason: {}]".format(successful_iterations, iterations, str(e)))
        final_df = pd.concat(all_dfs)
        summary = self._summarize_test_results(final_df)
        summary.to_csv(output_file, index=False)
        return summary

    def run_groundtruth_tests(self, filename):
        test_date = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
        base_dir = "tests"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        test_dir = base_dir + "/" + test_date
        os.mkdir(test_dir)
        f = open(filename)
        gt_tests = json.load(f)
        f.close()
        self._run_tests(test_dir, gt_tests)
        Report(test_date)

    def run_autogenerated_tests(self, num_tests):
        test_date = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
        base_dir = "tests"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        test_dir = base_dir + "/" + test_date
        os.mkdir(test_dir)
        testGenerator = TestGenerator()
        gt_tests = testGenerator.generate(num_samples=num_tests, num_documents_to_select=2, num_keywords_per_document=2)
        self._run_tests(test_dir, gt_tests)
        Report(test_date)
