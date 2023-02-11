import pandas as pd
import os
import glob


class Report:
    def __init__(self, folder):
        path = os.path.join(os.getcwd(), "tests")
        self.dir = os.path.join(path, folder)
        self.save_path = os.path.join(self.dir, "stats.csv")
        self._load_df()

    def _load_df(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        csv_path = os.path.join(self.dir, "**.csv")
        csv_files = glob.glob(csv_path)
        test_dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            filename = file.split("/")[-1].rsplit(".")[0]
            df["test_name"] = filename
            test_dfs.append(df)
        self.df = pd.concat(test_dfs)
        self.summary = self._summarize()
        self.summary.to_csv(self.save_path)

    def _summarize(self):
        sum_columns = [
            "num_relevant_found",
            "num_documents_found",
            "actual_relevants",
            "num_relevant_in_top_positions",
            "num_relevant_in_top_1",
            "num_relevant_in_top_2",
            "num_relevant_in_top_3",
            "num_relevant_in_top_5",
            "num_relevant_in_top_8",
            "num_relevant_in_top_10",
            "num_relevant_in_top_15"
        ]
        stats = self.df.groupby(["test_name", "search_engine"])[sum_columns].sum().reset_index()
        stats["recall"] = stats["num_relevant_found"] / stats["actual_relevants"]
        stats["precision"] = stats["num_relevant_found"] / stats["num_documents_found"]
        stats["f1_score"] = 2 * (stats["recall"] * stats["precision"] / (stats["recall"] + stats["precision"]))

        # Out of all relevant documents retrieved, what is the % of them that occupies the top K positions
        stats["perc_of_relevants_in_top_positions"] = stats["num_relevant_in_top_positions"] / stats["num_relevant_found"] * stats["recall"]
        stats["perc_of_relevants_in_top_1"] = stats["num_relevant_in_top_1"] / stats["num_relevant_found"] * stats["recall"]
        stats["perc_of_relevants_in_top_2"] = stats["num_relevant_in_top_2"] / stats["num_relevant_found"] * stats["recall"]
        stats["perc_of_relevants_in_top_3"] = stats["num_relevant_in_top_3"] / stats["num_relevant_found"] * stats["recall"]
        stats["perc_of_relevants_in_top_5"] = stats["num_relevant_in_top_5"] / stats["num_relevant_found"] * stats["recall"]
        stats["perc_of_relevants_in_top_8"] = stats["num_relevant_in_top_8"] / stats["num_relevant_found"] * stats["recall"]
        stats["perc_of_relevants_in_top_10"] = stats["num_relevant_in_top_10"] / stats["num_relevant_found"] * stats["recall"]
        stats["perc_of_relevants_in_top_15"] = stats["num_relevant_in_top_15"] / stats["num_relevant_found"] * stats["recall"]

        perc_columns = [
            "perc_of_relevants_in_top_positions",
            "perc_of_relevants_in_top_1",
            "perc_of_relevants_in_top_2",
            "perc_of_relevants_in_top_3",
            "perc_of_relevants_in_top_5",
            "perc_of_relevants_in_top_8",
            "perc_of_relevants_in_top_10",
            "perc_of_relevants_in_top_15"
        ]

        # Percentage of top K positions that is filled with a relevant search result
        mean_columns = [
            "perc_of_top_positions_occupied_by_relevants",
            "perc_of_top_1_occupied_by_relevants",
            "perc_of_top_2_occupied_by_relevants",
            "perc_of_top_3_occupied_by_relevants",
            "perc_of_top_5_occupied_by_relevants",
            "perc_of_top_8_occupied_by_relevants",
            "perc_of_top_10_occupied_by_relevants",
            "perc_of_top_15_occupied_by_relevants"
        ]

        selected_columns = [
            "test_name",
            "search_engine",
            "recall",
            "precision",
            "f1_score"] + sum_columns + perc_columns + mean_columns

        mean_stats = self.df.groupby(["test_name", "search_engine"])[mean_columns].mean().reset_index()
        return pd.concat([stats, mean_stats[mean_columns]], axis=1)[selected_columns].reset_index()


