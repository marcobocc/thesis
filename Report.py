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
        max_k = 15
        sum_columns = [
            "num_relevant_found",
            "num_documents_found",
            "actual_relevants"
        ]
        labels_num_relevants_at_top_k = [
            "num_relevants_at_top_{}".format(k+1)
            for k in range(max_k)
        ]
        sum_columns += labels_num_relevants_at_top_k

        self.df = self.df.reset_index()
        stats = self.df.groupby(["test_name", "search_engine"])[sum_columns].sum().reset_index()
        stats["count"] = self.df.groupby(["test_name", "search_engine"])["index"].transform("count")
        stats["recall"] = stats["num_relevant_found"] / stats["actual_relevants"]
        stats["precision"] = stats["num_relevant_found"] / stats["num_documents_found"]
        stats["f1_score"] = 2 * (stats["recall"] * stats["precision"] / (stats["recall"] + stats["precision"]))


        labels_precision_at_top_k = [
            "precision_at_top_{}".format(k+1)
            for k in range(max_k)
        ]

        mean_columns = labels_precision_at_top_k

        selected_columns = [
            "test_name",
            "search_engine",
            "recall",
            "precision",
            "f1_score"] + sum_columns + mean_columns

        mean_stats = self.df.groupby(["test_name", "search_engine"])[mean_columns].mean().reset_index()
        return pd.concat([stats, mean_stats[mean_columns]], axis=1)[selected_columns].reset_index()

Report("02-12-2023_17_09_16")