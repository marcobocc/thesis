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
        mean_stats = self.df.groupby(["test_name", "search_engine"]).mean()
        return mean_stats
