import argparse
import json

from SearchEngineConfig import SearchEngineConfig
from SearchEnginePrototype import SearchEnginePrototype


class bcolors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    INFO = '\x1b[0m'


parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-k", default=None, type=int)

args = parser.parse_args()
k = args.k

searchEngine = SearchEnginePrototype(SearchEngineConfig(
        config_name="Prototype_Search_Engine",
        expand_synonyms=False,
        remove_stopwords=True,
        cutoff=0.0,
        similarity=SearchEngineConfig.EUCLIDEAN,
        use_stemming=True
    )
)

while True:
    user_input = input("{}ENTER QUERY:{} ".format(bcolors.GREEN, bcolors.INFO))
    if user_input == "":
        break

    query = user_input

    all_results_with_score = searchEngine.search(query)
    all_results = [r["document"] for r in all_results_with_score]

    results = all_results if k is None else all_results[:min(k, len(all_results))]
    print("{}\n".format(json.dumps(results, indent=4)))
