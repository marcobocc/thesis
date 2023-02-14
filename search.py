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
parser.add_argument("-c", "--cutoff",  default=0.0, type=float)
parser.add_argument("-s", "--stemming",  default=True, type=bool)
parser.add_argument("-r", "--remove_stopwords",  default=True, type=bool)
parser.add_argument("-x", "--expand_synonyms",  default=False, type=bool)
parser.add_argument("--similarity",
                    choices=[SearchEngineConfig.COSINE,
                             SearchEngineConfig.DOT_PRODUCT,
                             SearchEngineConfig.EUCLIDEAN],
                    type=str.upper,
                    default=SearchEngineConfig.COSINE)

args = parser.parse_args()
k = args.k
cutoff = args.cutoff
stemming = args.stemming
remove_stopwords = args.remove_stopwords
expand_synonyms = args.expand_synonyms
similarity = args.similarity

searchEngine = SearchEnginePrototype(SearchEngineConfig(
        config_name="Prototype_Search_Engine",
        expand_synonyms=expand_synonyms,
        remove_stopwords=remove_stopwords,
        cutoff=cutoff,
        similarity=similarity,
        use_stemming=stemming
    )
)

print("Initialized search engine with config:")
print(json.dumps(searchEngine.config.to_dict(), indent=4))

while True:
    user_input = input("{}ENTER QUERY:{} ".format(bcolors.GREEN, bcolors.INFO))
    if user_input == "":
        break

    query = user_input

    all_results_with_score = searchEngine.search(query)
    all_results = [r["document"] for r in all_results_with_score]

    if k and len(all_results) > k:
        print("{}(Showing the {} most relevant results out of {}){}".format(bcolors.YELLOW, k, len(all_results), bcolors.INFO))

    results = all_results if k is None else all_results[:min(k, len(all_results))]
    print("{}\n".format(json.dumps(results, indent=4)))
