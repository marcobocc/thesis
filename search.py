import argparse
import json

from SearchEngineConfig import SearchEngineConfig
from SearchEnginePrototype import SearchEnginePrototype
from SearchEngineURP import SearchEngineURP


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
parser.add_argument("-u", "--urp",  default=False, type=bool)  # If true, also provides results from URP for comparison
parser.add_argument("--similarity",
                    choices=[SearchEngineConfig.COSINE,
                             SearchEngineConfig.DOT_PRODUCT,
                             SearchEngineConfig.EUCLIDEAN],
                    type=str.upper,
                    default=SearchEngineConfig.EUCLIDEAN)

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

urp = SearchEngineURP()

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

    if (args.urp):
        print(bcolors.INFO + "-" * 100)
        try:
            URP_all_results_with_score = urp.search(query)
            URP_all_results = [r["document"] for r in URP_all_results_with_score]
            print("{}(URP found {} search results){}".format(bcolors.YELLOW, len(URP_all_results), bcolors.INFO))
            print("{}\n".format(json.dumps(URP_all_results, indent=4)))
        except Exception:
            print("{}(Could not fetch query from URP){}".format(bcolors.YELLOW, bcolors.INFO))
        print(bcolors.INFO + "-" * 100)

