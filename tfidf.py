from cmath import exp
import re
import math
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

italian_stopwords = stopwords.words("italian")
stemmer = nltk.stem.snowball.ItalianStemmer()

contents1 = "Le scelte di istruzione, educazione e salute relative al minore devono essere sempre concordate dai genitori salvo che ci sia un affido super esclusivo; in caso di figlio divenuto maggiorenne tali scelte devono essere necessariamente concordate anche dal figlio con ambedue i genitori \
   L’assegno  di mantenimento periodico è destinato a coprire tutti i costi connessi alle esigenze ordinarie di vita del minore, devono ritenersi nello stesso incluse,  a titolo esemplificativo le seguenti spese: il vitto, la mensa scolastica, il concorso alle spese di casa (canone di locazione, utenze, consumi), l'abbigliamento ordinario inclusi i cambi di stagione, le spese di cancelleria scolastica ricorrenti nell’anno, i medicinali da banco. \
   Gli assegni familiari devono essere corrisposti al genitore collocatario (o affidatario) dei figli e rappresentano una voce aggiuntiva rispetto all’assegno di mantenimento, anche se erogati dal datore di lavoro dell’altro genitore, salvi diversi accordi fra le parti o diversa indicazione giudiziale."

contents2 = "La celebrazione del matrimonio deve essere preceduta dalla pubblicazione, che e' richiesta dagli sposi (o da persona da questi incaricata) all'ufficiale dello stato civile del Comune dove uno degli sposi ha la residenza. La pubblicazione e' disposta per almeno otto giorni, \
   a cura dell'ufficiale dello stato civile, nei Comuni di residenza degli sposi.Se il matrimonio non e' celebrato nei 180 giorni successivi, la pubblicazione perde efficacia e si considera come non avvenuta."


contents3 = "I genitori o gli esercenti la potesta' sui figli minori non possono assumere iniziative di natura patrimoniale per i minori se non a seguito di apposita istanza proposta al Giudice Tutelare competente (ovvero innanzi al Giudice ove il minore risiede)."

documents = [
   {
       "identifier" : "Document A",
       "contents" : contents1
   },
   {
       "identifier" : "Document B",
       "contents" : contents2
   },
   {
       "identifier" : "Document C",
       "contents" : contents3
   }
]

def build_index(documents):
    index = {}
    for document in documents:
        document_words = [word.lower() for word in re.findall(r'\w+', document["contents"])]
        for word in document_words:
            stemmed_word = stemmer.stem(word)
            if stemmed_word not in index:
                index[stemmed_word] = {}
            documentName = document["identifier"]
            if documentName not in index[stemmed_word]:
                index[stemmed_word][documentName] = {
                    "count" : 0,
                    "tf" : 0
                }
            index[stemmed_word][documentName]["count"] += 1
            index[stemmed_word][documentName]["tf"] = index[stemmed_word][documentName]["count"] / len(document_words)
    return index

def get_idf(index, term):
    stemmed_term = stemmer.stem(term)
    df = len(index[stemmed_term] if stemmed_term in index else 0.0)
    num_documents = len(documents)
    idf = math.log((num_documents + 1) / (float(df) + 1.0)) # Smooth TF-IDF to prevent division by 0
    return idf

def get_tf(index, term, documentIdentifier):
    stemmed_term = stemmer.stem(term)
    if stemmed_term not in index:
        return 0
    if documentIdentifier not in index[stemmed_term]:
        return 0
    return index[stemmed_term][documentIdentifier]["tf"]

def get_tfidf(index, term, documentIdentifier):
    return get_tf(index, term, documentIdentifier) * get_idf(index, term)

def find_documents(keywords, index):
   search_results = {}
   for word in keywords:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in index:
            for documentIdentifier in index[stemmed_word]:
                if documentIdentifier not in search_results:
                    search_results[documentIdentifier] = []
                search_results[documentIdentifier].append(word)
   return search_results

def rank_documents(search_results, index):
    ranked_search_results = {}
    for documentIdentifier in search_results:
        ranked_search_results[documentIdentifier] = []
        for keyword in search_results[documentIdentifier]:
            tfidf = get_tfidf(index, keyword, documentIdentifier)
            ranked_search_results[documentIdentifier].append({
                "keyword": keyword,
                "tfidf" : tfidf
            })
    return ranked_search_results

def sort_documents_by_rank(ranked_search_results):
    sorted_documents = []
    for documentIdentifier in ranked_search_results:
        keywords = ranked_search_results[documentIdentifier]
        tfidfs = []
        for entry in keywords:
            tfidfs.append(entry["tfidf"])
        final_score = 0
        for tfidf in tfidfs:
            final_score += math.pow(tfidf, 2)
        final_score = math.sqrt(final_score)
        sorted_documents.append({
                "document" : documentIdentifier,
                "score" : final_score
            })
    sorted_documents.sort(key=lambda document: document["score"], reverse=True)
    return sorted_documents

def expand_keywords(keywords):
   expanded_keywords = [keywords]
   for word in keywords:
        synonyms = wn.synonyms(word, lang="ita")
        expanded_keywords += synonyms
   return [item for sublist in expanded_keywords for item in sublist]

def remove_stopwords(keywords):
    keywords_no_stopwords = []
    for word in keywords:
        if word not in italian_stopwords:
            keywords_no_stopwords.append(word)
    return keywords_no_stopwords

index = build_index(documents)

query = "figli istruzione"
original_keywords = [word.lower() for word in re.findall(r'\w+', query)]
keywords_no_stopwords = remove_stopwords(original_keywords)
expanded_keywords = expand_keywords(keywords_no_stopwords)

search_results = find_documents(expanded_keywords, index)
ranked_search_results = rank_documents(search_results, index)
sorted_search_results = sort_documents_by_rank(ranked_search_results)

print("Search results (with rank):", ranked_search_results)
print("Search results (sorted by rank)", sorted_search_results)