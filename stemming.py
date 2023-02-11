import re
import nltk
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
       for word in re.findall(r'\w+', document["contents"]):
           stemmed_word = stemmer.stem(word)
           if stemmed_word not in index:
               index[stemmed_word] = set()
           index[stemmed_word].add(document["identifier"])
   return index

def find_keywords_include_stopwords(query, index):
   result = {}
   for word in re.findall(r'\w+', query):
        stemmed_word = stemmer.stem(word)
        if stemmed_word in index:
            result[word] = index[stemmed_word]
        else:
            result[word] = {}
   return result

def find_keywords_remove_stopwords(query, index):
   result = {}
   for word in re.findall(r'\w+', query):
        if word not in italian_stopwords:
            stemmed_word = stemmer.stem(word)
            if stemmed_word in index:
                result[word] = index[stemmed_word]
            else:
                result[word] = {}
   return result

index = build_index(documents)

query = "il minore e i suoi genitori"

# Result
print("Include stopwords:")
print(find_keywords_include_stopwords(query, index))

print("Remove stopwords:")
print(find_keywords_remove_stopwords(query, index))