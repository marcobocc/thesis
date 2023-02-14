import bs4
import requests
from SearchEngineConfig import SearchEngineConfig


class SearchEngineURP:
    def __init__(self):
        self.config = SearchEngineConfig(
            config_name="URP",
            use_stemming=None,
            remove_stopwords=None,
            expand_synonyms=None,
            cutoff=None,
            similarity=None,
        )

    def search(self, query):
        trimmed_query = query[:30].rsplit(" ", 1)[0] # Trim query to max 30 chars, without cutting words # noqa

        base_url = "https://urp-milano.giustizia.it/index.phtml?Id_VMenu=467&q=" # noqa
        url_query_params = trimmed_query.replace(" ", "+")
        url = base_url + url_query_params

        r = requests.get(url, timeout=2)
        soup = bs4.BeautifulSoup(r.content, 'html.parser')
        search_results_outer_div = soup.find("div", {"id": "cont_tag_search"})
        search_results_query = search_results_outer_div.find_all("p")[0].text[17:] # noqa

        search_results_inner_div = soup.find("div", {"id": "cont_links_search"}) # noqa
        all_search_results_html = search_results_inner_div.find_all("p")

        if trimmed_query.lower() != search_results_query.lower():
            raise Exception("Termine cercato: '{}' but query was '{}'".format(search_results_query.lower(), trimmed_query.lower())) # noqa

        inner_text = search_results_outer_div.find_all("p")[1].text
        if inner_text == "per favore cercare un termine di lunghezza superiore a 2 caratteri": # noqa
            raise Exception("The query was less than or equal to 2 characters")

        if (len(all_search_results_html) > 0 and (
            inner_text == "nessuna scheda trovata: si prega di cercare un termine differente e riprovare" or # noqa
            inner_text == "per favore cercare un termine di lunghezza superiore a 2 caratteri")): # noqa
            raise Exception("Error parsing search results (found search results but there was an error message)") # noqa

        if (len(all_search_results_html) == 0 and
                inner_text != "nessuna scheda trovata: si prega di cercare un termine differente e riprovare" and # noqa
                inner_text != "per favore cercare un termine di lunghezza superiore a 2 caratteri"): # noqa
            raise Exception("Error parsing search results (no search results found but no error message)") # noqa

        all_documents_found = [self._extract_search_result_titles(search_result_html) for search_result_html in all_search_results_html] # noqa
        documents_from_tribunale = [{
            "document": doc,
            "score": 1.0
        } for doc in all_documents_found if doc is not None]
        return documents_from_tribunale

    def _extract_search_result_titles(self, search_result_html):
        all_links = search_result_html.find_all("a", href=True)

        if len(all_links) == 0:
            raise Exception("Search provided results but HTML parsing did not find any link") # noqa
        elif len(all_links) >= 2:
            raise Exception("More than one link found for a search result")

        search_result_link = all_links[0]["href"]
        search_result_title = all_links[0].text

        is_link_from_tribunale = search_result_link.find('www.tribunale.milano.it') != -1 # noqa
        is_title_from_tribunale = search_result_title[0:9] == 'TRIBUNALE'

        if is_link_from_tribunale != is_title_from_tribunale:
            raise Exception("Search result title is {} but link points to another website {}".format(search_result_title, search_result_link)) # noqa

        if not is_title_from_tribunale:
            return None

        document_title = search_result_title[12:]
        return document_title