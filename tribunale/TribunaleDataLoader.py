import json

class TribunaleDataLoader:
    def __init__(self):
        file = open('sections.json')
        json_data = json.load(file)
        file.close()
        self.documents = []
        for section in json_data:
            section_name = section["section"]
            section_documents = section["items"]
            for document in section_documents:
                doc_title = section_name + "/" + document["title"]
                doc_paragraphs = document["paragraphs"]
                doc_contents = ""
                for paragraph in doc_paragraphs:
                    paragraph_content = paragraph["content"]
                    doc_contents += paragraph_content + " "
                doc_entry = {
                    "identifier" : doc_title,
                    "contents" : doc_contents
                }
                self.documents.append(doc_entry)