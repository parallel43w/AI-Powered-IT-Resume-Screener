import pandas
import pymupdf
import pytesseract
import re
import spacy
from spacy.matcher import Matcher

# load nlp model and skills dataset
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.from_disk("jz_skill_patterns.jsonl")

# Open CV and parse the data x

def is_selectable(pdf):
    for page in pdf:
        if page.get_text("text").strip():
            return True
    return False
pdf_path = "5.pdf"
pdf = pymupdf.open(pdf_path)
data = ""
for page in pdf:
    if is_selectable(pdf):
        data += page.get_text()
    else:
        img = page.get_pixmap().pil_image()
        data += pytesseract.image_to_string(img)


def extract_data(text):
# print all entities found by nlp
    doc = nlp(text)
    details = {
        'pages': 0,
        'names': '',
        'emails': '',
        'urls': '',
        'phone': '',
        'skills': '',
        'education': '',
        'job_description': '',
        'location': '',

    }

    tokens = [token.text for token in doc if not token.is_punct]
    print(tokens)
    for ent in doc.ents:
        print(f"Entity: {ent.text} | Label: {ent.label_}")

    details['pages'] = len(pdf)
    # details['names'] = extract_names(doc)
    details['emails'] = extract_matcher(doc=doc, pattern=[{"LIKE_EMAIL": True}], name="MAIL")
    details['urls'] = extract_matcher(doc=doc, pattern=[{"LIKE_URL": True}], name="URLS")
    
    # details['skills'] = extract_skills(doc)
    #details['education'] = extract_edu()
    #details['job_description'] = extract_desc()
    print(details)
    print(set([ent.text for ent in doc.ents if ent.label_ == "SKILL"]))
    print([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    print([ent.text for ent in doc.ents if ent.label_ == "LOC"])


def extract_matcher(doc, pattern, name):
    matcher = Matcher(nlp.vocab)
    matcher.add(name, [pattern])
    matches = matcher(doc)
    return " ".join([doc[start:end].text for id, start, end in matches])

# df = pandas.read_csv("Resume.csv")
# #parse csv training data
# df = df.loc[df["Category"]=="INFORMATION-TECHNOLOGY"]
# print(df)
# for row in df.values:
#     data = row[2]
#     parsed_data = BeautifulSoup(data)
#     print(parsed_data.find('div').text)

