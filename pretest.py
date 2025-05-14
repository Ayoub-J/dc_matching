import json
import re
import spacy

nlp = spacy.load("fr_core_news_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9àâçéèêëîïôûùüÿñæœ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return tokens

def get_combined_text(cv):
    fields = [
        cv.get("competences", []),
        cv.get("soft_skills", []),
        cv.get("experiences_professionnelles", []),
        [cv.get("formation", "")]
    ]
    return " ".join([item for sublist in fields for item in (sublist if isinstance(sublist, list) else [sublist])])

def preprocess_cvs(json_path="cvs.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        cvs = json.load(f)

    for cv in cvs:
        combined_text = get_combined_text(cv)
        cv["tokens"] = preprocess_text(combined_text)

    return cvs

cvs_processed = preprocess_cvs("cvs.json")

with open("cvs_preprocessed.json", "w", encoding="utf-8") as f:
    json.dump(cvs_processed, f, indent=2, ensure_ascii=False)

for cv in cvs_processed:
    print(f"{cv['nom_prenom']} → tokens: {cv['tokens']}")
