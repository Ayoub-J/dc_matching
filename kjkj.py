import json
import re
import spacy

# Chargement de spaCy français
nlp = spacy.load("fr_core_news_sm")

# Nettoyage de base
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9àâçéèêëîïôûùüÿñæœ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenization sans lemmatisation
def preprocess_text_no_lemmatization(text):
    text = clean_text(text)
    doc = nlp(text)
    tokens = [
        token.text.lower() for token in doc
        if token.is_alpha and not token.is_stop and len(token.text) > 2
    ]
    return tokens

# Tokenization avec lemmatisation
def preprocess_text_with_lemmatization(text):
    text = clean_text(text)
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() for token in doc
        if token.is_alpha and not token.is_stop and len(token.lemma_) > 2
    ]
    return tokens

# Combinaison des champs du CV pour traitement
def get_combined_text(cv):
    fields = [
        cv.get("competences", []),
        cv.get("soft_skills", []),
        cv.get("experiences_professionnelles", []),
        [cv.get("formation", "")]
    ]
    return " ".join(
        item for sublist in fields for item in (sublist if isinstance(sublist, list) else [sublist])
    )

# Prétraitement complet avec les deux versions
def preprocess_cvs(json_path="cvs.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        cvs = json.load(f)

    for cv in cvs:
        combined_text = get_combined_text(cv)
        cv["tokens_lemmatized"] = preprocess_text_with_lemmatization(combined_text)
        cv["tokens_non_lemmatized"] = preprocess_text_no_lemmatization(combined_text)

    return cvs

# Traitement
cvs_processed = preprocess_cvs("cvs.json")

# Sauvegarde
with open("cvs_preprocessed.json", "w", encoding="utf-8") as f:
    json.dump(cvs_processed, f, indent=2, ensure_ascii=False)

# Affichage exemple
for cv in cvs_processed:
    print(f"\n{cv['nom_prenom']}")
    print(f"🔹 Avec lemmatisation     : {cv['tokens_lemmatized']}")
    print(f"🔸 Sans lemmatisation     : {cv['tokens_non_lemmatized']}")
