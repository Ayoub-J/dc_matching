import json
import re
import spacy
from spacy.matcher import PhraseMatcher

# Charger spaCy
nlp = spacy.load("fr_core_news_sm")

# Exemples d'expressions techniques à préserver
technical_keywords = [
    "Power BI", "python", "sql", "excel", "tableau", "machine learning",
    "deep learning", "tensorflow", "keras", "pytorch", "scikit-learn",
    "nlp", "big data", "spark", "hadoop", "azure", "aws", "git"
]

# Créer un PhraseMatcher pour détecter ces expressions
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(phrase) for phrase in technical_keywords]
matcher.add("TECH_TERMS", patterns)

# Stopwords personnalisés à retirer (plus précis)
custom_stopwords = set(nlp.Defaults.stop_words)
custom_stopwords.update([
    "année", "mois", "stage", "projet", "travail", "expérience", "expériences",
    "chez", "en", "avec", "sur", "depuis", "pendant"
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9àâçéèêëîïôûùüÿñæœ\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def advanced_tokenize(text):
    text = clean_text(text)
    doc = nlp(text)

    # Détection des multi-word expressions
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    # Final token list
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and token.lemma_.lower() not in custom_stopwords
    ]
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
        cv["tokens"] = advanced_tokenize(combined_text)

    return cvs

# Traitement
cvs_processed = preprocess_cvs("cvs.json")

# Sauvegarde
with open("cvs_preprocessed.json", "w", encoding="utf-8") as f:
    json.dump(cvs_processed, f, indent=2, ensure_ascii=False)

# Affichage pour vérif
for cv in cvs_processed:
    print(f"{cv['nom_prenom']} → {cv['tokens']}")
