import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('CVVectorizer')

# Essayer de charger le modèle français de spaCy, sinon utiliser uniquement NLTK
try:
    nlp = spacy.load("fr_core_news_md")
    USE_SPACY = True
    logger.info("Modèle spaCy français chargé avec succès")
except (OSError, ImportError):
    USE_SPACY = False
    logger.warning("Modèle spaCy français non disponible, utilisation de NLTK uniquement")

# Stop words en français
STOP_WORDS = set(stopwords.words('french'))
# Ajouter quelques mots spécifiques aux CV qu'on ne veut pas considérer
CV_STOP_WORDS = {
    "cv", "curriculum", "vitae", "resume", "lettre", "motivation",
    "compétence", "competence", "experience", "expérience", "formation",
    "téléphone", "telephone", "email", "mail", "adresse", "portable",
    "nom", "prenom", "prénom", "date", "naissance"
}
STOP_WORDS.update(CV_STOP_WORDS)

class CVVectorizer:
    """
    Classe pour tokeniser et vectoriser les CV au format JSON
    """
    def __init__(
        self, 
        input_dir: str = "output/cv_individuels", 
        output_dir: str = "output/vectorisation",
        n_components: int = 100,  # Dimension pour la réduction SVD
        min_df: float = 0.01,     # Fréquence minimale des termes
        max_df: float = 0.95      # Fréquence maximale des termes
    ):
        """
        Initialisation du vectoriseur
        
        Args:
            input_dir: Répertoire contenant les fichiers JSON des CV
            output_dir: Répertoire où enregistrer les vecteurs
            n_components: Nombre de composantes pour la réduction dimensionnelle SVD
            min_df: Fréquence minimale des termes (pour TF-IDF)
            max_df: Fréquence maximale des termes (pour TF-IDF)
        """
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.tokens_dir = Path(output_dir) / "tokens"
        self.vectors_dir = Path(output_dir) / "vectors"
        self.models_dir = Path(output_dir) / "models"
        
        # Création des répertoires de sortie
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration des vectoriseurs
        self.n_components = n_components
        self.min_df = min_df
        self.max_df = max_df
        
        # Modèles pour la vectorisation
        self.vectorizer = None
        self.svd = None
        
        logger.info(f"CVVectorizer initialisé: input_dir={self.input_dir}, output_dir={self.output_dir}")
    
    def normalize_text(self, text: str) -> str:
        """
        Normalise un texte (suppression des caractères spéciaux, mise en minuscule, etc.)
        """
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les caractères spéciaux et les chiffres
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenise un texte et supprime les stop words
        """
        text = self.normalize_text(text)
        
        if USE_SPACY:
            # Utiliser spaCy pour une meilleure tokenisation et lemmatisation
            doc = nlp(text)
            tokens = [token.lemma_ for token in doc 
                     if token.lemma_ not in STOP_WORDS 
                     and not token.is_punct
                     and not token.is_space
                     and len(token.text) > 2]
        else:
            # Fallback sur NLTK si spaCy n'est pas disponible
            tokens = word_tokenize(text, language='french')
            tokens = [token for token in tokens 
                     if token not in STOP_WORDS 
                     and len(token) > 2]
            
        return tokens
    
    def extract_cv_content(self, cv_data: Dict[str, Any]) -> str:
        """
        Extrait le contenu pertinent d'un CV en excluant les informations personnelles
        """
        sections = []
        
        # Formation
        if "formation_et_education" in cv_data:
            for item in cv_data["formation_et_education"]:
                formation_text = " ".join([
                    item.get("diplome", ""),
                    item.get("ecole", ""),
                    item.get("filière", "")
                ])
                sections.append(formation_text)
        
        # Expérience professionnelle
        if "experience_professionnelle" in cv_data:
            for item in cv_data["experience_professionnelle"]:
                exp_text = " ".join([
                    item.get("entreprise", ""),
                    item.get("poste", "")
                ])
                
                # Ajouter la description si disponible
                if "description" in item:
                    if isinstance(item["description"], list):
                        exp_text += " " + " ".join(item["description"])
                    else:
                        exp_text += " " + item["description"]
                
                sections.append(exp_text)
        
        # Compétences techniques
        if "competences_techniques" in cv_data:
            if isinstance(cv_data["competences_techniques"], list):
                sections.extend(cv_data["competences_techniques"])
            elif isinstance(cv_data["competences_techniques"], str):
                sections.append(cv_data["competences_techniques"])
        
        # Langues (sans le niveau car cela apporte peu d'information sémantique)
        if "langues" in cv_data:
            for item in cv_data["langues"]:
                if isinstance(item, dict):
                    sections.append(item.get("langue", ""))
                else:
                    sections.append(item)
        
        # Certifications
        if "certifications" in cv_data:
            for item in cv_data["certifications"]:
                if isinstance(item, dict) and "certificate" in item:
                    sections.append(item["certificate"])
                elif isinstance(item, str):
                    sections.append(item)
        
        # Joindre toutes les sections
        return " ".join(sections)
    
    def tokenize_cv(self, cv_path: Union[str, Path]) -> Tuple[str, List[str]]:
        """
        Tokenise un CV et retourne son ID et les tokens
        """
        cv_path = Path(cv_path)
        
        try:
            with open(cv_path, "r", encoding="utf-8") as f:
                cv_data = json.load(f)
            
            # Extraire le contenu pertinent
            cv_content = self.extract_cv_content(cv_data)
            
            # Tokeniser le contenu
            tokens = self.tokenize_text(cv_content)
            
            # Créer un ID pour ce CV
            cv_id = cv_path.stem.replace('_analyse', '')
            
            return cv_id, tokens
            
        except Exception as e:
            logger.error(f"Erreur lors de la tokenisation du CV {cv_path}: {e}")
            return None, []
    
    def train_vectorizer(self, cv_tokens: Dict[str, List[str]]) -> None:
        """
        Entraîne un vectoriseur TF-IDF et un modèle SVD sur les tokens de CV
        """
        logger.info("Entraînement du vectoriseur TF-IDF...")
        
        # Préparer le corpus
        corpus_texts = []
        cv_ids = []
        
        for cv_id, tokens in cv_tokens.items():
            corpus_texts.append(" ".join(tokens))
            cv_ids.append(cv_id)
        
        # Créer et entraîner le vectoriseur TF-IDF
        self.vectorizer = TfidfVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=list(STOP_WORDS)
        )
        
        features = self.vectorizer.fit_transform(corpus_texts)
        
        # Appliquer SVD pour réduire la dimensionnalité 
        # (utile pour la recherche de similarité)
        logger.info(f"Réduction dimensionnelle avec SVD ({self.n_components} composantes)...")
        
        # Ajuster le nombre de composantes si nécessaire
        n_components = min(self.n_components, features.shape[0], features.shape[1])
        
        self.svd = TruncatedSVD(n_components=n_components)
        self.svd.fit(features)
        
        # Sauvegarder les modèles
        with open(self.models_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
            
        with open(self.models_dir / "svd_model.pkl", "wb") as f:
            pickle.dump(self.svd, f)
        
        # Sauvegarder les informations d'entraînement
        with open(self.models_dir / "training_info.json", "w", encoding="utf-8") as f:
            json.dump({
                "cv_ids": cv_ids,
                "n_components": n_components,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "vocabulary_size": len(self.vectorizer.vocabulary_)
            }, f, indent=2)
        
        logger.info(f"Vectoriseur entraîné avec succès. Taille du vocabulaire: {len(self.vectorizer.vocabulary_)}")
    
    def vectorize_cv(self, cv_tokens: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Vectorise les CV tokenisés
        """
        logger.info("Vectorisation des CV...")
        
        # Vérifier si le vectoriseur est entraîné
        if self.vectorizer is None or self.svd is None:
            logger.info("Vectoriseur non entraîné, entraînement en cours...")
            self.train_vectorizer(cv_tokens)
        
        # Vectoriser chaque CV
        cv_vectors = {}
        for cv_id, tokens in cv_tokens.items():
            try:
                # Transformer en vecteur TF-IDF
                text = " ".join(tokens)
                tfidf_vector = self.vectorizer.transform([text])
                
                # Réduire la dimensionnalité avec SVD
                reduced_vector = self.svd.transform(tfidf_vector)[0]
                
                # Stocker le vecteur
                cv_vectors[cv_id] = reduced_vector
                
                # Sauvegarder le vecteur
                np.save(self.vectors_dir / f"{cv_id}_vector.npy", reduced_vector)
                
            except Exception as e:
                logger.error(f"Erreur lors de la vectorisation du CV {cv_id}: {e}")
        
        logger.info(f"Vectorisation terminée. {len(cv_vectors)} CV vectorisés.")
        return cv_vectors
    
    def process_all_cvs(self) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
        """
        Traite tous les CV dans le répertoire d'entrée
        """
        logger.info(f"Traitement des CV dans {self.input_dir}...")
        
        # Trouver tous les fichiers JSON
        json_files = list(self.input_dir.glob("*_analyse.json"))
        logger.info(f"Trouvé {len(json_files)} fichiers JSON à traiter")
        
        # Tokeniser les CV
        cv_tokens = {}
        for json_file in json_files:
            cv_id, tokens = self.tokenize_cv(json_file)
            if cv_id and tokens:
                cv_tokens[cv_id] = tokens
                
                # Sauvegarder les tokens
                with open(self.tokens_dir / f"{cv_id}_tokens.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "cv_id": cv_id,
                        "tokens": tokens,
                        "token_count": len(tokens)
                    }, f, indent=2)
        
        logger.info(f"Tokenisation terminée. {len(cv_tokens)} CV tokenisés.")
        
        # Entraîner le vectoriseur et vectoriser les CV
        cv_vectors = self.vectorize_cv(cv_tokens)
        
        # Sauvegarder un fichier récapitulatif
        summary_data = {
            "total_cvs_processed": len(cv_tokens),
            "total_cvs_vectorized": len(cv_vectors),
            "tokens_directory": str(self.tokens_dir),
            "vectors_directory": str(self.vectors_dir),
            "models_directory": str(self.models_dir),
            "cv_ids": list(cv_tokens.keys())
        }
        
        with open(self.output_dir / "processing_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Traitement terminé. Résumé sauvegardé dans {self.output_dir}/processing_summary.json")
        
        return cv_tokens, cv_vectors


# Fonction principale pour exécuter le processus
def vectorize_cv_dataset(
    input_dir: str = "output/cv_individuels", 
    output_dir: str = "output/vectorisation",
    n_components: int = 50
) -> None:
    """
    Tokenise et vectorise un ensemble de CV JSON
    
    Args:
        input_dir: Répertoire contenant les fichiers JSON des CV
        output_dir: Répertoire où enregistrer les vecteurs et tokens
        n_components: Nombre de composantes pour la réduction SVD
    """
    vectorizer = CVVectorizer(
        input_dir=input_dir,
        output_dir=output_dir,
        n_components=n_components
    )
    
    vectorizer.process_all_cvs()


if __name__ == "__main__":
    # Exécution avec les paramètres par défaut
    vectorize_cv_dataset()