import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('CVMatcher')

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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
# Ajouter des mots spécifiques à ignorer
JOB_OFFER_STOP_WORDS = {
    "offre", "emploi", "poste", "candidat", "recherche", "cherche",
    "mission", "responsabilité", "profil", "qualification", "experience",
    "expérience", "formation", "nous", "vous", "entreprise", "société",
    "adresse", "mail", "email", "contact", "téléphone", "telephone"
}
STOP_WORDS.update(JOB_OFFER_STOP_WORDS)


class BM25Matcher:
    """
    Classe pour le matching entre CV et offres d'emploi utilisant l'algorithme BM25
    """
    def __init__(
        self,
        cv_vectors_dir: str = "output/vectorisation/vectors",
        cv_tokens_dir: str = "output/vectorisation/tokens",
        cv_json_dir: str = "output/cv_individuels",
        models_dir: str = "output/vectorisation/models",
        output_dir: str = "output/matching",
        k1: float = 1.5,  # Paramètre k1 pour BM25
        b: float = 0.75   # Paramètre b pour BM25
    ):
        """
        Initialisation du matcher BM25
        
        Args:
            cv_vectors_dir: Répertoire contenant les vecteurs des CV
            cv_tokens_dir: Répertoire contenant les tokens des CV
            cv_json_dir: Répertoire contenant les fichiers JSON des CV
            models_dir: Répertoire contenant les modèles (vectoriseur, SVD)
            output_dir: Répertoire pour sauvegarder les résultats de matching
            k1: Paramètre k1 pour l'algorithme BM25
            b: Paramètre b pour l'algorithme BM25
        """
        self.cv_vectors_dir = Path(cv_vectors_dir).resolve()
        self.cv_tokens_dir = Path(cv_tokens_dir).resolve()
        self.cv_json_dir = Path(cv_json_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        
        # Paramètres BM25
        self.k1 = k1
        self.b = b
        
        # Créer le répertoire de sortie s'il n'existe pas
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger les modèles de vectorisation
        self.vectorizer = None
        self.svd = None
        self.load_vectorization_models()
        
        # Dictionnaires pour stocker les données des CV
        self.cv_vectors = {}
        self.cv_tokens = {}
        self.cv_data = {}
        
        # Données pour BM25
        self.corpus = []
        self.doc_freqs = {}  # Fréquence des termes dans les documents (df)
        self.doc_lengths = {}  # Longueur des documents
        self.avg_doc_length = 0.0  # Longueur moyenne des documents
        self.N = 0  # Nombre total de documents
        self.idf = {}  # IDF pour chaque terme
        
        logger.info(f"BM25Matcher initialisé avec k1={k1}, b={b}")
    
    def load_vectorization_models(self) -> None:
        """
        Charge les modèles de vectorisation (TF-IDF et SVD)
        """
        try:
            with open(self.models_dir / "tfidf_vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            
            with open(self.models_dir / "svd_model.pkl", "rb") as f:
                self.svd = pickle.load(f)
            
            logger.info("Modèles de vectorisation chargés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles de vectorisation: {e}")
    
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
    
    def load_cv_data(self) -> None:
        """
        Charge les données des CV (vecteurs, tokens et JSON)
        """
        logger.info("Chargement des données des CV...")
        
        # Charger les vecteurs
        for vector_file in self.cv_vectors_dir.glob("*_vector.npy"):
            cv_id = vector_file.stem.replace("_vector", "")
            self.cv_vectors[cv_id] = np.load(vector_file)
        
        # Charger les tokens
        for token_file in self.cv_tokens_dir.glob("*_tokens.json"):
            cv_id = token_file.stem.replace("_tokens", "")
            with open(token_file, "r", encoding="utf-8") as f:
                token_data = json.load(f)
                self.cv_tokens[cv_id] = token_data["tokens"]
        
        # Charger les données JSON
        for json_file in self.cv_json_dir.glob("*_analyse.json"):
            cv_id = json_file.stem.replace("_analyse", "")
            with open(json_file, "r", encoding="utf-8") as f:
                self.cv_data[cv_id] = json.load(f)
        
        # Vérifier la cohérence des données
        cv_ids = set(self.cv_vectors.keys()) & set(self.cv_tokens.keys()) & set(self.cv_data.keys())
        
        # Ne garder que les CV ayant toutes les données
        cv_to_keep = list(cv_ids)
        self.cv_vectors = {cv_id: self.cv_vectors[cv_id] for cv_id in cv_to_keep}
        self.cv_tokens = {cv_id: self.cv_tokens[cv_id] for cv_id in cv_to_keep}
        self.cv_data = {cv_id: self.cv_data[cv_id] for cv_id in cv_to_keep}
        
        logger.info(f"Données chargées pour {len(cv_ids)} CV")
    
    def prepare_bm25_data(self) -> None:
        """
        Prépare les données pour le calcul BM25
        """
        logger.info("Préparation des données pour BM25...")
        
        self.corpus = list(self.cv_tokens.values())
        self.N = len(self.corpus)
        
        if self.N == 0:
            logger.warning("Aucun CV trouvé pour calculer BM25")
            return
        
        # Calculer les fréquences de document (df) pour chaque terme
        self.doc_freqs = {}
        self.doc_lengths = {}
        
        for doc_id, tokens in self.cv_tokens.items():
            self.doc_lengths[doc_id] = len(tokens)
            
            # Compter les termes uniques dans ce document
            unique_terms = set(tokens)
            for term in unique_terms:
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0
                self.doc_freqs[term] += 1
        
        # Calculer la longueur moyenne des documents
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.N
        
        # Calculer les scores IDF pour chaque terme
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        
        logger.info(f"Données BM25 préparées: {self.N} documents, {len(self.idf)} termes uniques")
    
    def read_job_offer(self, job_offer_path: str) -> str:
        """
        Lit une offre d'emploi à partir d'un fichier texte
        """
        try:
            with open(job_offer_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de l'offre d'emploi: {e}")
            return ""
    
    def calculate_bm25_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        Calcule les scores BM25 pour une requête par rapport aux CV
        """
        scores = {}
        
        for doc_id, doc_tokens in self.cv_tokens.items():
            score = 0.0
            doc_len = self.doc_lengths[doc_id]
            
            # Calculer tf pour chaque terme de la requête
            term_freqs = {}
            for token in doc_tokens:
                if token not in term_freqs:
                    term_freqs[token] = 0
                term_freqs[token] += 1
            
            # Calculer le score BM25 pour cette requête
            for query_term in query_tokens:
                if query_term in term_freqs:
                    tf = term_freqs[query_term]
                    idf_value = self.idf.get(query_term, 0)
                    
                    # Formule BM25
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                    score += idf_value * (numerator / denominator)
            
            scores[doc_id] = score
        
        return scores
    
    def calculate_vector_similarity(self, job_offer_text: str) -> Dict[str, float]:
        """
        Calcule la similarité entre l'offre d'emploi et les CV en utilisant les vecteurs
        """
        if self.vectorizer is None or self.svd is None:
            logger.warning("Modèles de vectorisation non chargés, impossible de calculer la similarité vectorielle")
            return {}
        
        try:
            # Vectoriser l'offre d'emploi
            tfidf_vector = self.vectorizer.transform([job_offer_text])
            job_vector = self.svd.transform(tfidf_vector)[0]
            
            # Calculer la similarité cosinus avec chaque CV
            similarities = {}
            for cv_id, cv_vector in self.cv_vectors.items():
                sim = cosine_similarity([job_vector], [cv_vector])[0][0]
                similarities[cv_id] = sim
            
            return similarities
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la similarité vectorielle: {e}")
            return {}
    
    def match_job_offer(self, job_offer_path: str, alpha: float = 1) -> List[Dict[str, Any]]:
        """
        Effectue le matching entre une offre d'emploi et les CV
        
        Args:
            job_offer_path: Chemin vers le fichier de l'offre d'emploi
            alpha: Coefficient pour la combinaison des scores (BM25 vs. vectoriel)
                   alpha = 1.0 => uniquement BM25, alpha = 0.0 => uniquement vectoriel
        
        Returns:
            Liste des CV matchés, triés du plus pertinent au moins pertinent
        """
        logger.info(f"Matching pour l'offre d'emploi: {job_offer_path}")
        
        # Charger les données nécessaires
        if not self.cv_tokens or not self.idf:
            self.load_cv_data()
            self.prepare_bm25_data()
        
        # Lire l'offre d'emploi
        job_offer_text = self.read_job_offer(job_offer_path)
        if not job_offer_text:
            logger.error("Offre d'emploi vide ou non trouvée")
            return []
        
        # Tokeniser l'offre d'emploi
        job_offer_tokens = self.tokenize_text(job_offer_text)
        
        # Calculer les scores BM25
        bm25_scores = self.calculate_bm25_scores(job_offer_tokens)
        
        # Calculer les similarités vectorielles
        vector_similarities = self.calculate_vector_similarity(job_offer_text)
        
        # Normalisation des scores
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        max_vector = max(vector_similarities.values()) if vector_similarities else 1.0
        
        # Combiner les scores (avec normalisation)
        combined_scores = {}
        for cv_id in self.cv_data.keys():
            bm25_score = bm25_scores.get(cv_id, 0) / max_bm25 if max_bm25 > 0 else 0
            vector_score = vector_similarities.get(cv_id, 0) / max_vector if max_vector > 0 else 0
            
            # Score combiné: alpha * BM25 + (1-alpha) * similitude vectorielle
            combined_scores[cv_id] = alpha * bm25_score + (1 - alpha) * vector_score
        
        # Trier les CV selon le score combiné
        sorted_cv_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Préparer les résultats
        results = []
        for cv_id in sorted_cv_ids:
            cv_info = self.cv_data[cv_id]["informations_personnelles"]
            results.append({
                "cv_id": cv_id,
                "nom": cv_info.get("nom", ""),
                "prenom": cv_info.get("prenom", ""),
                "email": cv_info.get("email", ""),
                "score": combined_scores[cv_id],
                "bm25_score": bm25_scores.get(cv_id, 0) / max_bm25 if max_bm25 > 0 else 0,
                "vector_score": vector_similarities.get(cv_id, 0) / max_vector if max_vector > 0 else 0
            })
        
        # Sauvegarder les résultats
        self._save_matching_results(job_offer_path, results)
        
        return results
    
    def _save_matching_results(self, job_offer_path: str, results: List[Dict[str, Any]]) -> None:
        """
        Sauvegarde les résultats du matching
        """
        job_offer_name = Path(job_offer_path).stem
        output_file = self.output_dir / f"matching_{job_offer_name}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "job_offer": job_offer_path,
                "matching_timestamp": logging.Formatter().converter(),
                "results": results
            }, f, indent=2)
        
        logger.info(f"Résultats de matching sauvegardés dans {output_file}")
    
    def explain_matching(self, job_offer_path: str, cv_id: str) -> Dict[str, Any]:
        """
        Explique pourquoi un CV a obtenu un certain score pour une offre d'emploi
        
        Args:
            job_offer_path: Chemin vers le fichier de l'offre d'emploi
            cv_id: Identifiant du CV à expliquer
        
        Returns:
            Explication du matching
        """
        # Lire l'offre d'emploi
        job_offer_text = self.read_job_offer(job_offer_path)
        if not job_offer_text:
            return {"error": "Offre d'emploi vide ou non trouvée"}
        
        # Tokeniser l'offre d'emploi
        job_offer_tokens = self.tokenize_text(job_offer_text)
        
        # Vérifier si le CV existe
        if cv_id not in self.cv_tokens:
            return {"error": f"CV {cv_id} non trouvé"}
        
        cv_tokens = self.cv_tokens[cv_id]
        
        # Trouver les termes communs entre l'offre et le CV
        common_terms = set(job_offer_tokens) & set(cv_tokens)
        
        # Calculer le score BM25 pour chaque terme commun
        term_scores = {}
        doc_len = self.doc_lengths[cv_id]
        
        # Calculer tf pour chaque terme du CV
        term_freqs = {}
        for token in cv_tokens:
            if token not in term_freqs:
                term_freqs[token] = 0
            term_freqs[token] += 1
        
        # Calculer le score pour chaque terme commun
        for term in common_terms:
            tf = term_freqs[term]
            idf_value = self.idf.get(term, 0)
            
            # Formule BM25 pour ce terme
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score = idf_value * (numerator / denominator)
            
            term_scores[term] = {
                "score": score,
                "tf": tf,
                "idf": idf_value
            }
        
        # Extraire les informations clés du CV
        cv_info = self.cv_data[cv_id]
        
        return {
            "cv_id": cv_id,
            "nom": cv_info["informations_personnelles"].get("nom", ""),
            "prenom": cv_info["informations_personnelles"].get("prenom", ""),
            "common_terms": len(common_terms),
            "top_matching_terms": sorted(term_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:10],
            "score_details": term_scores
        }


def match_cv_to_job_offer(
    job_offer_path: str,
    cv_vectors_dir: str = "output/vectorisation/vectors",
    cv_tokens_dir: str = "output/vectorisation/tokens",
    cv_json_dir: str = "output/cv_individuels",
    models_dir: str = "output/vectorisation/models",
    output_dir: str = "output/matching",
    alpha: float = 1, # Poids du score BM25 (1.0 = uniquement BM25, 0.0 = uniquement similitude vectorielle)
    top_n: int = 10     # Nombre de résultats à retourner
) -> List[Dict[str, Any]]:
    """
    Fonction principale pour matcher des CV avec une offre d'emploi
    
    Args:
        job_offer_path: Chemin vers le fichier de l'offre d'emploi
        cv_vectors_dir: Répertoire contenant les vecteurs des CV
        cv_tokens_dir: Répertoire contenant les tokens des CV
        cv_json_dir: Répertoire contenant les fichiers JSON des CV
        models_dir: Répertoire contenant les modèles de vectorisation
        output_dir: Répertoire pour sauvegarder les résultats
        alpha: Coefficient pour la combinaison des scores (BM25 vs. vectoriel)
        top_n: Nombre de résultats à retourner
    
    Returns:
        Liste des top_n CV les plus pertinents pour l'offre d'emploi
    """
    matcher = BM25Matcher(
        cv_vectors_dir=cv_vectors_dir,
        cv_tokens_dir=cv_tokens_dir,
        cv_json_dir=cv_json_dir,
        models_dir=models_dir,
        output_dir=output_dir
    )
    
    # Effectuer le matching
    results = matcher.match_job_offer(job_offer_path, alpha=alpha)
    
    # Retourner les top_n résultats
    return results[:top_n]


if __name__ == "__main__":
    # Exemple d'utilisation
    job_offer_path = "prompt_offre.txt"
    top_candidates = match_cv_to_job_offer(job_offer_path, top_n=5)
    
    print(f"Top 5 candidats pour l'offre d'emploi {job_offer_path}:")
    for i, candidate in enumerate(top_candidates, 1):
        print(f"{i}. {candidate['prenom']} {candidate['nom']} (Score: {candidate['score']:.4f})")