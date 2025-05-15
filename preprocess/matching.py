"""
Correction du module matching.py pour résoudre l'erreur de division par zéro
"""

from rank_bm25 import BM25Okapi
import numpy as np
import os
from pathlib import Path
import json
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ResumeMatcher')

class ResumeMatcher:
    """Classe pour effectuer le matching entre CV et offres d'emploi"""
    
    def __init__(self, database, vectorizer):
        """Initialiser le matcher avec la base de données et le vectoriseur"""
        self.database = database
        self.vectorizer = vectorizer
        self.resume_corpus = []
        self.resume_ids = []
        self.resume_names = {}  # Dictionnaire pour stocker les noms correspondant aux IDs
        self.bm25 = None
        self.loaded = False
        
    def load_resumes(self):
        """Charger tous les CV pour le matching"""
        # Récupérer tous les CV
        resumes = self.database.get_all_resumes()
        
        if not resumes:
            logger.warning("Aucun CV trouvé dans la base de données.")
            return False
            
        # Vectoriser chaque CV
        self.resume_corpus = []
        self.resume_ids = []
        self.resume_names = {}
        
        for resume in resumes:
            # Déterminer l'ID du CV (format Mistral)
            if '_id' in resume:
                resume_id = resume['_id']
            elif '_metadata' in resume and 'filename' in resume['_metadata']:
                resume_id = resume['_metadata']['filename']
            else:
                # Générer un ID basé sur le contenu
                resume_id = f"resume_{len(self.resume_ids)}"
                
            # Extraire le nom du candidat (format Mistral)
            candidate_name = "Candidat inconnu"
            if 'informations_personnelles' in resume:
                info = resume['informations_personnelles']
                prenom = info.get('prenom', '')
                nom = info.get('nom', '')
                if prenom and nom:
                    candidate_name = f"{prenom} {nom}"
                elif prenom:
                    candidate_name = prenom
                elif nom:
                    candidate_name = nom
            
            # Vectoriser le CV
            try:
                vector_data = self.vectorizer.vectorize_resume(resume)
                
                # Vérifier que des tokens ont été générés
                if not vector_data['tokens']:
                    logger.warning(f"Aucun token généré pour le CV de {candidate_name}. Ajout de tokens par défaut.")
                    # Ajouter des tokens par défaut pour éviter les erreurs
                    vector_data['tokens'] = ['default_token']
                
                # Ajouter au corpus
                self.resume_corpus.append(vector_data['tokens'])
                self.resume_ids.append(resume_id)
                self.resume_names[resume_id] = candidate_name
                
                # Sauvegarder le vecteur dans la base de données
                self.database.save_resume_vector(resume_id, vector_data)
                logger.info(f"CV vectorisé: {candidate_name} (ID: {resume_id})")
            except Exception as e:
                logger.error(f"Erreur lors de la vectorisation du CV {candidate_name}: {e}")
                continue
            
        # Créer l'index BM25 si nous avons au moins un CV
        if self.resume_corpus:
            try:
                # S'assurer que tous les CV ont des tokens pour éviter l'erreur de division par zéro
                filtered_corpus = []
                filtered_ids = []
                filtered_names = {}
                
                for i, tokens in enumerate(self.resume_corpus):
                    if tokens:  # Vérifier que la liste de tokens n'est pas vide
                        filtered_corpus.append(tokens)
                        resume_id = self.resume_ids[i]
                        filtered_ids.append(resume_id)
                        filtered_names[resume_id] = self.resume_names.get(resume_id, "Candidat inconnu")
                
                if not filtered_corpus:
                    logger.error("Aucun CV avec des tokens valides pour le matching.")
                    return False
                    
                # Créer l'index BM25 avec le corpus filtré
                self.resume_corpus = filtered_corpus
                self.resume_ids = filtered_ids
                self.resume_names = filtered_names
                
                self.bm25 = BM25Okapi(self.resume_corpus)
                self.loaded = True
                logger.info(f"Modèle BM25 chargé avec {len(self.resume_corpus)} CV.")
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la création de l'index BM25: {e}")
                return False
        else:
            logger.error("Échec du chargement des CV pour le matching.")
            return False
            
    def match_job(self, job_description, top_n=10):
        """Rechercher les CV correspondant à une offre d'emploi"""
        if not self.loaded:
            success = self.load_resumes()
            if not success:
                return []
                
        # Vectoriser l'offre d'emploi
        job_vector = self.vectorizer.vectorize_job_description(job_description)
        job_tokens = job_vector['tokens']
        
        # Vérifier que l'offre d'emploi contient des tokens
        if not job_tokens:
            logger.warning("Aucun token généré pour l'offre d'emploi. Ajout de tokens par défaut.")
            job_tokens = ['default_token']
        
        try:
            # Calculer les scores BM25
            scores = self.bm25.get_scores(job_tokens)
            
            # Vérifier si des scores extrêmes (NaN, inf) sont présents et les remplacer
            scores = np.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=0.0)
            
            # Obtenir les meilleurs résultats
            if len(scores) > 0:
                top_indices = np.argsort(scores)[::-1][:top_n]
            else:
                logger.warning("Aucun score généré pour l'offre d'emploi.")
                return []
            
            # Préparer les résultats
            results = []
            for idx in top_indices:
                resume_id = self.resume_ids[idx]
                score = scores[idx]
                
                # Récupérer les détails du CV
                resume = self.database.get_resume_by_id(resume_id)
                
                if resume:
                    # Calculer le pourcentage de similarité (normaliser le score)
                    # Éviter la division par zéro en fixant un seuil minimum
                    max_score = max(15.0, max(scores) if len(scores) > 0 else 15.0)
                    normalized_score = min(score / max_score, 1.0)  # Limiter à 100%
                    percentage = normalized_score * 100
                    
                    # Extraire les informations pertinentes du CV (format Mistral)
                    candidate_name = self.resume_names.get(resume_id, "Candidat inconnu")
                    
                    contact_info = {}
                    skills = []
                    
                    if 'informations_personnelles' in resume:
                        info = resume['informations_personnelles']
                        contact_info = {
                            'email': info.get('email', ''),
                            'telephone': info.get('telephone', ''),
                            'adresse': info.get('adresse', '')
                        }
                    
                    if 'competences_techniques' in resume:
                        skills = resume['competences_techniques']
                    
                    results.append({
                        "resume_id": resume_id,
                        "name": candidate_name,
                        "score": score,
                        "percentage": percentage,
                        "contact": contact_info,
                        "skills": skills,
                        "resume_data": resume  # Ajouter le CV complet pour plus de détails si nécessaire
                    })
            
            return results
        except Exception as e:
            logger.error(f"Erreur lors du matching de l'offre d'emploi: {e}")
            return []