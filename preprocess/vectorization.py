"""
Module de vectorisation simplifié pour le matching CV-Offres
Sans aucune dépendance externe complexe
"""

import re
import string
import json
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SimpleVectorizer')

class ResumeVectorizer:
    """Classe pour vectoriser les CV avec une approche simple et efficace"""
    
    def __init__(self):
        """Initialiser le vectoriseur"""
        # Définir les stopwords français et anglais courants
        self.stop_words = {
            # Français
            'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'a', 'à', 'au', 'aux',
            'ce', 'ces', 'cette', 'de', 'du', 'en', 'par', 'pour', 'sur', 'dans', 'avec',
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'mon', 'ton', 'son',
            'ma', 'ta', 'sa', 'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos',
            'leurs', 'que', 'qui', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi',
            
            # Anglais
            'the', 'a', 'an', 'and', 'or', 'to', 'in', 'on', 'by', 'for', 'with', 'of',
            'at', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
            'can', 'could', 'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Liste de mots spécifiques au domaine du recrutement à ne pas filtrer
        self.domain_specific_words = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node', 
            'django', 'flask', 'spring', 'express', 'sql', 'nosql', 'mongodb',
            'aws', 'azure', 'gcp', 'devops', 'ci', 'cd', 'agile', 'scrum', 'ia',
            'machine', 'learning', 'deep', 'tensorflow', 'pytorch', 'keras', 'pandas',
            'numpy', 'scikit', 'power', 'bi', 'tableau', 'data', 'science', 'analytics',
            'git', 'docker', 'kubernetes', 'jenkins', 'linux', 'unix', 'windows',
            'frontend', 'backend', 'fullstack', 'web', 'mobile', 'api', 'rest',
            'graphql', 'microservices', 'cloud', 'saas', 'paas', 'iaas'
        }
        
        # Mots à ne pas filtrer même s'ils sont dans les stopwords
        self.stop_words = self.stop_words - self.domain_specific_words
        
        # Expressions régulières compilées pour plus d'efficacité
        self.word_pattern = re.compile(r'\b[a-zA-Z0-9À-ÿ]+\b')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
    
    def simple_lemmatize(self, word):
        """Fonction de lemmatisation basique"""
        if len(word) <= 3:
            return word
            
        # Règles pour le français
        if word.endswith('s') and not word.endswith('ss'):
            return word[:-1]
        elif word.endswith('aux'):
            return word[:-3] + 'al'
        elif word.endswith('ent'):
            return word[:-3] + 'er'
        elif word.endswith('euses'):
            return word[:-5] + 'eur'
        elif word.endswith('euse'):
            return word[:-4] + 'eur'
        elif word.endswith('ives'):
            return word[:-4] + 'if'
        elif word.endswith('ive'):
            return word[:-3] + 'if'
            
        # Règles pour l'anglais
        elif word.endswith('ing') and len(word) > 5:
            return word[:-3]
        elif word.endswith('ed') and len(word) > 4:
            return word[:-2]
        elif word.endswith('ies'):
            return word[:-3] + 'y'
        elif word.endswith('es'):
            return word[:-2]
        elif word.endswith('s'):
            return word[:-1]
            
        return word
    
    def tokenize_text(self, text):
        """Tokeniser un texte en mots avec un nettoyage basique"""
        if not text:
            return []
            
        # Nettoyer la ponctuation et convertir en minuscules
        text = self.punctuation_pattern.sub(' ', text.lower())
        
        # Extraire les mots
        tokens = self.word_pattern.findall(text)
        
        # Appliquer la lemmatisation simple
        lemmas = [self.simple_lemmatize(token) for token in tokens]
        
        return lemmas
    
    def preprocess_text(self, text):
        """Prétraiter le texte pour la vectorisation"""
        if not text:
            return ""
        
        # Tokeniser le texte
        tokens = self.tokenize_text(text)
        
        # Supprimer les stopwords et les nombres
        filtered_tokens = [
            word for word in tokens 
            if (word not in self.stop_words and 
                not word.isdigit() and 
                len(word) > 2)
        ]
        
        # Si aucun token n'a été généré après filtrage, retourner quelques tokens du texte original
        if not filtered_tokens and text:
            logger.warning("Aucun token après filtrage, utilisation de tokens simples")
            # Prendre jusqu'à 5 mots du texte original
            simple_tokens = text.lower().split()[:5]
            return ' '.join(simple_tokens)
            
        return ' '.join(filtered_tokens)
    
    def extract_text_from_resume(self, resume_data):
        """Extraire le texte utile d'un CV en gérant différents formats possibles"""
        resume_text = []
        
        # IMPORTANT: IGNORER LES INFORMATIONS PERSONNELLES
        # Ne pas utiliser les informations personnelles pour éviter les biais
        
        # Ajouter les informations d'éducation
        for edu_key in ['formation_et_education', 'education', 'formation']:
            if edu_key in resume_data:
                for edu in resume_data.get(edu_key, []):
                    if isinstance(edu, dict):
                        education_text = []
                        for key, value in edu.items():
                            if value and isinstance(value, str):
                                education_text.append(value)
                        resume_text.append(" ".join(education_text))
        
        # Ajouter les expériences professionnelles
        for exp_key in ['experience_professionnelle', 'experience', 'experiences']:
            if exp_key in resume_data:
                for exp in resume_data.get(exp_key, []):
                    if isinstance(exp, dict):
                        # Extraire le poste et l'entreprise
                        for field in ['poste', 'position', 'title', 'role']:
                            if exp.get(field):
                                resume_text.append(exp[field])
                                break
                                
                        for field in ['entreprise', 'company', 'organization']:
                            if exp.get(field):
                                resume_text.append(exp[field])
                                break
                            
                        # Ajouter les descriptions
                        for desc_field in ['description', 'details', 'responsibilities']:
                            if desc_field in exp:
                                desc_value = exp[desc_field]
                                if isinstance(desc_value, list):
                                    for desc in desc_value:
                                        if desc:
                                            resume_text.append(desc)
                                elif isinstance(desc_value, str):
                                    resume_text.append(desc_value)
                            
        # Ajouter les compétences techniques
        for skills_key in ['competences_techniques', 'skills', 'technical_skills', 'competences']:
            if skills_key in resume_data:
                skills = resume_data[skills_key]
                if isinstance(skills, list):
                    for skill in skills:
                        if skill:
                            # Ajouter plusieurs fois pour augmenter le poids
                            resume_text.append(skill)
                            resume_text.append(skill)  # Duplication pour donner plus de poids
                elif isinstance(skills, dict):
                    for category, skill_list in skills.items():
                        if isinstance(skill_list, list):
                            for skill in skill_list:
                                if skill:
                                    resume_text.append(skill)
                                    resume_text.append(skill)
                        elif isinstance(skill_list, str):
                            resume_text.append(skill_list)
                            resume_text.append(skill_list)
                
        # Ajouter les langues
        for lang_key in ['langues', 'languages']:
            if lang_key in resume_data:
                langs = resume_data[lang_key]
                if isinstance(langs, list):
                    for lang in langs:
                        if isinstance(lang, dict):
                            langue = lang.get('langue', '') or lang.get('language', '')
                            niveau = lang.get('niveau', '') or lang.get('level', '')
                            if langue:
                                resume_text.append(langue)
                                if niveau:
                                    resume_text.append(f"{langue} {niveau}")
                        elif isinstance(lang, str):
                            resume_text.append(lang)
                            
        # Ajouter les certifications
        for cert_key in ['certifications', 'certificates']:
            if cert_key in resume_data:
                certs = resume_data[cert_key]
                if isinstance(certs, list):
                    for cert in certs:
                        if isinstance(cert, dict) and cert.get('name'):
                            resume_text.append(cert['name'])
                        elif isinstance(cert, str):
                            resume_text.append(cert)
        
        return resume_text
    
    def vectorize_resume(self, resume_data):
        """Convertir un CV en un vecteur de caractéristiques
        
        Args:
            resume_data: Dictionnaire JSON du CV standardisé
        """
        # Extraire le texte important du CV
        resume_text = self.extract_text_from_resume(resume_data)
                
        # Joindre tout le texte
        full_text = ' '.join(resume_text)
        
        if not full_text.strip():
            logger.warning(f"Aucun texte extrait du CV (ID: {resume_data.get('_id', 'inconnu')})")
            # Ajouter un texte de base pour éviter les erreurs
            full_text = "cv document profil competence experience formation"
        
        # Prétraiter le texte
        processed_text = self.preprocess_text(full_text)
        
        # Créer une représentation simplifiée pour BM25
        simple_tokens = processed_text.split()
        
        # Si aucun token n'a été généré, utiliser un token par défaut
        if not simple_tokens:
            logger.warning(f"Aucun token généré pour le CV. Ajout de tokens par défaut.")
            simple_tokens = ['profil', 'competence', 'experience']
        
        # Ajouter les compétences avec un poids plus important
        for skills_key in ['competences_techniques', 'skills', 'technical_skills', 'competences']:
            if skills_key in resume_data:
                skills = resume_data[skills_key]
                if isinstance(skills, list):
                    for skill in skills:
                        if skill:
                            # Prétraiter et ajouter avec un poids élevé
                            skill_text = self.preprocess_text(skill)
                            if skill_text:
                                skill_tokens = skill_text.split()
                                simple_tokens.extend(skill_tokens * 3)
            
        # Récupérer le nom du fichier
        filename = ""
        if '_metadata' in resume_data and 'filename' in resume_data['_metadata']:
            filename = resume_data['_metadata']['filename']
        elif 'file_name' in resume_data:
            filename = resume_data['file_name']
            
        return {
            "tokens": simple_tokens,
            "processed_text": processed_text,
            "original_text": full_text,
            "filename": filename
        }
    
    def vectorize_job_description(self, job_text):
        """Vectoriser une offre d'emploi"""
        # Prétraiter le texte de l'offre d'emploi
        processed_text = self.preprocess_text(job_text)
        
        # Créer une représentation simplifiée pour BM25
        simple_tokens = processed_text.split()
        
        # Si aucun token n'a été généré, utiliser des tokens génériques
        if not simple_tokens:
            logger.warning("Aucun token généré pour l'offre d'emploi. Ajout de tokens par défaut.")
            simple_tokens = ['emploi', 'poste', 'profil', 'competence', 'mission']
        
        return {
            "tokens": simple_tokens,
            "processed_text": processed_text,
            "original_text": job_text
        }
        
    def load_resume_from_json(self, json_path):
        """Charger un CV à partir d'un fichier JSON (format Mistral)"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                resume_data = json.load(f)
            return resume_data
        except Exception as e:
            logger.error(f"Erreur lors du chargement du CV JSON {json_path}: {e}")
            return None


# """
# Module pour tokeniser et vectoriser les CV en utilisant Spacy
# """

# import os
# import re
# import string
# import json
# from pathlib import Path
# import logging

# # Configuration du logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger('SpacyVectorizer')

# # Importer spacy
# try:
#     import spacy
#     # Charger un modèle léger qui prend en charge le français et l'anglais
#     # Si vous avez besoin de plus de langues, vous pouvez utiliser "xx_ent_wiki_sm"
#     try:
#         nlp = spacy.load("fr_core_news_sm")
#         logger.info("Modèle Spacy français chargé avec succès")
#     except OSError:
#         # Si le modèle n'est pas installé, essayer de télécharger
#         logger.info("Téléchargement du modèle Spacy français...")
#         os.system("python -m spacy download fr_core_news_sm")
#         try:
#             nlp = spacy.load("fr_core_news_sm")
#             logger.info("Modèle Spacy français chargé avec succès après téléchargement")
#         except Exception as e:
#             logger.error(f"Impossible de charger le modèle Spacy: {e}")
#             nlp = None
# except ImportError:
#     logger.error("Spacy n'est pas installé. Installation requise: pip install spacy")
#     nlp = None

# class ResumeVectorizer:
#     """Classe pour vectoriser les CV extraits avec le format standardisé Mistral"""
    
#     def __init__(self):
#         """Initialiser le vectoriseur avec Spacy"""
#         # Vérifier si Spacy est disponible
#         self.spacy_available = nlp is not None
#         if not self.spacy_available:
#             logger.warning("Spacy non disponible, utilisation du tokenizer de secours")
#         else:
#             logger.info("Spacy disponible, tokenization avancée activée")
        
#         # Définir les stopwords français et anglais courants
#         self.stop_words = {
#             # Français
#             'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'a', 'à', 'au', 'aux',
#             'ce', 'ces', 'cette', 'de', 'du', 'en', 'par', 'pour', 'sur', 'dans', 'avec',
#             'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'mon', 'ton', 'son',
#             'ma', 'ta', 'sa', 'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos',
#             'leurs', 'que', 'qui', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi',
            
#             # Anglais
#             'the', 'a', 'an', 'and', 'or', 'to', 'in', 'on', 'by', 'for', 'with', 'of',
#             'at', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
#             'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
#             'can', 'could', 'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
#             'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
#         }
        
#         # Liste de mots spécifiques au domaine du recrutement à ne pas filtrer
#         self.domain_specific_words = {
#             'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node', 
#             'django', 'flask', 'spring', 'express', 'sql', 'nosql', 'mongodb',
#             'aws', 'azure', 'gcp', 'devops', 'ci', 'cd', 'agile', 'scrum', 'ia',
#             'machine', 'learning', 'deep', 'tensorflow', 'pytorch', 'keras', 'pandas',
#             'numpy', 'scikit', 'power', 'bi', 'tableau', 'data', 'science', 'analytics'
#         }
        
#         # Mots à ne pas filtrer même s'ils sont dans les stopwords
#         self.stop_words = self.stop_words - self.domain_specific_words
        
#         # Pour le mode de secours (si spacy échoue)
#         self.tokenize_pattern = re.compile(r'\b\w+\b')
#         self.punctuation_pattern = re.compile(r'[^\w\s]')
    
#     def tokenize_text(self, text):
#         """Tokeniser le texte en utilisant Spacy ou un tokenizer de secours"""
#         if not text:
#             return []
            
#         # Utiliser Spacy si disponible
#         if self.spacy_available and nlp is not None:
#             try:
#                 # Analyser le texte avec Spacy
#                 doc = nlp(text)
                
#                 # Extraire les tokens (sans ponctuation et sans stopwords)
#                 tokens = [token.lemma_.lower() for token in doc 
#                          if not token.is_punct and not token.is_space and len(token.text) > 1]
                
#                 # Si des tokens ont été trouvés, les retourner
#                 if tokens:
#                     return tokens
#                 else:
#                     logger.warning("Aucun token généré par Spacy, utilisation du tokenizer de secours")
#             except Exception as e:
#                 logger.error(f"Erreur lors de la tokenisation avec Spacy: {e}")
#                 logger.info("Basculement sur le tokenizer de secours")
                
#         # Mode de secours
#         text = self.punctuation_pattern.sub(' ', text)
#         return self.tokenize_pattern.findall(text.lower())
    
#     def preprocess_text(self, text):
#         """Prétraiter le texte pour la vectorisation"""
#         # Vérifier que le texte n'est pas vide
#         if not text:
#             return ""
        
#         # Nettoyer le texte (enlever la ponctuation si pas Spacy)
#         if not self.spacy_available:
#             text = text.translate(str.maketrans('', '', string.punctuation))
        
#         # Tokeniser le texte
#         tokens = self.tokenize_text(text)
        
#         # Supprimer les stopwords et les nombres (si pas déjà fait par Spacy)
#         if not self.spacy_available:
#             filtered_tokens = [
#                 word for word in tokens 
#                 if (word not in self.stop_words and 
#                     not word.isdigit() and 
#                     len(word) > 2)
#             ]
#         else:
#             # Avec Spacy, on filtre juste les stopwords qui auraient pu passer
#             filtered_tokens = [
#                 word for word in tokens 
#                 if word not in self.stop_words and len(word) > 2
#             ]
        
#         # Si aucun token n'a été généré après filtrage, retourner quelques tokens du texte original
#         if not filtered_tokens and text:
#             logger.warning("Aucun token après filtrage, utilisation de tokens simples")
#             # Prendre jusqu'à 5 mots du texte original
#             simple_tokens = text.lower().split()[:5]
#             return ' '.join(simple_tokens)
            
#         return ' '.join(filtered_tokens)
    
#     def vectorize_resume(self, resume_data):
#         """Convertir un CV en un vecteur de caractéristiques
        
#         Args:
#             resume_data: Dictionnaire JSON du CV standardisé (format Mistral)
#         """
#         # Extraire le texte important du CV
#         resume_text = []
        
#         # IMPORTANT: IGNORER LES INFORMATIONS PERSONNELLES
#         # Ne pas utiliser les informations personnelles pour éviter les biais
        
#         # Traiter différentes structures de données possibles pour s'adapter aux formats
        
#         # Ajouter les informations d'éducation
#         for edu_key in ['formation_et_education', 'education', 'formation']:
#             if edu_key in resume_data:
#                 for edu in resume_data.get(edu_key, []):
#                     if isinstance(edu, dict):
#                         education_text = []
#                         for key, value in edu.items():
#                             if value and isinstance(value, str):
#                                 education_text.append(value)
#                         resume_text.append(" ".join(education_text))
        
#         # Ajouter les expériences professionnelles
#         for exp_key in ['experience_professionnelle', 'experience', 'experiences']:
#             if exp_key in resume_data:
#                 for exp in resume_data.get(exp_key, []):
#                     if isinstance(exp, dict):
#                         # Extraire le poste et l'entreprise
#                         for field in ['poste', 'position', 'title', 'role']:
#                             if exp.get(field):
#                                 resume_text.append(exp[field])
#                                 break
                                
#                         for field in ['entreprise', 'company', 'organization']:
#                             if exp.get(field):
#                                 resume_text.append(exp[field])
#                                 break
                            
#                         # Ajouter les descriptions
#                         for desc_field in ['description', 'details', 'responsibilities']:
#                             if desc_field in exp:
#                                 desc_value = exp[desc_field]
#                                 if isinstance(desc_value, list):
#                                     for desc in desc_value:
#                                         if desc:
#                                             resume_text.append(desc)
#                                 elif isinstance(desc_value, str):
#                                     resume_text.append(desc_value)
                            
#         # Ajouter les compétences techniques
#         for skills_key in ['competences_techniques', 'skills', 'technical_skills', 'competences']:
#             if skills_key in resume_data:
#                 skills = resume_data[skills_key]
#                 if isinstance(skills, list):
#                     for skill in skills:
#                         if skill:
#                             # Ajouter plusieurs fois pour augmenter le poids
#                             resume_text.append(skill)
#                             resume_text.append(skill)  # Duplication pour donner plus de poids
#                 elif isinstance(skills, dict):
#                     for category, skill_list in skills.items():
#                         if isinstance(skill_list, list):
#                             for skill in skill_list:
#                                 if skill:
#                                     resume_text.append(skill)
#                                     resume_text.append(skill)
#                         elif isinstance(skill_list, str):
#                             resume_text.append(skill_list)
#                             resume_text.append(skill_list)
                
#         # Ajouter les langues
#         for lang_key in ['langues', 'languages']:
#             if lang_key in resume_data:
#                 langs = resume_data[lang_key]
#                 if isinstance(langs, list):
#                     for lang in langs:
#                         if isinstance(lang, dict):
#                             langue = lang.get('langue', '') or lang.get('language', '')
#                             niveau = lang.get('niveau', '') or lang.get('level', '')
#                             if langue:
#                                 resume_text.append(langue)
#                                 if niveau:
#                                     resume_text.append(f"{langue} {niveau}")
#                         elif isinstance(lang, str):
#                             resume_text.append(lang)
                            
#         # Ajouter les certifications
#         for cert_key in ['certifications', 'certificates']:
#             if cert_key in resume_data:
#                 certs = resume_data[cert_key]
#                 if isinstance(certs, list):
#                     for cert in certs:
#                         if isinstance(cert, dict) and cert.get('name'):
#                             resume_text.append(cert['name'])
#                         elif isinstance(cert, str):
#                             resume_text.append(cert)
                
#         # Joindre tout le texte
#         full_text = ' '.join(resume_text)
        
#         if not full_text.strip():
#             logger.warning(f"Aucun texte extrait du CV (ID: {resume_data.get('_id', 'inconnu')})")
#             # Ajouter un texte de base pour éviter les erreurs
#             full_text = "cv document profil competence experience formation"
        
#         # Prétraiter le texte
#         processed_text = self.preprocess_text(full_text)
        
#         # Créer une représentation simplifiée pour BM25
#         simple_tokens = processed_text.split()
        
#         # Si aucun token n'a été généré, utiliser un token par défaut
#         if not simple_tokens:
#             logger.warning(f"Aucun token généré pour le CV. Ajout de tokens par défaut.")
#             simple_tokens = ['profil', 'competence', 'experience']
        
#         # Ajouter les compétences avec un poids plus important
#         for skills_key in ['competences_techniques', 'skills', 'technical_skills', 'competences']:
#             if skills_key in resume_data:
#                 skills = resume_data[skills_key]
#                 if isinstance(skills, list):
#                     for skill in skills:
#                         if skill:
#                             # Prétraiter et ajouter avec un poids élevé
#                             skill_text = self.preprocess_text(skill)
#                             if skill_text:
#                                 skill_tokens = skill_text.split()
#                                 simple_tokens.extend(skill_tokens * 3)
            
#         # Ajouter le nom du fichier original sans l'extension comme métadonnée
#         filename = ""
#         if '_metadata' in resume_data and 'filename' in resume_data['_metadata']:
#             filename = resume_data['_metadata']['filename']
#         elif 'file_name' in resume_data:
#             filename = resume_data['file_name']
            
#         return {
#             "tokens": simple_tokens,
#             "processed_text": processed_text,
#             "original_text": full_text,
#             "filename": filename
#         }
    
#     def vectorize_job_description(self, job_text):
#         """Vectoriser une offre d'emploi"""
#         # Prétraiter le texte de l'offre d'emploi
#         processed_text = self.preprocess_text(job_text)
        
#         # Créer une représentation simplifiée pour BM25
#         simple_tokens = processed_text.split()
        
#         # Si aucun token n'a été généré, utiliser des tokens génériques
#         if not simple_tokens:
#             logger.warning("Aucun token généré pour l'offre d'emploi. Ajout de tokens par défaut.")
#             simple_tokens = ['emploi', 'poste', 'profil', 'competence', 'mission']
        
#         return {
#             "tokens": simple_tokens,
#             "processed_text": processed_text,
#             "original_text": job_text
#         }
        
#     def load_resume_from_json(self, json_path):
#         """Charger un CV à partir d'un fichier JSON (format Mistral)"""
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 resume_data = json.load(f)
#             return resume_data
#         except Exception as e:
#             logger.error(f"Erreur lors du chargement du CV JSON {json_path}: {e}")
#             return None