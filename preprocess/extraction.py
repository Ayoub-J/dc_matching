# """
# Module pour extraire les informations des CV depuis différents formats (PDF, DOCX)
# """

# import re
# import json
# import os
# from pathlib import Path
# from datetime import datetime
# import uuid

# # PDF
# from pdfminer.high_level import extract_text

# # DOCX
# import docx

# # NLP
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords

# # Local imports
# from config import RESUME_SECTIONS, TECHNICAL_SKILLS, PROCESSED_DIR

# # Télécharger les ressources NLTK nécessaires
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)

# class ResumeParser:
#     """Classe pour analyser et extraire les informations d'un CV"""
    
#     def __init__(self):
#         self.stopwords = set(stopwords.words('english') + stopwords.words('french'))
        
#     def extract_text_from_pdf(self, pdf_path):
#         """Extraire le texte d'un fichier PDF"""
#         try:
#             text = extract_text(pdf_path)
#             return text
#         except Exception as e:
#             print(f"Erreur lors de l'extraction du texte du PDF {pdf_path}: {e}")
#             return ""
    
#     def extract_text_from_docx(self, docx_path):
#         """Extraire le texte d'un fichier DOCX"""
#         try:
#             doc = docx.Document(docx_path)
#             text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
#             return text
#         except Exception as e:
#             print(f"Erreur lors de l'extraction du texte du DOCX {docx_path}: {e}")
#             return ""
            
#     def extract_text(self, file_path):
#         """Extraire le texte d'un fichier (PDF ou DOCX)"""
#         file_extension = Path(file_path).suffix.lower()
        
#         if file_extension == '.pdf':
#             return self.extract_text_from_pdf(file_path)
#         elif file_extension in ['.docx', '.doc']:
#             return self.extract_text_from_docx(file_path)
#         else:
#             print(f"Format de fichier non supporté: {file_extension}")
#             return ""
    
#     def extract_name(self, text):
#         """Extraire le nom du candidat"""
#         # Prendre les 5 premières lignes et chercher un pattern de nom
#         lines = text.split('\n')[:5]
#         for line in lines:
#             # Un nom est généralement en début de CV, sur une ligne courte
#             line = line.strip()
#             if 3 < len(line) < 40 and not any(c.isdigit() for c in line):
#                 words = line.split()
#                 # Un nom complet a généralement entre 2 et 5 mots
#                 if 2 <= len(words) <= 5:
#                     return line
        
#         # Si aucun pattern clair n'est trouvé
#         return "Nom inconnu"
    
#     def extract_contact_info(self, text):
#         """Extraire les informations de contact (email, téléphone)"""
#         contact_info = {}
        
#         # Extraction de l'email
#         email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#         emails = re.findall(email_pattern, text)
#         if emails:
#             contact_info['email'] = emails[0]
            
#         # Extraction du téléphone (formats internationaux et français)
#         phone_patterns = [
#             r'(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}',
#             r'\b\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b'  # Format français
#         ]
        
#         for pattern in phone_patterns:
#             phones = re.findall(pattern, text)
#             if phones:
#                 # Nettoyer le numéro de téléphone
#                 phone = phones[0].replace(' ', '').replace('-', '').replace('.', '')
#                 contact_info['phone'] = phone
#                 break
                
#         # Extraction de l'adresse LinkedIn
#         linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/profile/|linkedin\.com/).+?(?=\s|$)'
#         linkedin = re.findall(linkedin_pattern, text.lower())
#         if linkedin:
#             contact_info['linkedin'] = linkedin[0]
            
#         return contact_info
        
#     def extract_education(self, text):
#         """Extraire les informations sur l'éducation"""
#         educations = []
        
#         # Identifier la section éducation
#         education_section = self._get_section(text, "education")
#         if not education_section:
#             return educations
            
#         # Découper par lignes et par phrases
#         lines = education_section.split('\n')
#         current_education = {}
        
#         for line in lines:
#             if not line.strip():
#                 continue
                
#             # Chercher des dates (années)
#             years = re.findall(r'(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2}|(?:\s*[-–]\s*)?(?:présent|present|aujourd\'hui|today|en cours|ongoing))?', line, re.IGNORECASE)
            
#             # Chercher des diplômes
#             degree_patterns = [
#                 r'\b(?:master|licence|bachelor|mba|phd|doctorat|diplôme|ingénieur|bac|bts|dut|m1|m2|l1|l2|l3)\b',
#                 r'\b(?:master|bachelor|mba|phd|engineer|degree)\b'
#             ]
            
#             has_degree = False
#             for pattern in degree_patterns:
#                 if re.search(pattern, line, re.IGNORECASE):
#                     has_degree = True
#                     break
            
#             # Si on trouve une nouvelle entrée d'éducation
#             if (years or has_degree) and (not current_education or len(current_education) > 0):
#                 if current_education:
#                     educations.append(current_education)
#                 current_education = {"description": line.strip()}
#                 if years:
#                     current_education["period"] = years[0]
#             elif current_education:
#                 # Ajouter à la description de l'éducation actuelle
#                 current_education["description"] = current_education.get("description", "") + " " + line.strip()
        
#         # Ajouter la dernière entrée d'éducation
#         if current_education:
#             educations.append(current_education)
            
#         return educations
        
#     def extract_experience(self, text):
#         """Extraire les expériences professionnelles"""
#         experiences = []
        
#         # Identifier la section expérience
#         experience_section = self._get_section(text, "experience")
#         if not experience_section:
#             return experiences
            
#         # Découper par lignes
#         lines = experience_section.split('\n')
#         current_experience = {}
        
#         for line in lines:
#             if not line.strip():
#                 continue
                
#             # Chercher des dates (années et mois)
#             years = re.findall(r'(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2}|(?:\s*[-–]\s*)?(?:présent|present|aujourd\'hui|today|en cours|ongoing))?', line, re.IGNORECASE)
            
#             # Chercher des indicateurs de poste/entreprise
#             position_indicators = re.search(r'(?:chez|at|pour|@|à)\s+\w+', line, re.IGNORECASE)
            
#             # Si on trouve une nouvelle entrée d'expérience
#             if (years or position_indicators) and (not current_experience or len(current_experience) > 0):
#                 if current_experience:
#                     experiences.append(current_experience)
#                 current_experience = {"description": line.strip()}
#                 if years:
#                     current_experience["period"] = years[0]
#             elif current_experience:
#                 # Ajouter à la description de l'expérience actuelle
#                 current_experience["description"] = current_experience.get("description", "") + " " + line.strip()
        
#         # Ajouter la dernière entrée d'expérience
#         if current_experience:
#             experiences.append(current_experience)
            
#         return experiences
        
#     def extract_skills(self, text):
#         """Extraire les compétences techniques et non techniques"""
#         skills = {"technical": [], "languages": [], "soft_skills": []}
        
#         # Identifier la section compétences
#         skills_section = self._get_section(text, "skills")
#         full_text = text.lower()
        
#         # Extraire les compétences techniques à partir de la liste prédéfinie
#         for skill in TECHNICAL_SKILLS:
#             if skill.lower() in full_text:
#                 skills["technical"].append(skill)
        
#         # Chercher des langues
#         language_pattern = r'\b(?:anglais|english|français|french|espagnol|spanish|allemand|german|italien|italian|chinois|chinese|russe|russian|arabe|arabic|portugais|portuguese|japonais|japanese)\b[\s:]*((?:débutant|beginner|intermédiaire|intermediate|avancé|advanced|courant|fluent|natif|native|bilingue|bilingual|c1|c2|b1|b2|a1|a2)?)'
#         languages = re.findall(language_pattern, full_text, re.IGNORECASE)
        
#         for lang_match in languages:
#             if lang_match:
#                 lang = lang_match[0] if isinstance(lang_match, tuple) else lang_match
#                 if lang not in skills["languages"]:
#                     skills["languages"].append(lang.strip())
                    
#         # Soft skills - liste basique à chercher
#         soft_skills_list = [
#             "communication", "team work", "travail d'équipe", "leadership", "problem solving", 
#             "résolution de problèmes", "adaptability", "adaptabilité", "time management", 
#             "gestion du temps", "creativity", "créativité", "critical thinking", "esprit critique",
#             "collaboration", "autonomie", "autonomy", "organisation", "organization"
#         ]
        
#         for soft_skill in soft_skills_list:
#             if soft_skill.lower() in full_text:
#                 skills["soft_skills"].append(soft_skill)
                
#         return skills
    
#     def _get_section(self, text, section_type):
#         """Extraire une section spécifique du CV"""
#         # Obtenir les mots-clés pour ce type de section
#         section_keywords = RESUME_SECTIONS.get(section_type, [])
#         if not section_keywords:
#             return ""
            
#         # Préparer le texte
#         text_lines = text.split('\n')
#         section_text = ""
#         current_section = None
        
#         # Parcourir le texte ligne par ligne
#         for i, line in enumerate(text_lines):
#             line_lower = line.lower()
            
#             # Vérifier si cette ligne est un en-tête de section
#             for keyword in section_keywords:
#                 if keyword.lower() in line_lower and (len(line) < 50 or keyword.lower() in line_lower.split()):
#                     current_section = section_type
#                     section_text = ""
#                     break
                    
#             # Vérifier si on entre dans une nouvelle section
#             if current_section == section_type:
#                 for other_section_type, keywords in RESUME_SECTIONS.items():
#                     if other_section_type != section_type:
#                         for keyword in keywords:
#                             if keyword.lower() in line_lower and (len(line) < 50 or keyword.lower() in line_lower.split()):
#                                 current_section = other_section_type
#                                 break
                
#             # Ajouter la ligne au texte de la section si on est dans la bonne section
#             if current_section == section_type:
#                 section_text += line + '\n'
                
#         return section_text.strip()
    
#     def extract_information(self, file_path):
#         """Extraire toutes les informations d'un CV"""
#         # Extraire le texte du fichier
#         text = self.extract_text(file_path)
#         if not text:
#             return None
            
#         # Générer un ID unique pour ce CV
#         resume_id = str(uuid.uuid4())
        
#         # Construire le dictionnaire des informations
#         resume_data = {
#             "id": resume_id,
#             "file_name": os.path.basename(file_path),
#             "file_path": str(file_path),
#             "name": self.extract_name(text),
#             "contact_info": self.extract_contact_info(text),
#             "education": self.extract_education(text),
#             "experience": self.extract_experience(text),
#             "skills": self.extract_skills(text),
#             "raw_text": text,
#             "processed_date": datetime.now().isoformat()
#         }
        
#         # Enregistrer les données extraites
#         output_path = PROCESSED_DIR / f"{resume_id}.json"
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(resume_data, f, ensure_ascii=False, indent=2)
            
#         return resume_data

import os
import base64
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import fitz  # PyMuPDF pour traiter les PDF

# Import correct pour la dernière version de Mistral
from mistralai import Mistral

# Configuration globale
DEFAULT_MODEL = "pixtral-12b-2409"
SUPPORTED_EXTENSIONS = ['.pdf', '.PDF']
DEFAULT_RETRY_DELAY = 10  # secondes
DEFAULT_REQUEST_PAUSE = 2  # secondes
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 10  # Réduit pour les PDF qui sont plus lourds
DEFAULT_DELAY_BETWEEN_BATCHES = 15  # secondes
MAX_IMAGES_PER_CV = 5  # Maximum de pages par CV à analyser

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('CVAnalyzer')

def get_api_key(env_path: Optional[str] = None) -> str:
    """
    Obtient la clé API Mistral de diverses sources.
    """
    # Essayer les variables d'environnement
    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key:
        return api_key
    
    # Charger depuis un fichier .env
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        return api_key
        
    # Essayer de lire directement le fichier .env si présent
    env_path = Path(".env")
    if env_path.exists():
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("MISTRAL_API_KEY="):
                        return line.split("=")[1].strip()
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier .env: {e}")
    
    return ""

def ensure_directory(directory: str) -> str:
    """
    S'assure que le répertoire existe, le crée si nécessaire.
    """
    path = Path(directory).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def load_prompt(path: str) -> str:
    """
    Charge le prompt depuis un fichier.
    """
    try:
        prompt_path = Path(path)
        if not prompt_path.exists():
            logger.warning(f"Fichier prompt non trouvé: {path}")
            return """Analyser ce CV et extraire les informations suivantes au format JSON structuré:
            1. Informations personnelles (nom, prénom, email, téléphone, adresse, etc.)
            2. Formation et éducation (diplômes, écoles, dates)
            3. Expérience professionnelle (entreprises, postes, dates, descriptions)
            4. Compétences techniques
            5. Langues
            6. Certifications
            7. Projets importants
            8. Centres d'intérêt

            Faire attention à la structure hiérarchique des données. Répondre uniquement avec un JSON valide."""
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Erreur lors du chargement du prompt: {e}")
        return """Analyser ce CV et extraire les informations suivantes au format JSON structuré:
            1. Informations personnelles (nom, prénom, email, téléphone, adresse, etc.)
            2. Formation et éducation (diplômes, écoles, dates)
            3. Expérience professionnelle (entreprises, postes, dates, descriptions)
            4. Compétences techniques
            5. Langues
            6. Certifications
            7. Projets importants
            8. Centres d'intérêt

            Faire attention à la structure hiérarchique des données. Répondre uniquement avec un JSON valide."""

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode une image en base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pdf_to_images(pdf_path: str, temp_dir: str) -> List[str]:
    """
    Convertit un PDF en une liste de chemins d'images temporaires.
    """
    image_paths = []
    try:
        pdf_document = fitz.open(pdf_path)
        base_filename = Path(pdf_path).stem
        
        # Limiter le nombre de pages à traiter
        num_pages = min(len(pdf_document), MAX_IMAGES_PER_CV)
        
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom x2 pour meilleure qualité
            
            # Chemin pour l'image temporaire
            img_path = os.path.join(temp_dir, f"{base_filename}_page_{page_num+1}.png")
            pix.save(img_path)
            image_paths.append(img_path)
            
        pdf_document.close()
        
        if len(image_paths) == 0:
            logger.warning(f"Aucune page convertie pour le PDF {pdf_path}")
            
        return image_paths
        
    except Exception as e:
        logger.error(f"Erreur lors de la conversion du PDF {pdf_path}: {e}")
        return []

def perform_analysis(image_path: str, filename: str, prompt_path: str, model: str, client, is_page: bool = False) -> Dict[str, Any]:
    """
    Exécute l'analyse d'une page de CV selon le format exact attendu par l'API Mistral.
    """
    try:
        # Encoder l'image en base64
        base64_image = encode_image_to_base64(image_path)
    
        # Charger le prompt depuis le fichier
        prompt = load_prompt(prompt_path)
    
        # Ajuster le prompt en fonction du type d'analyse
        if is_page:
            prompt_text = f"{prompt}\nCeci est une page d'un CV multi-pages. Le nom du fichier est: {filename}"
        else:
            prompt_text = f"{prompt}\nLe nom du fichier est: {filename}"
    
        # Créer la liste de messages dans le format exact attendu par l'API Mistral
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt_text
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    
    # Appeler l'API avec le format correct
    
        response = client.chat.complete(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # Extraire le contenu JSON
        # result = json.loads(response.choices[0].message.content)
        
        # Extraire le contenu JSON - corrigé pour gérer les erreurs
        content = response.choices[0].message.content
        logger.info(f" 🟢 Contenu brut reçu: {content}")
        try:
            result = json.loads(content)

            # Vérifier si le JSON est valide
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            logger.info(f"Analyse réussie pour {filename} ")
            logger.info(f"Contenu JSON: {result}")

        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON: {e}")
            logger.error(f"Contenu reçu: {content}")
            clean_content = extract_json_from_text(content)
            result = json.loads(clean_content)
            

    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API Mistral: {str(e)}")
        raise
    
    # Ajouter des métadonnées
    result["_metadata"] = {
        "filename": filename,
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "is_page": is_page
    }
    
    return result


def extract_json_from_text(text: str) -> str:
    """
    Extrait le JSON d'un texte brut.
    """
    
    # Chercher le début et la fin du JSON
    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        return json_str
    else:
        logger.error("Aucun JSON valide trouvé dans le texte.")
        raise ValueError("Aucun JSON valide trouvé dans le texte.")


def merge_cv_results(page_results: List[Dict[str, Any]], filename: str, model: str) -> Dict[str, Any]:
    """
    Fusionne les résultats de plusieurs pages d'un même CV.
    """
    if not page_results:
        return {"error": "Aucun résultat à fusionner"}
    
    # Si une seule page, retourner directement le résultat
    if len(page_results) == 1:
        result = page_results[0].copy()
        # Mettre à jour les métadonnées
        result["_metadata"] = {
            "filename": filename,
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "pages_analyzed": 1,
            "is_page": False
        }
        return result
    
    # Initialiser le résultat fusionné avec les sections standard d'un CV
    merged = {
        "informations_personnelles": {},
        "formation": [],
        "experience_professionnelle": [],
        "competences": [],
        "langues": [],
        "certifications": [],
        "projets": [],
        "centres_interet": [],
        "_metadata": {
            "filename": filename,
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "pages_analyzed": len(page_results),
            "is_page": False
        }
    }
    
    # Parcourir chaque résultat de page et fusionner les données
    for page_result in page_results:
        # Fusionner informations personnelles (prendre la plus complète)
        if "informations_personnelles" in page_result:
            info_perso = page_result.get("informations_personnelles", {})
            if len(info_perso) > len(merged["informations_personnelles"]):
                merged["informations_personnelles"] = info_perso
        
        # Fusionner les listes
        for key in ["formation", "experience_professionnelle", "competences", "langues", 
                    "certifications", "projets", "centres_interet"]:
            if key in page_result and isinstance(page_result[key], list):
                merged[key].extend(page_result[key])
        
        # Variantes possibles des noms de section
        alt_keys = {
            "education": "formation",
            "experiences": "experience_professionnelle",
            "skills": "competences",
            "languages": "langues",
            "certifications": "certifications",
            "projects": "projets",
            "interets": "centres_interet",
            "hobbies": "centres_interet"
        }
        
        # Traiter les variantes de noms
        for alt_key, std_key in alt_keys.items():
            if alt_key in page_result and alt_key != std_key:
                if isinstance(page_result[alt_key], list):
                    merged[std_key].extend(page_result[alt_key])
    
    # Dédupliquer les entrées dans les listes
    for key in ["formation", "experience_professionnelle", "competences", "langues", 
                "certifications", "projets", "centres_interet"]:
        if merged[key]:
            # Convertir chaque élément en chaîne JSON pour comparaison
            unique_items = []
            seen = set()
            for item in merged[key]:
                item_json = json.dumps(item, sort_keys=True)
                if item_json not in seen:
                    seen.add(item_json)
                    unique_items.append(item)
            merged[key] = unique_items
    
    return merged

def save_individual_cv_result(result: Dict[str, Any], filename: str, individual_dir: str) -> str:
    """
    Sauvegarde le résultat d'un CV individuel.
    """
    # Créer un nom de fichier pour la sortie individuelle
    base_filename = Path(filename).stem.split(" - page")[0]  # Enlever la partie "- page X" si présente
    output_filename = f"{base_filename}_analyse.json"
    
    final_path = Path(individual_dir) / output_filename
    
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    logger.info(f"💾 CV individuel sauvegardé: {final_path}")
    return str(final_path)

def analyze_cv(pdf_path: str, filename: str, prompt_path: str, model: str, client, temp_dir: str, 
               individual_dir: str, max_retries: int = DEFAULT_MAX_RETRIES) -> Dict[str, Any]:
    """
    Analyse un CV avec gestion des tentatives en cas d'erreur.
    """
    try:
        # Vérifier si ce CV a déjà été traité
        base_filename = Path(filename).stem
        output_filename = f"{base_filename}_analyse.json"
        final_path = Path(individual_dir) / output_filename
        
        if final_path.exists():
            logger.info(f"📄 CV déjà analysé: {filename}, chargement du résultat existant")
            with open(final_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Convertir le PDF en images
        image_paths = pdf_to_images(pdf_path, temp_dir)
    
        if not image_paths:
            raise ValueError(f"Impossible de convertir le PDF {filename} en images")
    
        # Analyser chaque page et fusionner les résultats
        all_page_results = []
    
        for idx, img_path in enumerate(image_paths):
            page_num = idx + 1
            logger.info(f"Analyse de la page {page_num}/{len(image_paths)} du CV {filename}...")
            
            for attempt in range(max_retries):
                try:
                    result = perform_analysis(
                        img_path, 
                        f"{filename} - page {page_num}", 
                        prompt_path, 
                        model, 
                        client, 
                        is_page=True
                    )
                    all_page_results.append(result)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"⏳ Limite atteinte (429). Attente {DEFAULT_RETRY_DELAY} secondes "
                                    f"avant nouvelle tentative... ({attempt + 1}/{max_retries})")
                        time.sleep(DEFAULT_RETRY_DELAY)
                    else:
                        logger.error(f"Erreur d'analyse pour {filename} - page {page_num}: {e}")
                        raise
            
            # Pause entre les requêtes
            time.sleep(DEFAULT_REQUEST_PAUSE)
        
        # Fusionner les résultats de toutes les pages
        merged_result = merge_cv_results(all_page_results, filename, model)
        
        standardized_result = standardize_cv_result(merged_result)  # Appliquer la standardisation

        # Nettoyer les fichiers temporaires
        for img_path in image_paths:
            try:
                os.remove(img_path)
            except Exception as e:
                logger.warning(f"Impossible de supprimer le fichier temporaire {img_path}: {e}")
        
        # Sauvegarder le CV individuel
        save_individual_cv_result(standardized_result, filename, individual_dir)
        
        return standardized_result
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du CV {filename}: {e}")
        return {
            "error": str(e),
            "_metadata": {
                "filename": filename,
                "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": True
            }
        }

def process_single_cv(pdf_path: str, prompt_path: str = "cv_prompt.txt", 
                     model: str = DEFAULT_MODEL, output_dir: str = "output", 
                     env_path: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Traite un seul CV PDF et sauvegarde le résultat.
    """
    # Initialisation
    api_key = api_key or get_api_key(env_path)
    if not api_key:
        raise ValueError("La clé API Mistral est manquante.")
    
    # Utiliser MistralClient avec la nouvelle API
    client = Mistral(api_key=api_key)
    
    # Création des dossiers nécessaires
    output_dir = ensure_directory(output_dir)
    temp_dir = ensure_directory(os.path.join(output_dir, "temp"))
    individual_dir = ensure_directory(os.path.join(output_dir, "cv_individuels"))
    
    # Vérification du chemin du PDF
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Le PDF {pdf_path} n'existe pas")
    
    filename = pdf_path.name
    logger.info(f"Traitement du CV unique: {filename}")
    
    # Analyser et sauvegarder
    result = analyze_cv(
        str(pdf_path), 
        filename, 
        prompt_path, 
        model, 
        client, 
        temp_dir, 
        individual_dir
    )
    
    return result


def find_pdf_files(directory: str) -> List[Path]:
    """
    Trouver tous les fichiers PDF dans un répertoire sans duplication.
    """
    pdf_files = set()
    for ext in ['.pdf', '.PDF']:
        for file_path in Path(directory).glob(f"*{ext}"):
            pdf_files.add(file_path)
    return sorted(list(pdf_files))


def batch_process(cvs_dir: str = "cvs", prompt_path: str = "cv_prompt.txt", 
                 model: str = DEFAULT_MODEL, output_dir: str = "output", 
                 env_path: Optional[str] = None, api_key: Optional[str] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Traite un lot de CV PDF.
    """
    # Initialisation
    api_key = api_key or get_api_key(env_path)
    if not api_key:
        raise ValueError("La clé API Mistral est manquante.")
    
    # Utiliser MistralClient
    client = Mistral(api_key=api_key)
    
    # Création des dossiers nécessaires
    cvs_dir = ensure_directory(cvs_dir)
    output_dir = ensure_directory(output_dir)
    temp_dir = ensure_directory(os.path.join(output_dir, "temp"))
    individual_dir = ensure_directory(os.path.join(output_dir, "cv_individuels"))
    
    logger.info(f"CVAnalyzer initialisé avec: model={model}, "
                f"cvs_dir={cvs_dir}, output_dir={output_dir}")
    
    # Trouver tous les fichiers avec les extensions supportées
    pdf_files = find_pdf_files(cvs_dir)
    # pdf_files = []
    # for ext in SUPPORTED_EXTENSIONS:
    #     pdf_files.extend(list(Path(cvs_dir).glob(f"*{ext}")))
    #     pdf_files.extend(list(Path(cvs_dir).glob(f"*{ext.upper()}")))
    
    # Trier les fichiers pour un traitement cohérent
    # pdf_files.sort()
    
    # Appliquer la limite si spécifiée
    # if limit and limit > 0:
    #     pdf_files = pdf_files[:limit]
    
    # total_files = len(pdf_files)
    # logger.info(f"🔍 Traitement de {total_files} CV dans le dossier `{cvs_dir}`...")
    
    # Le code en haut est remplacer par :
    pdf_files = find_pdf_files(cvs_dir)

    # Vérification des fichiers déjà traités pour éviter les doublons
    already_processed = set()
    for existing_file in Path(individual_dir).glob("*_analyse.json"):
        base_name = existing_file.stem.replace("_analyse", "")
        already_processed.add(base_name)
    
    # Filtrer les fichiers non traités
    pdf_files_to_process = []
    for pdf_path in pdf_files:
        base_name = pdf_path.stem
        if base_name not in already_processed:
            pdf_files_to_process.append(pdf_path)
        else:
            logger.info(f"✓ Fichier déjà traité, ignoré: {pdf_path.name}")
    
    # Appliquer la limite si spécifiée
    if limit and limit > 0:
        pdf_files_to_process = pdf_files_to_process[:limit]
    
    total_files = len(pdf_files_to_process)
    logger.info(f"🔍 Traitement de {total_files} CV dans le dossier `{cvs_dir}`...")

    results = []
    
    # Traitement par lots
    for batch_idx in range(0, len(pdf_files), DEFAULT_BATCH_SIZE):
        batch = pdf_files[batch_idx:batch_idx + DEFAULT_BATCH_SIZE]
        batch_num = batch_idx // DEFAULT_BATCH_SIZE + 1
        total_batches = (total_files + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE
        
        logger.info(f"🔄 Traitement du batch {batch_num}/{total_batches}...")
        
        for idx, pdf_path in enumerate(batch, start=batch_idx + 1):
            filename = pdf_path.name
            logger.info(f"📄 {idx}/{total_files}. Traitement de : {filename}...")
            
            try:
                # Analyser le CV
                result = analyze_cv(
                    str(pdf_path), 
                    filename, 
                    prompt_path, 
                    model, 
                    client, 
                    temp_dir, 
                    individual_dir
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"⚠️ Erreur sur {filename} : {e}")
        
        # Pause entre les lots
        if batch_num < total_batches:
            logger.info(f"⏳ Pause de {DEFAULT_DELAY_BETWEEN_BATCHES} secondes entre les batches...")
            time.sleep(DEFAULT_DELAY_BETWEEN_BATCHES)
    
    logger.info(f"✅ Traitement terminé. {len(results)} CV analysés avec succès.")
    return results


def standardize_cv_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardise le résultat d'analyse CV pour garantir une structure uniforme.
    """
    # Modèle standard conforme au format attendu
    standard_format = {
        "informations_personnelles": {
            "nom": "",
            "prenom": "",
            "email": ""
        },
        "formation_et_education": [],
        "experience_professionnelle": [],
        "competences_techniques": [],
        "langues": [],
        "certifications": [],
        "_metadata": {
            "filename": result.get("_metadata", {}).get("filename", ""),
            "analyzed_at": result.get("_metadata", {}).get("analyzed_at", ""),
            "model": result.get("_metadata", {}).get("model", ""),
            "pages_analyzed": result.get("_metadata", {}).get("pages_analyzed", 1),
            "is_page": result.get("_metadata", {}).get("is_page", False)
        }
    }
    
    # Normalisation des clés variantes pour les informations personnelles
    personal_keys = ["informations_personnelles", "personal_info", "personal_information"]
    for key in personal_keys:
        if key in result:
            # Copier les informations personnelles disponibles
            personal_info = result[key]
            standard_format["informations_personnelles"]["nom"] = personal_info.get("nom", personal_info.get("name", personal_info.get("last_name", "")))
            standard_format["informations_personnelles"]["prenom"] = personal_info.get("prenom", personal_info.get("first_name", ""))
            standard_format["informations_personnelles"]["email"] = personal_info.get("email", "")
            
            # Ajouter d'autres champs d'informations personnelles s'ils existent
            for field, value in personal_info.items():
                if field not in ["nom", "prenom", "email", "name", "first_name", "last_name"]:
                    standard_format["informations_personnelles"][field] = value
            break
    
    # Normalisation de l'éducation
    education_keys = ["formation_et_education", "education", "formation"]
    for key in education_keys:
        if key in result and isinstance(result[key], list):
            for item in result[key]:
                education_item = {
                    "diplome": item.get("diplome", item.get("degree", "")),
                    "ecole": item.get("ecole", item.get("institution", item.get("school", ""))),
                    "dates": item.get("dates", "")
                }
                standard_format["formation_et_education"].append(education_item)
            break
    
    # Normalisation de l'expérience professionnelle
    experience_keys = ["experience_professionnelle", "experience", "work_experience", "professional_experience"]
    for key in experience_keys:
        if key in result and isinstance(result[key], list):
            for item in result[key]:
                experience_item = {
                    "entreprise": item.get("entreprise", item.get("company", "")),
                    "poste": item.get("poste", item.get("position", item.get("role", item.get("title", "")))),
                    "dates": item.get("dates", ""),
                    "description": []
                }
                
                # Gestion des descriptions sous forme de liste ou de chaîne
                if "description" in item:
                    if isinstance(item["description"], list):
                        experience_item["description"] = item["description"]
                    elif isinstance(item["description"], str):
                        experience_item["description"] = [item["description"]]
                elif "responsibilities" in item and isinstance(item["responsibilities"], list):
                    experience_item["description"] = item["responsibilities"]
                elif "details" in item:
                    if isinstance(item["details"], list):
                        experience_item["description"] = item["details"]
                    elif isinstance(item["details"], str):
                        experience_item["description"] = [item["details"]]
                
                standard_format["experience_professionnelle"].append(experience_item)
            break
    
    # Normalisation des compétences techniques
    skills_keys = ["competences_techniques", "skills", "technical_skills"]
    for key in skills_keys:
        if key in result:
            if isinstance(result[key], list):
                standard_format["competences_techniques"] = result[key]
            elif isinstance(result[key], dict):
                # Si les compétences sont un dictionnaire, les aplatir en liste
                flat_skills = []
                for category, skills in result[key].items():
                    if isinstance(skills, list):
                        flat_skills.extend(skills)
                    else:
                        flat_skills.append(f"{category}: {skills}")
                standard_format["competences_techniques"] = flat_skills
            break
    
    # Normalisation des langues
    languages_keys = ["langues", "languages"]
    for key in languages_keys:
        if key in result:
            if isinstance(result[key], list):
                normalized_languages = []
                for item in result[key]:
                    if isinstance(item, dict):
                        normalized_languages.append(item)
                    elif isinstance(item, str):
                        # Essai de séparation du texte pour extraire la langue et le niveau
                        parts = item.split(" - ") if " - " in item else item.split(": ") if ": " in item else [item, ""]
                        normalized_languages.append({
                            "langue": parts[0].strip(),
                            "niveau": parts[1].strip() if len(parts) > 1 else ""
                        })
                standard_format["langues"] = normalized_languages
            break
    
    # Normalisation des certifications
    cert_keys = ["certifications", "certificates"]
    for key in cert_keys:
        if key in result:
            if isinstance(result[key], list):
                standard_format["certifications"] = []
                for item in result[key]:
                    if isinstance(item, dict) and "certificate" in item:
                        standard_format["certifications"].append(item["certificate"])
                    elif isinstance(item, str):
                        standard_format["certifications"].append(item)
            break
    
    return standard_format


if __name__ == "__main__":
    # Exemple d'utilisation 
    # Pour traiter un seul CV
    # process_single_cv("chemin/vers/cv.pdf")
    
    # Pour traiter tous les CV dans le dossier
    batch_process()
