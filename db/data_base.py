"""
Module pour gérer la base de données des CV avec support du format d'extraction Mistral
"""

import json
import os
from pathlib import Path
from pymongo import MongoClient
from datetime import datetime
import logging

from config import DB_HOST, DB_PORT, DB_NAME, PROCESSED_DIR

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ResumeDatabase')

class ResumeDatabase:
    """Classe pour gérer la base de données des CV"""
    
    def __init__(self, use_mongodb=True):
        """Initialiser la connexion à la base de données"""
        self.use_mongodb = use_mongodb
        
        if use_mongodb:
            try:
                self.client = MongoClient(host=DB_HOST, port=DB_PORT)
                self.db = self.client[DB_NAME]
                self.resumes_collection = self.db.resumes
                self.vectors_collection = self.db.resume_vectors
                logger.info(f"Connecté à la base de données MongoDB: {DB_NAME}")
            except Exception as e:
                logger.error(f"Erreur de connexion à MongoDB: {e}")
                self.use_mongodb = False
                logger.info("Basculement vers le stockage fichier.")
    
    def save_resume(self, resume_data):
        """Sauvegarder les données d'un CV dans la base de données
        
        Args:
            resume_data: Dictionnaire JSON avec les informations du CV (format Mistral)
        """
        # Générer un ID si nécessaire
        if '_id' not in resume_data:
            resume_data['_id'] = resume_data.get('_metadata', {}).get('filename', str(datetime.now().timestamp()))
            
        resume_id = resume_data['_id']
            
        if self.use_mongodb:
            try:
                # Vérifier si ce CV existe déjà
                existing = self.resumes_collection.find_one({"_id": resume_id})
                
                if existing:
                    # Mettre à jour le CV existant
                    self.resumes_collection.update_one(
                        {"_id": resume_id},
                        {"$set": resume_data}
                    )
                    logger.info(f"CV mis à jour: {resume_id}")
                else:
                    # Ajouter un nouveau CV
                    resume_data["created_at"] = datetime.now()
                    resume_data["updated_at"] = datetime.now()
                    self.resumes_collection.insert_one(resume_data)
                    logger.info(f"CV ajouté: {resume_id}")
                
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du CV dans MongoDB: {e}")
                return False
        else:
            # Sauvegarde dans un fichier JSON
            try:
                # Utiliser le nom de fichier sans extension comme identifiant si disponible
                filename = resume_data.get('_metadata', {}).get('filename', '')
                if filename:
                    file_id = Path(filename).stem
                else:
                    file_id = resume_id
                    
                output_path = PROCESSED_DIR / f"{file_id}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(resume_data, f, ensure_ascii=False, indent=2)
                logger.info(f"CV sauvegardé dans le fichier: {output_path}")
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du CV dans un fichier: {e}")
                return False
    
    def save_resume_vector(self, resume_id, vector_data):
        """Sauvegarder les vecteurs d'un CV"""
        if self.use_mongodb:
            try:
                # Vérifier si ce vecteur existe déjà
                existing = self.vectors_collection.find_one({"resume_id": resume_id})
                
                if existing:
                    # Mettre à jour le vecteur existant
                    self.vectors_collection.update_one(
                        {"resume_id": resume_id},
                        {"$set": {
                            "vector_data": vector_data,
                            "updated_at": datetime.now()
                        }}
                    )
                else:
                    # Ajouter un nouveau vecteur
                    self.vectors_collection.insert_one({
                        "resume_id": resume_id,
                        "vector_data": vector_data,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now()
                    })
                
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du vecteur dans MongoDB: {e}")
                return False
        else:
            # Sauvegarde dans un fichier JSON
            try:
                # Traiter resume_id comme un nom de fichier s'il se termine par .pdf
                if resume_id.lower().endswith('.pdf'):
                    file_id = Path(resume_id).stem
                else:
                    file_id = resume_id
                    
                output_path = PROCESSED_DIR / f"{file_id}_vector.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(vector_data, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du vecteur dans un fichier: {e}")
                return False
    
    def get_all_resumes(self):
        """Récupérer tous les CV de la base de données avec le format Mistral"""
        if self.use_mongodb:
            try:
                return list(self.resumes_collection.find({}))
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des CV depuis MongoDB: {e}")
                return []
        else:
            # Récupération depuis les fichiers JSON
            resumes = []
            try:
                for file_path in PROCESSED_DIR.glob('*.json'):
                    if '_vector' not in file_path.name:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            resume_data = json.load(f)
                            resumes.append(resume_data)
                return resumes
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des CV depuis les fichiers: {e}")
                return []
    
    def get_resume_by_id(self, resume_id):
        """Récupérer un CV par son ID ou nom de fichier"""
        if self.use_mongodb:
            try:
                # Rechercher par ID ou par nom de fichier dans les métadonnées
                resume = self.resumes_collection.find_one({"_id": resume_id})
                if not resume:
                    resume = self.resumes_collection.find_one({"_metadata.filename": resume_id})
                return resume
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du CV {resume_id} depuis MongoDB: {e}")
                return None
        else:
            # Récupération depuis un fichier JSON
            try:
                # Essayer avec l'ID exact
                file_path = PROCESSED_DIR / f"{resume_id}.json"
                
                # Si le fichier n'existe pas, essayer en considérant que c'est un nom de fichier PDF
                if not file_path.exists() and resume_id.lower().endswith('.pdf'):
                    file_id = Path(resume_id).stem
                    file_path = PROCESSED_DIR / f"{file_id}.json"
                    
                # Si toujours pas trouvé, chercher parmi tous les fichiers
                if not file_path.exists():
                    for p in PROCESSED_DIR.glob('*.json'):
                        if '_vector' not in p.name:
                            try:
                                with open(p, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    if data.get('_metadata', {}).get('filename', '') == resume_id:
                                        file_path = p
                                        break
                            except:
                                continue
                
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return None
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du CV {resume_id} depuis le fichier: {e}")
                return None
    
    def get_all_resume_vectors(self):
        """Récupérer tous les vecteurs de CV"""
        if self.use_mongodb:
            try:
                return list(self.vectors_collection.find({}))
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des vecteurs depuis MongoDB: {e}")
                return []
        else:
            # Récupération depuis les fichiers JSON
            vectors = []
            try:
                for file_path in PROCESSED_DIR.glob('*_vector.json'):
                    resume_id = file_path.name.replace('_vector.json', '')
                    with open(file_path, 'r', encoding='utf-8') as f:
                        vector_data = json.load(f)
                        vectors.append({
                            "resume_id": resume_id,
                            "vector_data": vector_data
                        })
                return vectors
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des vecteurs depuis les fichiers: {e}")
                return []
                
    def get_resume_count(self):
        """Obtenir le nombre de CV dans la base de données"""
        if self.use_mongodb:
            try:
                return self.resumes_collection.count_documents({})
            except Exception as e:
                logger.error(f"Erreur lors du comptage des CV dans MongoDB: {e}")
                return 0
        else:
            # Comptage depuis les fichiers JSON
            try:
                count = 0
                for file_path in PROCESSED_DIR.glob('*.json'):
                    if '_vector' not in file_path.name:
                        count += 1
                return count
            except Exception as e:
                logger.error(f"Erreur lors du comptage des CV dans les fichiers: {e}")
                return 0
                
    def import_mistral_json(self, json_directory):
        """Importer des fichiers JSON générés par l'extraction Mistral
        
        Args:
            json_directory: Chemin vers le répertoire contenant les fichiers JSON des CV
        
        Returns:
            int: Nombre de CV importés avec succès
        """
        count = 0
        json_dir = Path(json_directory)
        
        if not json_dir.exists() or not json_dir.is_dir():
            logger.error(f"Le répertoire {json_directory} n'existe pas ou n'est pas un dossier.")
            return 0
            
        # Parcourir tous les fichiers JSON dans le répertoire
        for json_path in json_dir.glob('*.json'):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    resume_data = json.load(f)
                    
                # Vérifier si c'est bien un CV au format Mistral
                if '_metadata' in resume_data:
                    # Sauvegarder le CV
                    if self.save_resume(resume_data):
                        count += 1
                        logger.info(f"CV importé avec succès: {json_path.name}")
                    else:
                        logger.warning(f"Erreur lors de l'importation du CV: {json_path.name}")
                else:
                    logger.warning(f"Format de fichier incorrect (pas de métadonnées): {json_path.name}")
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'importation du fichier {json_path.name}: {e}")
                
        logger.info(f"{count} CV importés avec succès depuis {json_directory}")
        return count
    
    
    def count_all_resumes(self):
        """Récupérer tous les CV de la base de données"""
        if self.use_mongodb:
            try:
                return list(self.resumes_collection.find({}))
            except Exception as e:
                print(f"Erreur lors de la récupération des CV depuis MongoDB: {e}")
                return []
        else:
            # Récupération depuis les fichiers JSON
            resumes = []
            try:
                for file_path in PROCESSED_DIR.glob('*.json'):
                    if '_vector' not in file_path.name:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            resume_data = json.load(f)
                            resumes.append(resume_data)
                return resumes
            except Exception as e:
                print(f"Erreur lors de la récupération des CV depuis les fichiers: {e}")
                return []
    
    def get_resume_by_id(self, resume_id):
        """Récupérer un CV par son ID"""
        if self.use_mongodb:
            try:
                return self.resumes_collection.find_one({"id": resume_id})
            except Exception as e:
                print(f"Erreur lors de la récupération du CV {resume_id} depuis MongoDB: {e}")
                return None
        else:
            # Récupération depuis un fichier JSON
            try:
                file_path = PROCESSED_DIR / f"{resume_id}.json"
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return None
            except Exception as e:
                print(f"Erreur lors de la récupération du CV {resume_id} depuis le fichier: {e}")
                return None
    
    def get_all_resume_vectors(self):
        """Récupérer tous les vecteurs de CV"""
        if self.use_mongodb:
            try:
                return list(self.vectors_collection.find({}))
            except Exception as e:
                print(f"Erreur lors de la récupération des vecteurs depuis MongoDB: {e}")
                return []
        else:
            # Récupération depuis les fichiers JSON
            vectors = []
            try:
                for file_path in PROCESSED_DIR.glob('*_vector.json'):
                    resume_id = file_path.name.replace('_vector.json', '')
                    with open(file_path, 'r', encoding='utf-8') as f:
                        vector_data = json.load(f)
                        vectors.append({
                            "resume_id": resume_id,
                            "vector_data": vector_data
                        })
                return vectors
            except Exception as e:
                print(f"Erreur lors de la récupération des vecteurs depuis les fichiers: {e}")
                return []
                
    def get_resume_count(self):
        """Obtenir le nombre de CV dans la base de données"""
        if self.use_mongodb:
            try:
                return self.resumes_collection.count_documents({})
            except Exception as e:
                print(f"Erreur lors du comptage des CV dans MongoDB: {e}")
                return 0
        else:
            # Comptage depuis les fichiers JSON
            try:
                count = 0
                for file_path in PROCESSED_DIR.glob('*.json'):
                    if '_vector' not in file_path.name:
                        count += 1
                return count
            except Exception as e:
                print(f"Erreur lors du comptage des CV dans les fichiers: {e}")
                return 0
