"""
Configuration du projet de matching CV-Offres
"""

import os
from pathlib import Path

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESUMES_DIR = DATA_DIR / "resumes"
PROCESSED_DIR = DATA_DIR / "processed"

# Créer les dossiers s'ils n'existent pas
for directory in [DATA_DIR, RESUMES_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuration de la base de données
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "27017"))
DB_NAME = os.getenv("DB_NAME", "resume_matcher")

# Configuration du modèle de matching
TOP_N_CANDIDATES = 10
MINIMUM_SCORE_THRESHOLD = 0.3

# Liste des compétences techniques à rechercher
TECHNICAL_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby", "go", "rust",
    "react", "angular", "vue", "node.js", "django", "flask", "spring", "express", "fastapi",
    "docker", "kubernetes", "aws", "azure", "gcp", "sql", "mongodb", "postgresql", "mysql",
    "html", "css", "sass", "webpack", "git", "jenkins", "circleci", "github actions",
    "machine learning", "deep learning", "nlp", "computer vision", "data science", "ai",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "hadoop", "spark", "kafka",
    "elasticsearch", "redis", "tableau", "power bi", "excel", "r", "matlab"
]

# Configuration de l'extraction des CV
RESUME_SECTIONS = {
    "education": ["education", "formation", "academic", "études", "diplômes", "diplome", "qualifications"],
    "experience": ["experience", "work", "employment", "expériences", "expérience professionnelle", "parcours professionnel"],
    "skills": ["skills", "compétences", "competences", "technical", "technologies", "outils", "languages"],
    "projects": ["projects", "projets", "réalisations", "portfolio", "achievements"],
    "certifications": ["certifications", "certificates", "certifications", "accréditations", "formations"]
}
