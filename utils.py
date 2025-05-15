"""
Fonctions utilitaires pour le projet
"""

import os
import base64
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO

def generate_chart(data, chart_type="bar"):
    """Générer un graphique à partir de données"""
    plt.figure(figsize=(10, 6))
    
    if chart_type == "bar":
        # Créer un graphique à barres
        if isinstance(data, dict):
            sns.barplot(x=list(data.keys()), y=list(data.values()))
        elif isinstance(data, pd.DataFrame):
            if 'name' in data.columns and 'percentage' in data.columns:
                ax = sns.barplot(x='name', y='percentage', data=data)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    elif chart_type == "pie":
        # Créer un graphique circulaire
        if isinstance(data, dict):
            plt.pie(list(data.values()), labels=list(data.keys()), autopct='%1.1f%%')
    
    plt.tight_layout()
    
    # Sauvegarder l'image en mémoire
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encoder l'image en base64
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return image_base64

def get_file_extension(file_path):
    """Obtenir l'extension d'un fichier"""
    return os.path.splitext(file_path)[1].lower()

def is_valid_resume_file(file_path):
    """Vérifier si le fichier est un CV valide (PDF ou DOCX)"""
    valid_extensions = ['.pdf', '.docx', '.doc']
    extension = get_file_extension(file_path)
    return extension in valid_extensions

def format_percentage(value):
    """Formater un pourcentage"""
    return f"{value:.2f}%"