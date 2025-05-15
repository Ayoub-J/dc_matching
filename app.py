"""
Application Streamlit pour le matching entre CV et offres d'emploi
"""

import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import base64
from io import BytesIO

# Modules locaux
from config import RESUMES_DIR, TOP_N_CANDIDATES
from preprocess.extraction import ResumeParser
from db.data_base import ResumeDatabase
from preprocess.vectorization import ResumeVectorizer
from preprocess.matching import ResumeMatcher
from utils import is_valid_resume_file, generate_chart, format_percentage

# Configuration de la page
st.set_page_config(
    page_title="Moteur de Matching CV-Emploi",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 4rem;
    }
    .highlight {
        background-color: #e6f7ff;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des sessions state
if 'parser' not in st.session_state:
    st.session_state.parser = ResumeParser()
    
if 'database' not in st.session_state:
    # Vérifier si MongoDB est disponible, sinon utiliser le stockage fichier
    use_mongodb = False  # Pour l'exemple, on utilise le stockage fichier
    st.session_state.database = ResumeDatabase(use_mongodb=use_mongodb)
    
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = ResumeVectorizer()
    
if 'matcher' not in st.session_state:
    st.session_state.matcher = ResumeMatcher(
        st.session_state.database,
        st.session_state.vectorizer
    )
    
if 'matching_results' not in st.session_state:
    st.session_state.matching_results = None

# Titre principal
st.markdown('<h1 class="main-header">Moteur Intelligent de Matching CV-Emploi</h1>', unsafe_allow_html=True)

# Barre latérale
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/find-matching-job.png", width=80)
    st.title("Paramètres")
    
    # Afficher le nombre de CV dans la base de données
    resume_count = st.session_state.database.get_resume_count()
    st.metric(label="Nombre de CV", value=resume_count)
    
    # Options
    top_n = st.slider("Nombre de candidats à afficher", min_value=1, max_value=20, value=TOP_N_CANDIDATES)
    
    # Bouton pour recharger les CV
    if st.button("Recharger les CV"):
        with st.spinner("Chargement des CV..."):
            success = st.session_state.matcher.load_resumes()
            if success:
                st.success(f"{resume_count} CV chargés avec succès")
            else:
                st.error("Erreur lors du chargement des CV")
    
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("""
    Ce système utilise l'algorithme BM25 pour mesurer la similarité sémantique entre les CV et les offres d'emploi.
    
    1. Importez vos CV (PDF/DOCX)
    2. Saisissez une offre d'emploi
    3. Trouvez les meilleurs candidats
    """)

# Onglets principaux
tab1, tab2, tab3 = st.tabs(["📥 Import de CV", "🔍 Matching", "📊 Statistiques"])

# Onglet 1: Import de CV
with tab1:
    st.markdown('<h2 class="sub-header">Importer des CV</h2>', unsafe_allow_html=True)
    
    # Zone de dépôt de fichiers
    uploaded_files = st.file_uploader(
        "Déposez vos fichiers CV (PDF ou DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} fichiers déposés. Cliquez sur 'Traiter les CV' pour continuer.")
        
        if st.button("Traiter les CV"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Traiter chaque fichier
            for i, file in enumerate(uploaded_files):
                # Mettre à jour la barre de progression
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Traitement du CV {i+1}/{len(uploaded_files)}: {file.name}")
                
                # Sauvegarder le fichier temporairement
                temp_file_path = Path(RESUMES_DIR) / file.name
                with open(temp_file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Extraire les informations du CV
                resume_data = st.session_state.parser.extract_information(temp_file_path)
                
                if resume_data:
                    # Sauvegarder dans la base de données
                    st.session_state.database.save_resume(resume_data)
                    
            # Recharger les CV pour le matching
            st.session_state.matcher.load_resumes()
            
            # Mise à jour du compteur de CV
            resume_count = st.session_state.database.get_resume_count()
            
            # Affichage final
            progress_bar.progress(1.0)
            status_text.text("")
            st.success(f"{len(uploaded_files)} CV traités avec succès. Total dans la base: {resume_count}")
    
    # Afficher les CV existants
    st.markdown('<h3 class="sub-header">CV dans la base de données</h3>', unsafe_allow_html=True)
    
    resumes = st.session_state.database.get_all_resumes()
    
    if resumes:
        # Créer un DataFrame pour l'affichage
        resume_data = []
        for resume in resumes:
            contact = resume.get('contact_info', {})
            email = contact.get('email', '-')
            
            # Compter les compétences
            skills = resume.get('skills', {})
            tech_skills_count = len(skills.get('technical', []))
            
            # Expérience professionnelle
            exp_count = len(resume.get('experience', []))
            
            resume_data.append({
                'ID': resume.get('id', ''),
                'Nom': resume.get('name', 'Inconnu'),
                'Email': email,
                'Compétences': tech_skills_count,
                'Expériences': exp_count
            })
        
        if resume_data:
            df = pd.DataFrame(resume_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Aucun CV importé pour le moment.")
    else:
        st.info("Aucun CV importé pour le moment.")

# Onglet 2: Matching
with tab2:
    st.markdown('<h2 class="sub-header">Matching CV - Offre d\'emploi</h2>', unsafe_allow_html=True)
    
    # Zone de texte pour l'offre d'emploi
    job_description = st.text_area(
        "Entrez le texte de l'offre d'emploi",
        height=300,
        placeholder="Copiez-collez ici le texte complet de l'offre d'emploi..."
    )
    
    # Options de matching
    col1, col2 = st.columns(2)
    with col1:
        match_button = st.button("Lancer le matching")
    with col2:
        min_score = st.slider("Score minimum (%)", 0, 100, 30)
    
    # Lancer le matching
    if match_button and job_description:
        with st.spinner("Recherche des candidats correspondants..."):
            # Lancer le matching
            results = st.session_state.matcher.match_job(job_description, top_n=top_n)
            
            # Filtrer par score minimum
            results = [r for r in results if r['percentage'] >= min_score]
            
            # Sauvegarder les résultats
            st.session_state.matching_results = results
            
    # Afficher les résultats
    if st.session_state.matching_results:
        results = st.session_state.matching_results
        
        st.markdown(f'<h3 class="sub-header">Top {len(results)} candidats</h3>', unsafe_allow_html=True)
        
        if results:
            # Création d'un graphique
            df_results = pd.DataFrame([
                {"name": r['name'], "percentage": r['percentage']} 
                for r in results
            ])
            
            # Afficher le graphique
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='name', y='percentage', data=df_results)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel('Score de correspondance (%)')
            ax.set_xlabel('Candidat')
            plt.title('Score de correspondance par candidat')
            plt.tight_layout()
            
            # Afficher le graphique dans Streamlit
            st.pyplot(plt)
            
            # Afficher les détails de chaque candidat
            for i, result in enumerate(results):
                with st.expander(f"{i+1}. {result['name']} - Score: {format_percentage(result['percentage'])}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**Contact:**")
                        contact = result.get('contact', {})
                        if contact.get('email'):
                            st.markdown(f"📧 Email: {contact['email']}")
                        if contact.get('phone'):
                            st.markdown(f"📱 Téléphone: {contact['phone']}")
                        if contact.get('linkedin'):
                            st.markdown(f"🔗 LinkedIn: {contact['linkedin']}")
                            
                    with col2:
                        st.markdown("**Compétences techniques:**")
                        
                        # Format original (dictionnaire avec clé 'technical')
                        if isinstance(result.get('skills'), dict):
                            tech_skills = result.get('skills', {}).get('technical', [])
                        # Format Mistral (liste directe)
                        else:
                            tech_skills = result.get('skills', [])
                        
                        if tech_skills:
                            for skill in tech_skills:
                                st.markdown(f"- {skill}")
                        else:
                            st.info("Aucune compétence technique extraite.")
                    
                    # Récupérer le CV complet
                    resume = st.session_state.database.get_resume_by_id(result['resume_id'])
                    if resume:
                        # Afficher l'expérience
                        st.markdown("**Expérience professionnelle:**")
                        for exp in resume.get('experience', []):
                            if isinstance(exp, dict):
                                period = exp.get('period', '')
                                desc = exp.get('description', '')
                                if period and desc:
                                    st.markdown(f"- **{period}**: {desc}")
                                elif desc:
                                    st.markdown(f"- {desc}")
                        
                        # Afficher l'éducation
                        st.markdown("**Formation:**")
                        for edu in resume.get('education', []):
                            if isinstance(edu, dict):
                                period = edu.get('period', '')
                                desc = edu.get('description', '')
                                if period and desc:
                                    st.markdown(f"- **{period}**: {desc}")
                                elif desc:
                                    st.markdown(f"- {desc}")
        else:
            st.warning("Aucun candidat ne correspond aux critères de recherche.")
    
    # Si aucun résultat n'est disponible
    if not job_description:
        st.info("Entrez le texte d'une offre d'emploi pour trouver les candidats correspondants.")

# Onglet 3: Statistiques
with tab3:
    st.markdown('<h2 class="sub-header">Statistiques de la base de données</h2>', unsafe_allow_html=True)
    
    # Récupérer les CV
    resumes = st.session_state.database.get_all_resumes()
    
    if resumes:
        # Préparer les données pour les statistiques
        skills_count = {}
        education_levels = {}
        experience_years = {}
        
        for resume in resumes:
            # Compter les compétences
            skills = resume.get('skills', {})
            for skill in skills.get('technical', []):
                skills_count[skill] = skills_count.get(skill, 0) + 1
                
            # Analyser les formations
            for edu in resume.get('education', []):
                if isinstance(edu, dict) and 'description' in edu:
                    desc = edu['description'].lower()
                    
                    # Détecter le niveau d'éducation
                    if any(term in desc for term in ['master', 'm2', 'bac+5']):
                        education_levels['Master/Bac+5'] = education_levels.get('Master/Bac+5', 0) + 1
                    elif any(term in desc for term in ['licence', 'bachelor', 'l3', 'bac+3']):
                        education_levels['Licence/Bac+3'] = education_levels.get('Licence/Bac+3', 0) + 1
                    elif any(term in desc for term in ['bts', 'dut', 'bac+2']):
                        education_levels['BTS/DUT/Bac+2'] = education_levels.get('BTS/DUT/Bac+2', 0) + 1
                    elif any(term in desc for term in ['ingénieur', 'engineer']):
                        education_levels['Ingénieur'] = education_levels.get('Ingénieur', 0) + 1
                    elif any(term in desc for term in ['doctorat', 'phd', 'doctorate']):
                        education_levels['Doctorat/PhD'] = education_levels.get('Doctorat/PhD', 0) + 1
                    else:
                        education_levels['Autre'] = education_levels.get('Autre', 0) + 1
        
        # Afficher les statistiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top compétences techniques")
            
            # Trier et limiter aux 15 compétences les plus fréquentes
            top_skills = dict(sorted(skills_count.items(), key=lambda x: x[1], reverse=True)[:15])
            
            # Créer un DataFrame pour le graphique
            df_skills = pd.DataFrame({
                'Compétence': list(top_skills.keys()),
                'Nombre': list(top_skills.values())
            })
            
            # Afficher le graphique
            plt.figure(figsize=(10, 8))
            ax = sns.barplot(x='Nombre', y='Compétence', data=df_skills, orient='h')
            plt.title('Top 15 des compétences techniques')
            plt.tight_layout()
            st.pyplot(plt)
            
        with col2:
            st.markdown("### Répartition des niveaux d'éducation")
            
            if education_levels:
                # Créer un graphique circulaire
                plt.figure(figsize=(8, 8))
                plt.pie(
                    list(education_levels.values()), 
                    labels=list(education_levels.keys()), 
                    autopct='%1.1f%%',
                    startangle=90
                )
                plt.axis('equal')
                plt.title('Répartition des niveaux d\'éducation')
                st.pyplot(plt)
            else:
                st.info("Données d'éducation insuffisantes pour créer des statistiques.")
        
        # Métriques globales
        st.markdown("### Métriques globales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nombre de CV", len(resumes))
        
        with col2:
            avg_skills = sum(len(r.get('skills', {}).get('technical', [])) for r in resumes) / max(1, len(resumes))
            st.metric("Moy. compétences techniques", f"{avg_skills:.1f}")
            
        with col3:
            avg_exp = sum(len(r.get('experience', [])) for r in resumes) / max(1, len(resumes))
            st.metric("Moy. expériences pro.", f"{avg_exp:.1f}")
            
        with col4:
            avg_edu = sum(len(r.get('education', [])) for r in resumes) / max(1, len(resumes))
            st.metric("Moy. formations", f"{avg_edu:.1f}")
            
    else:
        st.info("Aucune donnée disponible pour afficher des statistiques. Importez des CV dans l'onglet 'Import de CV'.")
        
# Footer
st.markdown("""
<div class="footer">
    <p>Moteur intelligent de matching CV-Emploi basé sur la similarité sémantique</p>
    <p>Utilise l'algorithme BM25 et le traitement du langage naturel</p>
</div>
""", unsafe_allow_html=True)