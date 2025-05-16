import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import tempfile
import time
from pathlib import Path
import base64
import shutil
import logging
from typing import List, Dict, Any, Optional, Union


from preprocess.extraction import process_single_cv, get_api_key
from preprocess.tokenization import vectorize_cv_dataset
from preprocess.matcher_module import match_cv_to_job_offer, BM25Matcher

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('StreamlitCV')

# Configurer les chemins des répertoires
DEFAULT_DIRS = {
    "cv_dir": "cvs",
    "output_dir": "output",
    "cv_json_dir": "output/cv_individuels",
    "vectorisation_dir": "output/vectorisation",
    "vectors_dir": "output/vectorisation/vectors",
    "tokens_dir": "output/vectorisation/tokens",
    "models_dir": "output/vectorisation/models",
    "matching_dir": "output/matching"
}

# Créer les répertoires s'ils n'existent pas
for dir_path in DEFAULT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Fonction pour créer ou mettre à jour le fichier de prompt d'offre d'emploi
def save_job_offer(job_offer_text: str, job_offer_path: str = "prompt_offre.txt") -> str:
    """
    Sauvegarde le texte de l'offre d'emploi dans un fichier
    """
    try:
        with open(job_offer_path, "w", encoding="utf-8") as f:
            f.write(job_offer_text)
        return job_offer_path
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'offre d'emploi: {e}")
        return ""

# Fonction pour traiter un CV nouvellement chargé
def process_new_cv(uploaded_file, api_key: str, model: str = "pixtral-12b-2409",
                  prompt_path: str = "cv_prompt.txt") -> Dict[str, Any]:
    """
    Traite un nouveau CV chargé via l'interface Streamlit
    """
    logger.info(f"Traitement du CV: {uploaded_file.name}")
    
    # Créer un fichier temporaire pour le CV chargé
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Copier le fichier dans le répertoire des CV
        cv_path = os.path.join(DEFAULT_DIRS["cv_dir"], uploaded_file.name)
        shutil.copy(tmp_path, cv_path)
        
        # Traiter le CV
        result = process_single_cv(
            pdf_path=cv_path,
            prompt_path=prompt_path,
            model=model,
            output_dir=DEFAULT_DIRS["output_dir"],
            api_key=api_key
        )
        
        # Vectoriser le CV traité
        vectorize_cv_dataset(
            input_dir=DEFAULT_DIRS["cv_json_dir"],
            output_dir=DEFAULT_DIRS["vectorisation_dir"]
        )
        
        return {
            "success": True,
            "message": f"CV {uploaded_file.name} traité avec succès",
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement du CV: {e}")
        return {
            "success": False,
            "message": f"Erreur: {str(e)}",
            "result": None
        }
    finally:
        # Supprimer le fichier temporaire
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Fonction pour traiter plusieurs CV en lot
def process_multiple_cvs(uploaded_files, api_key: str, model: str = "pixtral-12b-2409",
                         prompt_path: str = "cv_prompt.txt") -> List[Dict[str, Any]]:
    """
    Traite plusieurs CV chargés simultanément via l'interface Streamlit
    """
    results = []
    
    total_files = len(uploaded_files)
    
    # Créer la barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Traitement du CV {i+1}/{total_files}: {uploaded_file.name}")
        progress_bar.progress((i / total_files) * 100)
        
        result = process_new_cv(uploaded_file, api_key, model, prompt_path)
        results.append(result)
        
        # Petite pause pour éviter de surcharger l'API
        if i < total_files - 1:
            time.sleep(1)
    
    # Finaliser la progression
    progress_bar.progress(100)
    status_text.text(f"Traitement terminé: {total_files} CV traités")
    
    return results

# Fonction pour charger les CV déjà traités
def load_processed_cvs() -> List[Dict[str, Any]]:
    """
    Charge les données des CV déjà traités
    """
    cv_data = []
    
    try:
        json_dir = Path(DEFAULT_DIRS["cv_json_dir"])
        json_files = list(json_dir.glob("*_analyse.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Extraire les informations clés
                cv_id = json_file.stem.replace("_analyse", "")
                metadata = data.get("_metadata", {})
                info_perso = data.get("informations_personnelles", {})
                
                cv_data.append({
                    "cv_id": cv_id,
                    "nom": info_perso.get("nom", ""),
                    "prenom": info_perso.get("prenom", ""),
                    "email": info_perso.get("email", ""),
                    "filename": metadata.get("filename", ""),
                    "analyzed_at": metadata.get("analyzed_at", "")
                })
            
            except Exception as e:
                logger.error(f"Erreur lors du chargement du CV {json_file}: {e}")
        
        return cv_data
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des CV traités: {e}")
        return []

# Fonction pour afficher les détails d'un CV
def display_cv_details(cv_id: str) -> None:
    """
    Affiche les détails d'un CV spécifique
    """
    try:
        json_path = Path(DEFAULT_DIRS["cv_json_dir"]) / f"{cv_id}_analyse.json"
        
        if not json_path.exists():
            st.error(f"Le fichier {json_path} n'existe pas")
            return
        
        with open(json_path, "r", encoding="utf-8") as f:
            cv_data = json.load(f)
        
        # Afficher les informations personnelles
        st.subheader("Informations personnelles")
        info_perso = cv_data.get("informations_personnelles", {})
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Nom:** {info_perso.get('nom', 'Non spécifié')}")
            st.write(f"**Prénom:** {info_perso.get('prenom', 'Non spécifié')}")
        with col2:
            st.write(f"**Email:** {info_perso.get('email', 'Non spécifié')}")
            if "telephone" in info_perso:
                st.write(f"**Téléphone:** {info_perso['telephone']}")
        
        # Afficher la formation
        st.subheader("Formation")
        formations = cv_data.get("formation_et_education", [])
        if formations:
            for formation in formations:
                st.markdown(f"""
                **{formation.get('diplome', 'Diplôme non spécifié')}**  
                *{formation.get('ecole', 'École non spécifiée')}*  
                {formation.get('dates', '')}
                """)
        else:
            st.info("Aucune information de formation disponible")
        
        # Afficher l'expérience professionnelle
        st.subheader("Expérience professionnelle")
        experiences = cv_data.get("experience_professionnelle", [])
        if experiences:
            for exp in experiences:
                st.markdown(f"""
                **{exp.get('poste', 'Poste non spécifié')}** - {exp.get('entreprise', 'Entreprise non spécifiée')}  
                *{exp.get('dates', '')}*
                """)
                
                # Afficher la description si disponible
                if "description" in exp and exp["description"]:
                    if isinstance(exp["description"], list):
                        for desc in exp["description"]:
                            st.markdown(f"- {desc}")
                    else:
                        st.write(exp["description"])
        else:
            st.info("Aucune expérience professionnelle disponible")
        
        # Afficher les compétences
        st.subheader("Compétences techniques")
        competences = cv_data.get("competences_techniques", [])
        if competences:
            if isinstance(competences, list):
                # Afficher en colonnes si la liste est longue
                if len(competences) > 5:
                    cols = st.columns(3)
                    for i, comp in enumerate(competences):
                        cols[i % 3].markdown(f"- {comp}")
                else:
                    for comp in competences:
                        st.markdown(f"- {comp}")
            else:
                st.write(competences)
        else:
            st.info("Aucune compétence technique disponible")
        
        # Afficher les langues
        st.subheader("Langues")
        langues = cv_data.get("langues", [])
        if langues:
            for langue in langues:
                if isinstance(langue, dict):
                    st.markdown(f"- **{langue.get('langue', '')}**: {langue.get('niveau', '')}")
                else:
                    st.markdown(f"- {langue}")
        else:
            st.info("Aucune information sur les langues disponible")
        
        # Afficher les certifications
        if "certifications" in cv_data and cv_data["certifications"]:
            st.subheader("Certifications")
            for cert in cv_data["certifications"]:
                if isinstance(cert, dict):
                    st.markdown(f"- {cert.get('certificate', '')}")
                else:
                    st.markdown(f"- {cert}")
    
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des détails du CV: {str(e)}")
        logger.error(f"Erreur lors de l'affichage des détails du CV {cv_id}: {e}")

# Fonction pour exécuter le matching
def run_matching(job_offer_text: str, alpha: float = 0.7, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Exécute le matching entre les CV et l'offre d'emploi
    """
    try:
        # Sauvegarder l'offre d'emploi
        job_offer_path = save_job_offer(job_offer_text)
        
        if not job_offer_path:
            return []
        
        # Exécuter le matching
        results = match_cv_to_job_offer(
            job_offer_path=job_offer_path,
            cv_vectors_dir=DEFAULT_DIRS["vectors_dir"],
            cv_tokens_dir=DEFAULT_DIRS["tokens_dir"],
            cv_json_dir=DEFAULT_DIRS["cv_json_dir"],
            models_dir=DEFAULT_DIRS["models_dir"],
            output_dir=DEFAULT_DIRS["matching_dir"],
            alpha=alpha,
            top_n=top_n
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Erreur lors du matching: {e}")
        return []

# Fonction pour expliquer le matching
def explain_matching(job_offer_text: str, cv_id: str) -> Dict[str, Any]:
    """
    Explique pourquoi un CV a obtenu un certain score pour une offre d'emploi
    """
    try:
        # Sauvegarder l'offre d'emploi
        job_offer_path = save_job_offer(job_offer_text)
        
        if not job_offer_path:
            return {"error": "Impossible de sauvegarder l'offre d'emploi"}
        
        # Créer le matcher
        matcher = BM25Matcher(
            cv_vectors_dir=DEFAULT_DIRS["vectors_dir"],
            cv_tokens_dir=DEFAULT_DIRS["tokens_dir"],
            cv_json_dir=DEFAULT_DIRS["cv_json_dir"],
            models_dir=DEFAULT_DIRS["models_dir"],
            output_dir=DEFAULT_DIRS["matching_dir"]
        )
        
        # Charger les données
        matcher.load_cv_data()
        matcher.prepare_bm25_data()
        
        # Obtenir l'explication
        explanation = matcher.explain_matching(job_offer_path, cv_id)
        
        return explanation
    
    except Exception as e:
        logger.error(f"Erreur lors de l'explication du matching: {e}")
        return {"error": str(e)}

# Interface Streamlit principale
def main():
    st.set_page_config(
        page_title="IA Matching CV",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🧠 Plateforme IA de Matching CV - Offres d'emploi")
    
    # Sidebar pour les paramètres
    st.sidebar.title("⚙️ Paramètres")
    
    # Authentification Mistral API
    api_key = st.sidebar.text_input("Clé API Mistral", type="password", help="Entrez votre clé API Mistral pour analyser les CV")
    
    # Modèle à utiliser
    model = st.sidebar.selectbox(
        "Modèle Mistral",
        ["pixtral-12b-2409", "mistral-large", "mistral-small"],
        help="Sélectionnez le modèle Mistral à utiliser pour l'analyse des CV"
    )
    
    # Paramètres pour le matching
    alpha = st.sidebar.slider(
        "Poids BM25 (alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Ratio entre le score BM25 et la similitude vectorielle"
    )
    
    # Onglets pour organiser l'interface
    tab1, tab2, tab3 = st.tabs(["Traitement CV", "Offre d'emploi & Matching", "Base de CV"])
    
    # Onglet 1: Traitement des CV
    with tab1:
        st.header("🔍 Traitement de CV")
        st.info("Téléchargez un ou plusieurs CV au format PDF pour les analyser avec Mistral AI.")
        
        # Modification: Accepter plusieurs fichiers en même temps
        uploaded_files = st.file_uploader("Choisir des fichiers PDF", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            st.write(f"{len(uploaded_files)} CV sélectionnés")
            
            # Afficher les noms des fichiers
            for i, file in enumerate(uploaded_files):
                st.write(f"{i+1}. **{file.name}**")
            
            if st.button("Traiter les CV"):
                if not api_key:
                    st.error("Veuillez entrer votre clé API Mistral dans la barre latérale")
                else:
                    with st.spinner('Traitement des CV en cours...'):
                        # Traiter les CV
                        results = process_multiple_cvs(uploaded_files, api_key, model)
                        
                        # Afficher les résultats
                        success_count = sum(1 for r in results if r["success"])
                        if success_count > 0:
                            st.success(f"{success_count} CV traités avec succès sur {len(results)}")
                            
                            # Afficher un aperçu des données extraites
                            with st.expander("Aperçu des données extraites"):
                                for i, result in enumerate(results):
                                    if result["success"]:
                                        st.subheader(f"CV {i+1}: {uploaded_files[i].name}")
                                        st.json(result["result"])
                        else:
                            st.error("Erreur lors du traitement des CV")
    
    # Onglet 2: Offre d'emploi et Matching
    with tab2:
        st.header("🔎 Matching CV - Offre d'emploi")
        
        # Zone de texte pour l'offre d'emploi
        job_offer_text = st.text_area(
            "Entrez le texte de l'offre d'emploi",
            height=200,
            help="Saisissez le texte complet de l'offre d'emploi pour laquelle vous recherchez des candidats"
        )
        
        # Nombre de résultats à afficher
        top_n = st.slider(
            "Nombre de candidats à afficher",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
        
        # Utiliser un bouton de session state pour stocker les résultats du matching
        if "matching_results" not in st.session_state:
            st.session_state.matching_results = None
        
        if "selected_candidate" not in st.session_state:
            st.session_state.selected_candidate = None
        
        if "job_offer_for_matching" not in st.session_state:
            st.session_state.job_offer_for_matching = None
        
        # Fonction pour mettre à jour le candidat sélectionné sans rechargement
        def update_selected_candidate(cv_id):
            for candidate in st.session_state.matching_results:
                if candidate["cv_id"] == cv_id:
                    st.session_state.selected_candidate = candidate
                    break
        
        if st.button("Lancer le matching"):
            if not job_offer_text:
                st.error("Veuillez entrer le texte de l'offre d'emploi")
            else:
                with st.spinner('Recherche des candidats en cours...'):
                    # Exécuter le matching
                    results = run_matching(job_offer_text, alpha, top_n)
                    
                    if results:
                        st.success(f"{len(results)} candidats trouvés pour cette offre d'emploi")
                        # Stocker les résultats dans le session state
                        st.session_state.matching_results = results
                        st.session_state.job_offer_for_matching = job_offer_text
                        
                        # Initialiser le candidat sélectionné
                        if results:
                            st.session_state.selected_candidate = results[0]
                    else:
                        st.error("Aucun résultat trouvé ou erreur lors du matching")
        
        # Afficher les résultats du matching s'ils existent
        if st.session_state.matching_results:
            # Afficher les résultats dans un tableau
            df = pd.DataFrame(st.session_state.matching_results)
            df = df.rename(columns={
                "cv_id": "ID",
                "nom": "Nom",
                "prenom": "Prénom",
                "email": "Email",
                "score": "Score global",
                "bm25_score": "Score BM25",
                "vector_score": "Score vectoriel"
            })
            
            # Formater les scores
            for col in ["Score global", "Score BM25", "Score vectoriel"]:
                df[col] = df[col].apply(lambda x: f"{x:.2f}")
            
            # Afficher le tableau
            st.dataframe(df, use_container_width=True)
            
            # Sélectionner un candidat sans recharger la page
            cv_ids = [candidate["cv_id"] for candidate in st.session_state.matching_results]
            selected_cv_id = st.selectbox(
                "Sélectionnez un candidat pour voir les détails",
                options=cv_ids,
                format_func=lambda x: next((f"{c['prenom']} {c['nom']} (Score: {c['score']:.2f})" 
                                           for c in st.session_state.matching_results if c["cv_id"] == x), ""),
                key="candidate_selector",
                on_change=lambda: update_selected_candidate(st.session_state.candidate_selector)
            )
            
            # Afficher les détails du candidat sélectionné
            if st.session_state.selected_candidate:
                selected_candidate = st.session_state.selected_candidate
                
                # Afficher l'explication et les détails du CV
                st.subheader(f"Détails du matching pour {selected_candidate['prenom']} {selected_candidate['nom']}")
                
                # Obtenir l'explication du matching
                explanation = explain_matching(st.session_state.job_offer_for_matching, selected_candidate["cv_id"])
                
                if "error" in explanation:
                    st.error(explanation["error"])
                else:
                    # Afficher les termes communs
                    st.write(f"**Nombre de termes communs:** {explanation.get('common_terms', 0)}")
                    
                    # Afficher les termes les plus importants
                    st.subheader("Termes les plus pertinents")
                    top_terms = explanation.get("top_matching_terms", [])
                    
                    if top_terms:
                        # Créer un DataFrame pour les termes
                        terms_df = pd.DataFrame([
                            {
                                "Terme": term,
                                "Score": details["score"],
                                "Fréquence (TF)": details["tf"],
                                "IDF": details["idf"]
                            }
                            for term, details in top_terms
                        ])
                        
                        # Formater les scores
                        for col in ["Score", "IDF"]:
                            terms_df[col] = terms_df[col].apply(lambda x: f"{x:.4f}")
                        
                        # Afficher le tableau
                        st.dataframe(terms_df, use_container_width=True)
                    
                    # Afficher les détails du CV
                    with st.expander("Voir le CV complet"):
                        display_cv_details(selected_candidate["cv_id"])
    
    # Onglet 3: Base de CV
    with tab3:
        st.header("📚 Base de CV")
        
        # Charger les CV traités
        cv_data = load_processed_cvs()
        
        if cv_data:
            st.success(f"{len(cv_data)} CV dans la base de données")
            
            # Afficher la liste des CV
            df = pd.DataFrame(cv_data)
            df = df.rename(columns={
                "cv_id": "ID",
                "nom": "Nom",
                "prenom": "Prénom",
                "email": "Email",
                "filename": "Fichier",
                "analyzed_at": "Date d'analyse"
            })
            
            st.dataframe(df, use_container_width=True)
            
            # Sélectionner un CV pour voir les détails
            selected_cv = st.selectbox(
                "Sélectionnez un CV pour voir les détails",
                options=cv_data,
                format_func=lambda x: f"{x['prenom']} {x['nom']} - {x['filename']}"
            )
            
            if selected_cv:
                display_cv_details(selected_cv["cv_id"])
        else:
            st.info("Aucun CV traité dans la base de données. Veuillez ajouter des CV dans l'onglet 'Traitement CV'.")
            
            # Option pour créer un CV de démonstration
            if st.button("Créer un CV de démonstration"):
                st.warning("Cette fonctionnalité n'est pas encore implémentée")

if __name__ == "__main__":
    main()


# import os
# import streamlit as st
# import pandas as pd
# import numpy as np
# import json
# import tempfile
# import time
# from pathlib import Path
# import base64
# import shutil
# import logging
# from typing import List, Dict, Any, Optional, Union


# from preprocess.extraction import process_single_cv, get_api_key
# from preprocess.tokenization import vectorize_cv_dataset
# from preprocess.matcher_module import match_cv_to_job_offer, BM25Matcher

# # Configuration du logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger('StreamlitCV')

# # Configurer les chemins des répertoires
# DEFAULT_DIRS = {
#     "cv_dir": "cvs",
#     "output_dir": "output",
#     "cv_json_dir": "output/cv_individuels",
#     "vectorisation_dir": "output/vectorisation",
#     "vectors_dir": "output/vectorisation/vectors",
#     "tokens_dir": "output/vectorisation/tokens",
#     "models_dir": "output/vectorisation/models",
#     "matching_dir": "output/matching"
# }

# # Créer les répertoires s'ils n'existent pas
# for dir_path in DEFAULT_DIRS.values():
#     os.makedirs(dir_path, exist_ok=True)

# # Fonction pour créer ou mettre à jour le fichier de prompt d'offre d'emploi
# def save_job_offer(job_offer_text: str, job_offer_path: str = "prompt_offre.txt") -> str:
#     """
#     Sauvegarde le texte de l'offre d'emploi dans un fichier
#     """
#     try:
#         with open(job_offer_path, "w", encoding="utf-8") as f:
#             f.write(job_offer_text)
#         return job_offer_path
#     except Exception as e:
#         logger.error(f"Erreur lors de la sauvegarde de l'offre d'emploi: {e}")
#         return ""

# # Fonction pour traiter un CV nouvellement chargé
# def process_new_cv(uploaded_file, api_key: str, model: str = "pixtral-12b-2409",
#                   prompt_path: str = "cv_prompt.txt") -> Dict[str, Any]:
#     """
#     Traite un nouveau CV chargé via l'interface Streamlit
#     """
#     logger.info(f"Traitement du CV: {uploaded_file.name}")
    
#     # Créer un fichier temporaire pour le CV chargé
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_path = tmp_file.name
    
#     try:
#         # Copier le fichier dans le répertoire des CV
#         cv_path = os.path.join(DEFAULT_DIRS["cv_dir"], uploaded_file.name)
#         shutil.copy(tmp_path, cv_path)
        
#         # Traiter le CV
#         result = process_single_cv(
#             pdf_path=cv_path,
#             prompt_path=prompt_path,
#             model=model,
#             output_dir=DEFAULT_DIRS["output_dir"],
#             api_key=api_key
#         )
        
#         # Vectoriser le CV traité
#         vectorize_cv_dataset(
#             input_dir=DEFAULT_DIRS["cv_json_dir"],
#             output_dir=DEFAULT_DIRS["vectorisation_dir"]
#         )
        
#         return {
#             "success": True,
#             "message": f"CV {uploaded_file.name} traité avec succès",
#             "result": result
#         }
    
#     except Exception as e:
#         logger.error(f"Erreur lors du traitement du CV: {e}")
#         return {
#             "success": False,
#             "message": f"Erreur: {str(e)}",
#             "result": None
#         }
#     finally:
#         # Supprimer le fichier temporaire
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

# # Fonction pour charger les CV déjà traités
# def load_processed_cvs() -> List[Dict[str, Any]]:
#     """
#     Charge les données des CV déjà traités
#     """
#     cv_data = []
    
#     try:
#         json_dir = Path(DEFAULT_DIRS["cv_json_dir"])
#         json_files = list(json_dir.glob("*_analyse.json"))
        
#         for json_file in json_files:
#             try:
#                 with open(json_file, "r", encoding="utf-8") as f:
#                     data = json.load(f)
                
#                 # Extraire les informations clés
#                 cv_id = json_file.stem.replace("_analyse", "")
#                 metadata = data.get("_metadata", {})
#                 info_perso = data.get("informations_personnelles", {})
                
#                 cv_data.append({
#                     "cv_id": cv_id,
#                     "nom": info_perso.get("nom", ""),
#                     "prenom": info_perso.get("prenom", ""),
#                     "email": info_perso.get("email", ""),
#                     "filename": metadata.get("filename", ""),
#                     "analyzed_at": metadata.get("analyzed_at", "")
#                 })
            
#             except Exception as e:
#                 logger.error(f"Erreur lors du chargement du CV {json_file}: {e}")
        
#         return cv_data
    
#     except Exception as e:
#         logger.error(f"Erreur lors du chargement des CV traités: {e}")
#         return []

# # Fonction pour afficher les détails d'un CV
# def display_cv_details(cv_id: str) -> None:
#     """
#     Affiche les détails d'un CV spécifique
#     """
#     try:
#         json_path = Path(DEFAULT_DIRS["cv_json_dir"]) / f"{cv_id}_analyse.json"
        
#         if not json_path.exists():
#             st.error(f"Le fichier {json_path} n'existe pas")
#             return
        
#         with open(json_path, "r", encoding="utf-8") as f:
#             cv_data = json.load(f)
        
#         # Afficher les informations personnelles
#         st.subheader("Informations personnelles")
#         info_perso = cv_data.get("informations_personnelles", {})
#         col1, col2 = st.columns(2)
#         with col1:
#             st.write(f"**Nom:** {info_perso.get('nom', 'Non spécifié')}")
#             st.write(f"**Prénom:** {info_perso.get('prenom', 'Non spécifié')}")
#         with col2:
#             st.write(f"**Email:** {info_perso.get('email', 'Non spécifié')}")
#             if "telephone" in info_perso:
#                 st.write(f"**Téléphone:** {info_perso['telephone']}")
        
#         # Afficher la formation
#         st.subheader("Formation")
#         formations = cv_data.get("formation_et_education", [])
#         if formations:
#             for formation in formations:
#                 st.markdown(f"""
#                 **{formation.get('diplome', 'Diplôme non spécifié')}**  
#                 *{formation.get('ecole', 'École non spécifiée')}*  
#                 {formation.get('dates', '')}
#                 """)
#         else:
#             st.info("Aucune information de formation disponible")
        
#         # Afficher l'expérience professionnelle
#         st.subheader("Expérience professionnelle")
#         experiences = cv_data.get("experience_professionnelle", [])
#         if experiences:
#             for exp in experiences:
#                 st.markdown(f"""
#                 **{exp.get('poste', 'Poste non spécifié')}** - {exp.get('entreprise', 'Entreprise non spécifiée')}  
#                 *{exp.get('dates', '')}*
#                 """)
                
#                 # Afficher la description si disponible
#                 if "description" in exp and exp["description"]:
#                     if isinstance(exp["description"], list):
#                         for desc in exp["description"]:
#                             st.markdown(f"- {desc}")
#                     else:
#                         st.write(exp["description"])
#         else:
#             st.info("Aucune expérience professionnelle disponible")
        
#         # Afficher les compétences
#         st.subheader("Compétences techniques")
#         competences = cv_data.get("competences_techniques", [])
#         if competences:
#             if isinstance(competences, list):
#                 # Afficher en colonnes si la liste est longue
#                 if len(competences) > 5:
#                     cols = st.columns(3)
#                     for i, comp in enumerate(competences):
#                         cols[i % 3].markdown(f"- {comp}")
#                 else:
#                     for comp in competences:
#                         st.markdown(f"- {comp}")
#             else:
#                 st.write(competences)
#         else:
#             st.info("Aucune compétence technique disponible")
        
#         # Afficher les langues
#         st.subheader("Langues")
#         langues = cv_data.get("langues", [])
#         if langues:
#             for langue in langues:
#                 if isinstance(langue, dict):
#                     st.markdown(f"- **{langue.get('langue', '')}**: {langue.get('niveau', '')}")
#                 else:
#                     st.markdown(f"- {langue}")
#         else:
#             st.info("Aucune information sur les langues disponible")
        
#         # Afficher les certifications
#         if "certifications" in cv_data and cv_data["certifications"]:
#             st.subheader("Certifications")
#             for cert in cv_data["certifications"]:
#                 if isinstance(cert, dict):
#                     st.markdown(f"- {cert.get('certificate', '')}")
#                 else:
#                     st.markdown(f"- {cert}")
    
#     except Exception as e:
#         st.error(f"Erreur lors de l'affichage des détails du CV: {str(e)}")
#         logger.error(f"Erreur lors de l'affichage des détails du CV {cv_id}: {e}")

# # Fonction pour exécuter le matching
# def run_matching(job_offer_text: str, alpha: float = 0.7, top_n: int = 10) -> List[Dict[str, Any]]:
#     """
#     Exécute le matching entre les CV et l'offre d'emploi
#     """
#     try:
#         # Sauvegarder l'offre d'emploi
#         job_offer_path = save_job_offer(job_offer_text)
        
#         if not job_offer_path:
#             return []
        
#         # Exécuter le matching
#         results = match_cv_to_job_offer(
#             job_offer_path=job_offer_path,
#             cv_vectors_dir=DEFAULT_DIRS["vectors_dir"],
#             cv_tokens_dir=DEFAULT_DIRS["tokens_dir"],
#             cv_json_dir=DEFAULT_DIRS["cv_json_dir"],
#             models_dir=DEFAULT_DIRS["models_dir"],
#             output_dir=DEFAULT_DIRS["matching_dir"],
#             alpha=alpha,
#             top_n=top_n
#         )
        
#         return results
    
#     except Exception as e:
#         logger.error(f"Erreur lors du matching: {e}")
#         return []

# # Fonction pour expliquer le matching
# def explain_matching(job_offer_text: str, cv_id: str) -> Dict[str, Any]:
#     """
#     Explique pourquoi un CV a obtenu un certain score pour une offre d'emploi
#     """
#     try:
#         # Sauvegarder l'offre d'emploi
#         job_offer_path = save_job_offer(job_offer_text)
        
#         if not job_offer_path:
#             return {"error": "Impossible de sauvegarder l'offre d'emploi"}
        
#         # Créer le matcher
#         matcher = BM25Matcher(
#             cv_vectors_dir=DEFAULT_DIRS["vectors_dir"],
#             cv_tokens_dir=DEFAULT_DIRS["tokens_dir"],
#             cv_json_dir=DEFAULT_DIRS["cv_json_dir"],
#             models_dir=DEFAULT_DIRS["models_dir"],
#             output_dir=DEFAULT_DIRS["matching_dir"]
#         )
        
#         # Charger les données
#         matcher.load_cv_data()
#         matcher.prepare_bm25_data()
        
#         # Obtenir l'explication
#         explanation = matcher.explain_matching(job_offer_path, cv_id)
        
#         return explanation
    
#     except Exception as e:
#         logger.error(f"Erreur lors de l'explication du matching: {e}")
#         return {"error": str(e)}

# # Interface Streamlit principale
# def main():
#     st.set_page_config(
#         page_title="IA Matching CV",
#         page_icon="📄",
#         layout="wide"
#     )
    
#     st.title("🧠 Plateforme IA de Matching CV - Offres d'emploi")
    
#     # Sidebar pour les paramètres
#     st.sidebar.title("⚙️ Paramètres")
    
#     # Authentification Mistral API
#     api_key = st.sidebar.text_input("Clé API Mistral", type="password", help="Entrez votre clé API Mistral pour analyser les CV")
    
#     # Modèle à utiliser
#     model = st.sidebar.selectbox(
#         "Modèle Mistral",
#         ["pixtral-12b-2409", "mistral-large", "mistral-small"],
#         help="Sélectionnez le modèle Mistral à utiliser pour l'analyse des CV"
#     )
    
#     # Paramètres pour le matching
#     alpha = st.sidebar.slider(
#         "Poids BM25 (alpha)",
#         min_value=0.0,
#         max_value=1.0,
#         value=0.7,
#         step=0.1,
#         help="Ratio entre le score BM25 et la similitude vectorielle"
#     )
    
#     # Onglets pour organiser l'interface
#     tab1, tab2, tab3 = st.tabs(["Traitement CV", "Offre d'emploi & Matching", "Base de CV"])
    
#     # Onglet 1: Traitement des CV
#     with tab1:
#         st.header("🔍 Traitement de CV")
#         st.info("Téléchargez un nouveau CV au format PDF pour l'analyser avec Mistral AI.")
        
#         uploaded_file = st.file_uploader("Choisir un fichier PDF", type=["pdf"])
        
#         if uploaded_file is not None:
#             st.write(f"CV sélectionné: **{uploaded_file.name}**")
            
#             if st.button("Traiter le CV"):
#                 if not api_key:
#                     st.error("Veuillez entrer votre clé API Mistral dans la barre latérale")
#                 else:
#                     with st.spinner('Traitement du CV en cours...'):
#                         # Afficher la progression
#                         progress_bar = st.progress(0)
#                         status_text = st.empty()
                        
#                         # Simuler les étapes du traitement
#                         status_text.text("Étape 1/4: Extraction du texte du PDF...")
#                         progress_bar.progress(25)
#                         time.sleep(1)
                        
#                         status_text.text("Étape 2/4: Envoi à l'API Mistral...")
#                         progress_bar.progress(50)
                        
#                         # Traiter le CV
#                         result = process_new_cv(uploaded_file, api_key, model)
                        
#                         status_text.text("Étape 3/4: Vectorisation du CV...")
#                         progress_bar.progress(75)
#                         time.sleep(1)
                        
#                         status_text.text("Étape 4/4: Finalisation...")
#                         progress_bar.progress(100)
#                         time.sleep(0.5)
                        
#                         # Afficher le résultat
#                         if result["success"]:
#                             st.success(result["message"])
                            
#                             # Afficher un aperçu des données extraites
#                             if result["result"]:
#                                 with st.expander("Aperçu des données extraites"):
#                                     st.json(result["result"])
#                         else:
#                             st.error(result["message"])
    
#     # Onglet 2: Offre d'emploi et Matching
#     with tab2:
#         st.header("🔎 Matching CV - Offre d'emploi")
        
#         # Zone de texte pour l'offre d'emploi
#         job_offer_text = st.text_area(
#             "Entrez le texte de l'offre d'emploi",
#             height=200,
#             help="Saisissez le texte complet de l'offre d'emploi pour laquelle vous recherchez des candidats"
#         )
        
#         # Nombre de résultats à afficher
#         top_n = st.slider(
#             "Nombre de candidats à afficher",
#             min_value=1,
#             max_value=20,
#             value=5,
#             step=1
#         )
        
#         if st.button("Lancer le matching"):
#             if not job_offer_text:
#                 st.error("Veuillez entrer le texte de l'offre d'emploi")
#             else:
#                 with st.spinner('Recherche des candidats en cours...'):
#                     # Exécuter le matching
#                     results = run_matching(job_offer_text, alpha, top_n)
                    
#                     if results:
#                         st.success(f"{len(results)} candidats trouvés pour cette offre d'emploi")
                        
#                         # Afficher les résultats dans un tableau
#                         df = pd.DataFrame(results)
#                         df = df.rename(columns={
#                             "cv_id": "ID",
#                             "nom": "Nom",
#                             "prenom": "Prénom",
#                             "email": "Email",
#                             "score": "Score global",
#                             "bm25_score": "Score BM25",
#                             "vector_score": "Score vectoriel"
#                         })
                        
#                         # Formater les scores
#                         for col in ["Score global", "Score BM25", "Score vectoriel"]:
#                             df[col] = df[col].apply(lambda x: f"{x:.2f}")
                        
#                         # Afficher le tableau
#                         st.dataframe(df, use_container_width=True)
                        
#                         # Sélectionner un candidat pour voir les détails
#                         selected_candidate = st.selectbox(
#                             "Sélectionnez un candidat pour voir les détails",
#                             options=results,
#                             format_func=lambda x: f"{x['prenom']} {x['nom']} (Score: {x['score']:.2f})"
#                         )
                        
#                         if selected_candidate:
#                             # Afficher l'explication et les détails du CV
#                             st.subheader(f"Détails du matching pour {selected_candidate['prenom']} {selected_candidate['nom']}")
                            
#                             # Obtenir l'explication du matching
#                             explanation = explain_matching(job_offer_text, selected_candidate["cv_id"])
                            
#                             if "error" in explanation:
#                                 st.error(explanation["error"])
#                             else:
#                                 # Afficher les termes communs
#                                 st.write(f"**Nombre de termes communs:** {explanation.get('common_terms', 0)}")
                                
#                                 # Afficher les termes les plus importants
#                                 st.subheader("Termes les plus pertinents")
#                                 top_terms = explanation.get("top_matching_terms", [])
                                
#                                 if top_terms:
#                                     # Créer un DataFrame pour les termes
#                                     terms_df = pd.DataFrame([
#                                         {
#                                             "Terme": term,
#                                             "Score": details["score"],
#                                             "Fréquence (TF)": details["tf"],
#                                             "IDF": details["idf"]
#                                         }
#                                         for term, details in top_terms
#                                     ])
                                    
#                                     # Formater les scores
#                                     for col in ["Score", "IDF"]:
#                                         terms_df[col] = terms_df[col].apply(lambda x: f"{x:.4f}")
                                    
#                                     # Afficher le tableau
#                                     st.dataframe(terms_df, use_container_width=True)
                                
#                                 # Afficher les détails du CV
#                                 with st.expander("Voir le CV complet"):
#                                     display_cv_details(selected_candidate["cv_id"])
#                     else:
#                         st.error("Aucun résultat trouvé ou erreur lors du matching")
    
#     # Onglet 3: Base de CV
#     with tab3:
#         st.header("📚 Base de CV")
        
#         # Charger les CV traités
#         cv_data = load_processed_cvs()
        
#         if cv_data:
#             st.success(f"{len(cv_data)} CV dans la base de données")
            
#             # Afficher la liste des CV
#             df = pd.DataFrame(cv_data)
#             df = df.rename(columns={
#                 "cv_id": "ID",
#                 "nom": "Nom",
#                 "prenom": "Prénom",
#                 "email": "Email",
#                 "filename": "Fichier",
#                 "analyzed_at": "Date d'analyse"
#             })
            
#             st.dataframe(df, use_container_width=True)
            
#             # Sélectionner un CV pour voir les détails
#             selected_cv = st.selectbox(
#                 "Sélectionnez un CV pour voir les détails",
#                 options=cv_data,
#                 format_func=lambda x: f"{x['prenom']} {x['nom']} - {x['filename']}"
#             )
            
#             if selected_cv:
#                 display_cv_details(selected_cv["cv_id"])
#         else:
#             st.info("Aucun CV traité dans la base de données. Veuillez ajouter des CV dans l'onglet 'Traitement CV'.")
            
#             # Option pour créer un CV de démonstration
#             if st.button("Créer un CV de démonstration"):
#                 st.warning("Cette fonctionnalité n'est pas encore implémentée")

# if __name__ == "__main__":
#     main()