import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Charger les stopwords en français et anglais
stop_words = set(stopwords.words('french') + stopwords.words('english'))

def nettoyer_texte(texte):
    if not isinstance(texte, str) or texte.strip() == "":
        return ""
    
    # Suppression des adresses email
    texte = re.sub(r'\S+@\S+', '', texte)
    
    # Suppression des caractères spéciaux, mais garder les chiffres
    texte = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', '', texte)
    
    # Conversion en minuscules
    texte = texte.lower()
    
    # Tokenization
    mots = word_tokenize(texte)
    
    # Suppression des stopwords
    mots = [mot for mot in mots if mot not in stop_words]
    
    return " ".join(mots)

# Charger les tickets depuis un CSV
def preprocess_tickets(input_data):
    """
    Nettoie et prépare les données des tickets pour l'analyse.
    Accepte soit un chemin de fichier Excel, soit un DataFrame pandas.
    """
    # Vérifier si l'entrée est un DataFrame ou un chemin de fichier
    if isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        # C'est un chemin de fichier
        df = pd.read_excel(input_data, engine="openpyxl")

    # Liste des colonnes à supprimer
    colonnes_a_supprimer = [
        'created_date', 'updated_date', 'assignee', 'reporter', 'estimated_budget', 
        'original_estimate', 'htu', 'git_branch', 'rank', 'date_of_first_response', 
        'participants', 'git_commits_referenced', 'rank_obsolete', 'request_participants', 
        'sprint', 'estimation_due_date', 'fix_estimation', 'classement', 'resolution', 'time_in_status',
        'last_commented'
    ]
    
    # Supprimer les colonnes inutiles (si elles existent)
    df = df.drop(columns=[colonne for colonne in colonnes_a_supprimer if colonne in df.columns])
    
    # Appliquer le nettoyage à toutes les colonnes contenant des chaînes de caractères
    for colonne in df.columns:
        if df[colonne].dtype == 'object':  # Vérifie si la colonne contient des chaînes
            df[colonne] = df[colonne].apply(nettoyer_texte)
    
    # Filtrer les lignes vides après nettoyage
    df = df.dropna(how='all')  # Supprime les lignes où toutes les colonnes sont vides
    
    return df
   
    
