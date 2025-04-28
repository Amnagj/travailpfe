import requests
import json
import time

# URL de l'API locale de LM Studio
LM_STUDIO_API = "http://localhost:1234/v1/chat/completions"

def extract_ticket_info(ticket_text):
    """
    Envoie un texte de ticket à LM Studio et récupère la réponse analysée.
    """
    prompt = f"""
    Insctruction:
    Vous êtes un expert en analyse de tickets Jira spécialisé dans l'extraction intelligente de problématiques et solutions. Votre tâche est d'analyser l'intégralité du contenu du ticket pour en dégager l'essence, indépendamment de la structure des champs.
    Objectif:
    Extraire une représentation structurée et standardisée du ticket qui servira à alimenter une base vectorielle pour la recherche de similarité entre tickets. Cette extraction doit être optimisée pour permettre l'identification rapide de problèmes similaires et leurs solutions associées.
    Analyse requise:
    Examinez tous les éléments du ticket (type, description, commentaires, composants,  solution, impact …) et analysez-les comme un ensemble cohérent pour identifier:
    Problématique principale - Identifiez le cœur du problème en termes techniques précis, au-delà de ce qui est simplement déclaré dans le titre ou la description. Recherchez les indices dans l'ensemble du ticket, y compris les commentaires techniques.
    Solution effective - Déterminez quelle action a réellement résolu le problème (pas seulement ce qui est indiqué dans le champ "solution")

    Ticket:
    {ticket_text}

    Format de sortie:
    ### Problématique
    [Synthèse technique du problème]
    ### Solution
    [Solution appliquée ou "Non résolu"]
    ### Mots-clés techniques
    [Liste de 3-7 mots clés techniques]
    """

    payload = {
        "model": "mistral-nemo-instruct-2407",  # Remplace par ton modèle téléchargé dans LM Studio
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(LM_STUDIO_API, json=payload)
        response.raise_for_status()  # Vérifie les erreurs HTTP
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête à LM Studio: {e}")
        return None
