import pandas as pd
from backend.nettoyage import preprocess_tickets
from backend.traitement_ai import extract_ticket_info
from backend.stockage import store_ticket_in_mongo, get_collection_stats

def main(fichier_excel):
    print(" Démarrage du pipeline de traitement des tickets Jira...")
    print(f" Chargement du fichier: {fichier_excel}")
    
    # Étape 1 : Charger et nettoyer les tickets
    df = pd.read_excel(fichier_excel)
    
    # Initialiser les compteurs
    stats = {
        "total": len(df),
        "processed": 0,
        "fixed": 0,
        "skipped": 0
    }
    
    # Filtrer les tickets avec statut "Fixed"
    df_fixed = df[df['status'].str.contains('Fixed', case=False, na=False)]
    stats["fixed"] = len(df_fixed)
    stats["skipped"] = stats["total"] - stats["fixed"]
    
    if stats["fixed"] == 0:
        print(" Aucun ticket avec le statut 'Fixed' trouvé dans le fichier.")
        return stats
    
    # Nettoyer les tickets filtrés
    df_propre = preprocess_tickets(df_fixed)
    print(f" Nettoyage terminé - {len(df_propre)} tickets traités")
    
    # Étape 2 : Extraire les informations avec l'IA
    print("\n Extraction des informations avec l'IA...")
    extracted_data = []
    
    for index, row in df_propre.iterrows():
        print(f" Traitement du ticket {index+1}/{len(df_propre)}...")
        ticket_id = row.get('key', '') # Utiliser l'ID original du fichier
        ticket_text = " ".join(str(value) for value in row.values if pd.notna(value))
        summary = extract_ticket_info(ticket_text)
        
        if summary:
            # Initialiser les variables
            problem = "Inconnu"
            solution = "Non résolu"
            keywords = ""
            
            # Extraire proprement les informations
            lines = summary.split("\n")
            for i, line in enumerate(lines):
                line = line.strip()
                if line.lower().startswith("### problématique"):
                    problem = lines[i+1].strip() if i+1 < len(lines) and lines[i+1].strip() else "Inconnu"
                elif line.lower().startswith("### solution"):
                    solution = lines[i+1].strip() if i+1 < len(lines) and lines[i+1].strip() else "Non résolu"
                elif line.lower().startswith("### mots-clés techniques"):
                    keywords = lines[i+1].strip() if i+1 < len(lines) and lines[i+1].strip() else ""
            
            # Ajouter au tableau extrait
            extracted_data.append({
                "ID": ticket_id,
                "Problématique": problem,
                "Solution": solution,
                "Mots-clés techniques": keywords,
                "Texte brut": ticket_text
            })
    
    # Convertir en DataFrame et sauvegarder dans un fichier CSV
    if extracted_data:
        df_extracted = pd.DataFrame(extracted_data)
        output_file = "résultat_résumé_tickets.csv"
        df_extracted.to_csv(output_file, index=False)
        print(f" Données extraites sauvegardées dans {output_file}")
        
        # Étape 3 : Stocker dans MongoDB
        print("\n Stockage des tickets dans MongoDB...")
        for index, row in df_extracted.iterrows():
            ticket_data = {
                "ID": row["ID"],
                "problem": row["Problématique"],
                "solution": row["Solution"],
                "keywords": row["Mots-clés techniques"]
            }
            store_ticket_in_mongo(ticket_data)
            stats["processed"] += 1
    
    # Vérifier les statistiques de la collection
    mongo_stats = get_collection_stats()
    print(f" Base de données: {mongo_stats['count']} tickets stockés au total")
    print("\n Pipeline terminé avec succès!")
    
    return stats
