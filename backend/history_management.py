"""
Module de gestion de l'historique des recherches pour l'application Ticket AI Wizard.
Permet de stocker, récupérer et gérer l'historique des recherches des utilisateurs.
"""
import os
from pymongo import MongoClient
import logging
import datetime
import uuid

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration MongoDB
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "Access"
COLLECTION_NAME = "Historique_Messages"

def connect_to_mongodb():
    """Établit une connexion à la base de données MongoDB pour l'historique."""
    try:
        # Afficher l'URI pour déboguer
        logger.info(f"Tentative de connexion à MongoDB avec URI: {MONGO_URI}")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Tester la connexion
        client.server_info()
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        logger.info("Connexion à MongoDB (historique) établie avec succès")
        return client, collection
    except Exception as e:
        logger.error(f"Erreur de connexion à MongoDB: {e}")
        # Ne pas lever d'exception, mais retourner None
        return None, None

def add_history_item(user_id, query_text, result, ticket_ids=None):
    try:
        client, collection = connect_to_mongodb()
        # Vérifier si la connexion a réussi
        if not client or not collection:
            return {
                "status": "error",
                "message": "Échec de la connexion à MongoDB"
            }
        
        # S'assurer que user_id est une chaîne
        user_id = str(user_id)
        
        # Créer l'élément d'historique
        history_item = {
            "id": str(uuid.uuid4()),
            "userId": user_id,
            "queryText": query_text,
            "result": result if isinstance(result, str) else str(result),
            "ticketIds": ticket_ids if ticket_ids else [],
            "timestamp": int(datetime.datetime.now().timestamp() * 1000),
            "visible": True
        }
        
        # Insérer dans la collection
        collection.insert_one(history_item)
        return {
            "status": "success",
            "message": "Recherche ajoutée à l'historique",
            "item": history_item
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur lors de l'ajout à l'historique: {str(e)}"
        }
    finally:
        if client:
            client.close()

def get_user_history(user_id):
    """
    Récupère l'historique des recherches d'un utilisateur (seulement les éléments visibles).
    Args:
        user_id (str): Identifiant de l'utilisateur
    Returns:
        dict: Historique des recherches de l'utilisateur
    """
    try:
        client, collection = connect_to_mongodb()
        
        # Vérifier si la connexion a réussi
        if not client or not collection:
            return {
                "status": "error",
                "message": "Échec de la connexion à MongoDB"
            }
            
        # Récupérer les éléments visibles, triés par date (du plus récent au plus ancien)
        cursor = collection.find(
            {"userId": user_id, "visible": True}
        ).sort("timestamp", -1)
        
        # Convertir le curseur en liste
        history_items = list(cursor)
        
        # Supprimer l'ID MongoDB pour éviter des problèmes de sérialisation
        for item in history_items:
            if "_id" in item:
                del item["_id"]
                
        logger.info(f"Historique récupéré pour l'utilisateur {user_id}: {len(history_items)} éléments")
        
        return {
            "status": "success",
            "history": history_items
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {e}")
        return {
            "status": "error",
            "message": f"Erreur lors de la récupération de l'historique: {str(e)}"
        }
    finally:
        if 'client' in locals() and client:
            client.close()


def hide_history_item(item_id):
    """
    Masque un élément d'historique (le rend invisible pour l'utilisateur).
    
    Args:
        item_id (str): Identifiant de l'élément à masquer
        
    Returns:
        dict: Résultat de l'opération
    """
    try:
        client, collection = connect_to_mongodb()
        
        # Mettre à jour l'élément pour le rendre invisible
        result = collection.update_one(
            {"id": item_id},
            {"$set": {"visible": False}}
        )
        
        client.close()
        
        if result.modified_count > 0:
            logger.info(f"Élément d'historique {item_id} masqué avec succès")
            return {
                "status": "success",
                "message": "Élément masqué avec succès"
            }
        else:
            logger.warning(f"Élément d'historique {item_id} non trouvé ou déjà masqué")
            return {
                "status": "warning",
                "message": "Élément non trouvé ou déjà masqué"
            }
    except Exception as e:
        logger.error(f"Erreur lors du masquage de l'élément: {e}")
        return {
            "status": "error",
            "message": f"Erreur lors du masquage de l'élément: {str(e)}"
        }

def clear_user_history(user_id):
    """
    Masque tous les éléments d'historique d'un utilisateur.
    
    Args:
        user_id (str): Identifiant de l'utilisateur
        
    Returns:
        dict: Résultat de l'opération
    """
    try:
        client, collection = connect_to_mongodb()
        
        # Mettre à jour tous les éléments de l'utilisateur pour les rendre invisibles
        result = collection.update_many(
            {"userId": user_id, "visible": True},
            {"$set": {"visible": False}}
        )
        
        client.close()
        
        logger.info(f"Historique effacé pour l'utilisateur {user_id}: {result.modified_count} éléments masqués")
        return {
            "status": "success",
            "message": f"{result.modified_count} éléments masqués avec succès"
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'effacement de l'historique: {e}")
        return {
            "status": "error",
            "message": f"Erreur lors de l'effacement de l'historique: {str(e)}"
        }