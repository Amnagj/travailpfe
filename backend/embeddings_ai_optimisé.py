import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import re
import logging
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from backend.nettoyage import nettoyer_texte
from traitement_ai import extract_ticket_info

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "MONGODB": {
        "uri": "mongodb://localhost:27017/",
        "db_name": "jira",
        "collections": {
            "tickets": "tickets",
            "resultats": "tickets_similarite_tfidf",
            "vecteurs": "tickets_vecteurs" 
        }
    },
    "PATHS": {
        "excel_input": "C:/Users/gouja/Downloads/Ticket117.xlsx"
    },
    "SEARCH_PARAMS": {
        "max_tickets": None,
        "use_cached_vectors": True,
        "similarity_threshold": 0.00,
        "top_n_results": 5,
        "batch_size": 100,  # Pour le traitement par lots des embeddings
        "use_parallel_processing": True,  # Nouveau paramètre pour le traitement parallèle
        "num_workers": 4  # Nombre de workers pour le traitement parallèle
    }
}

class RechercheTicketsEmbeddingsOptimized:
    def __init__(self, config=CONFIG):
        """
        Initialise la connexion à MongoDB et prépare le modèle d'embeddings.
        """
        try:
            # Stocker les paramètres de configuration
            self.config = config
            mongodb_config = config["MONGODB"]
            search_params = config["SEARCH_PARAMS"]
            self.fichier_excel = config["PATHS"]["excel_input"]
            
            # Connexion à MongoDB
            self.client = MongoClient(mongodb_config["uri"])
            self.db = self.client[mongodb_config["db_name"]]
            self.tickets_collection = self.db[mongodb_config["collections"]["tickets"]]
            self.resultats_collection = self.db[mongodb_config["collections"]["resultats"]]
            self.vecteurs_collection = self.db[mongodb_config["collections"]["vecteurs"]]
            
            # Initialiser le modèle d'embeddings
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Initialiser le stemmer et les stop words
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english') + stopwords.words('french'))
            
            # Configurer les paramètres de recherche
            self.max_tickets = search_params.get("max_tickets")
            self.use_cached_vectors = search_params.get("use_cached_vectors", True)
            self.similarity_threshold = search_params.get("similarity_threshold", 0.25)
            self.top_n_results = search_params.get("top_n_results", 5)
            self.batch_size = search_params.get("batch_size", 100)
            self.use_parallel = search_params.get("use_parallel_processing", True)
            self.num_workers = search_params.get("num_workers", 4)
            
            # S'assurer que la collection des vecteurs est correctement indexée
            self._create_vector_index()
            
            logger.info(f"✅ Connexion réussie à MongoDB, base '{mongodb_config['db_name']}'")
            count = self.tickets_collection.count_documents({})
            logger.info(f"📊 Nombre de tickets résolus dans la base: {count}")
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
            print(f"❌ Erreur d'initialisation: {e}")
            raise

    def _create_vector_index(self):
        """
        Crée un index sur le champ embedding_vector pour accélérer les recherches de similarité
        """
        try:
            # Vérifier si l'index existe déjà
            existing_indexes = self.vecteurs_collection.list_indexes()
            has_vector_index = any('embedding_vector' in idx.get('key', {}) for idx in existing_indexes)
            
            if not has_vector_index:
                logger.info("🔧 Création d'un index pour les vecteurs d'embeddings...")
                # Création d'un index composé pour améliorer les performances de recherche
                self.vecteurs_collection.create_index([("embedding_vector", 1), ("ticket_id", 1)])
                logger.info("✅ Index créé avec succès")
            else:
                logger.info("✅ Index d'embeddings déjà existant")
                
        except OperationFailure as e:
            logger.warning(f"⚠️ Impossible de créer l'index: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'index: {e}")

    def preprocess_text(self, text):
        """
        Prétraite le texte pour la vectorisation
        """
        if not text or not isinstance(text, str):
            return ""
            
        try:
            # Convertir en minuscules
            text = text.lower()
            
            # Supprimer les caractères spéciaux et la ponctuation
            text = re.sub(r'[^\w\s]', '', text)
            
            # Tokeniser le texte
            tokens = text.split()
            
            # Supprimer les stop words et appliquer stemming en une seule passe
            processed_tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
            
            # Rejoindre les tokens
            return ' '.join(processed_tokens)
        
        except Exception as e:
            logger.error(f"❌ Erreur de prétraitement du texte: {e}")
            return text if isinstance(text, str) else ""

    def formater_requete(self, ticket_text):
        """
        Formate la requête de ticket avec extraction d'informations et prétraitement
        """
        try:
            # Nettoyer le texte
            cleaned_text = nettoyer_texte(ticket_text)

            # Prétraiter le texte pour la recherche
            processed_text = self.preprocess_text(cleaned_text)
            
            # Extraire les informations avec l'IA (partie à conserver)
            extracted_info = extract_ticket_info(processed_text)
            
            if extracted_info:
                lines = extracted_info.split("\n")
                problem, keywords = "Inconnu", ""
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.lower().startswith("### problématique"):
                        problem = lines[i+1].strip() if i+1 < len(lines) else "Inconnu"
                    elif line.lower().startswith("### mots-clés techniques"):
                        keywords = lines[i+1].strip() if i+1 < len(lines) else ""
                
                return {
                    "problem": problem, 
                    "keywords": keywords, 
                    "original_text": cleaned_text,
                    "processed_text": processed_text
                }
            
            # Version simplifiée sans extraction réussie
            return {
                "problem": cleaned_text,
                "original_text": cleaned_text,
                "processed_text": processed_text
            }
        except Exception as e:
            logger.error(f"❌ Erreur formatage requête: {e}")
            # Fallback si tout échoue
            return {
                "problem": ticket_text, 
                "original_text": ticket_text,
                "processed_text": self.preprocess_text(ticket_text) if isinstance(ticket_text, str) else ""
            }

    def verifier_vecteurs_cache(self):
        """
        Vérifie si des embeddings sont déjà stockés dans MongoDB
        """
        count_vecteurs = self.vecteurs_collection.count_documents({"embedding_vector": {"$exists": True}})
        count_tickets = self.tickets_collection.count_documents({})
        
        if count_vecteurs > 0:
            logger.info(f"📊 {count_vecteurs}/{count_tickets} embeddings trouvés en cache")
            if count_vecteurs < count_tickets:
                logger.warning(f"⚠️ {count_tickets - count_vecteurs} tickets n'ont pas d'embeddings")
            return count_vecteurs > 0
        return False

    def generer_embeddings_manquants(self):
        """
        Génère les embeddings pour les tickets qui n'en ont pas encore
        """
        try:
            # Récupérer les IDs des tickets qui ont déjà des embeddings
            vecteurs_ids = set(str(vec["ticket_id"]) for vec in self.vecteurs_collection.find({}, {"ticket_id": 1}))
            
            # Préparer le pipeline pour les tickets manquants
            pipeline = [
                {"$match": {"_id": {"$nin": list(vecteurs_ids)}}},
            ]
            
            if self.max_tickets:
                pipeline.append({"$limit": self.max_tickets})
                
            # Exécuter l'agrégation pour obtenir les tickets manquants
            tickets_sans_vecteurs = list(self.tickets_collection.aggregate(pipeline))
            
            if not tickets_sans_vecteurs:
                logger.info("✅ Tous les tickets ont des embeddings")
                return True
                
            logger.info(f"🔍 Génération d'embeddings pour {len(tickets_sans_vecteurs)} tickets...")
            
            # Traitement parallèle si activé
            if self.use_parallel and len(tickets_sans_vecteurs) > 10:
                from concurrent.futures import ThreadPoolExecutor
                
                def process_batch(batch):
                    embeddings_batch = []
                    for ticket in batch:
                        problem_text = ticket.get('problem', '')
                        keywords = ticket.get('keywords', '')
                        combined_text = f"{problem_text} {keywords}"
                        embedding_vector = self.model.encode(combined_text).tolist()
                        
                        embeddings_batch.append({
                            'ticket_id': ticket.get('_id'),
                            'embedding_vector': embedding_vector,
                            'last_updated': time.time()
                        })
                    return embeddings_batch
                
                # Diviser en sous-lots pour le traitement parallèle
                batches = [tickets_sans_vecteurs[i:i+self.batch_size] 
                          for i in range(0, len(tickets_sans_vecteurs), self.batch_size)]
                
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    results = list(executor.map(process_batch, batches))
                
                # Aplatir les résultats et insérer
                all_embeddings = [item for sublist in results for item in sublist]
                
                # Insertion en une seule opération
                if all_embeddings:
                    # Utiliser insert_many avec ordered=False pour plus de performance
                    self.vecteurs_collection.insert_many(all_embeddings, ordered=False)
                
                logger.info(f"✅ {len(all_embeddings)} embeddings générés et insérés en parallèle")
                
            else:
                # Traitement séquentiel traditionnel
                for i in range(0, len(tickets_sans_vecteurs), self.batch_size):
                    batch = tickets_sans_vecteurs[i:i+self.batch_size]
                    embeddings_batch = []
                    
                    for ticket in batch:
                        problem_text = ticket.get('problem', '')
                        keywords = ticket.get('keywords', '')
                        combined_text = f"{problem_text} {keywords}"
                        embedding_vector = self.model.encode(combined_text).tolist()
                        
                        embeddings_batch.append({
                            'ticket_id': ticket.get('_id'),
                            'embedding_vector': embedding_vector,
                            'last_updated': time.time()
                        })
                    
                    # Insertion par lots
                    if embeddings_batch:
                        self.vecteurs_collection.insert_many(embeddings_batch)
                    
                    logger.info(f"✅ Lot {i//self.batch_size + 1}/{(len(tickets_sans_vecteurs)-1)//self.batch_size + 1} traité")
            
            return True
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération des embeddings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def rechercher_tickets_similaires(self, ticket_text):
        """
        Recherche optimisée des tickets similaires en utilisant les embeddings
        """
        try:
            start_time = time.time()
            
            # S'assurer que tous les tickets ont des embeddings
            if not self.verifier_vecteurs_cache() or not self.generer_embeddings_manquants():
                return {"status": "error", "message": "❌ Impossible de générer les embeddings"}
            
            # Formater la requête
            query_metadata = self.formater_requete(ticket_text)
            problem_text = query_metadata["problem"]
            keywords = query_metadata.get("keywords", "")
            combined_text = f"{problem_text} {keywords}"
            
            logger.info(f"🔍 Recherche avec la problématique: {problem_text[:100]}...")
            
            # Calculer l'embedding de la requête
            query_embedding = self.model.encode(combined_text).tolist()
            
            # Amélioration: utiliser une approche de recherche vectorielle optimisée
            # Récupérer tous les vecteurs en une seule requête avec projection
            vecteurs = list(self.vecteurs_collection.find({}, {"ticket_id": 1, "embedding_vector": 1, "_id": 0}))
            
            # Préparer les matrices pour le calcul de similarité
            embeddings_matrix = [vec.get('embedding_vector', []) for vec in vecteurs]
            ticket_ids = [vec.get('ticket_id', '') for vec in vecteurs]
            
            # Calculer toutes les similarités en une seule opération
            similarities = cosine_similarity([query_embedding], embeddings_matrix).flatten()
            
            # Créer un tableau de correspondances (id, score)
            similarity_pairs = list(zip(ticket_ids, similarities))
            
            # Filtrer et trier par score
            filtered_pairs = [(tid, score) for tid, score in similarity_pairs if score >= self.similarity_threshold]
            filtered_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Limiter aux top N résultats
            top_pairs = filtered_pairs[:self.top_n_results]
            
            # Récupérer les détails des tickets correspondants
            found_tickets = []
            if top_pairs:
                top_ids = [pair[0] for pair in top_pairs]
                scores_by_id = {pair[0]: pair[1] for pair in top_pairs}
                
                # Obtenir les détails complets des tickets en une seule requête
                tickets_details = list(self.tickets_collection.find({"_id": {"$in": top_ids}}))
                
                for ticket in tickets_details:
                    found_tickets.append({
                        "ticket_id": str(ticket.get("_id", "")),
                        "problem": ticket.get("problem", "Non spécifié"),
                        "solution": ticket.get("solution", "Non spécifié"),
                        "keywords": ticket.get("keywords", ""),
                        "similarity_score": float(scores_by_id.get(ticket.get("_id"), 0))
                    })
                
                # Rétablir l'ordre par score
                found_tickets.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            search_time = time.time() - start_time
            logger.info(f"⏱ Temps de recherche: {search_time:.4f} secondes")
            
            if found_tickets:
                # Stocker tous les détails dans un seul document MongoDB
                resultats_document = {
                    "timestamp": time.time(),
                    "nouveau_probleme": query_metadata["problem"],
                    "resultats": found_tickets,
                    "temps_recherche": search_time
                }
                self.resultats_collection.insert_one(resultats_document)
                
                logger.info(f"✅ {len(found_tickets)} tickets similaires trouvés")
                for ticket in found_tickets[:3]:
                    logger.info(f" - ID: {ticket['ticket_id']}, Score: {ticket.get('similarity_score', 0):.3f}")
                
                return {
                    "status": "success",
                    "message": f"✅ {len(found_tickets)} tickets similaires trouvés en {search_time:.4f} secondes.",
                    "tickets": found_tickets,
                    "temps_recherche": search_time,
                    "query": query_metadata["problem"]
                }
            else:
                return {
                    "status": "not_found",
                    "message": "❌ Aucun ticket suffisamment similaire trouvé.",
                    "temps_recherche": search_time
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": f"❌ Erreur: {str(e)}"}

    def traiter_fichier_excel(self):
        """
        Traite le fichier Excel configuré contenant des tickets à rechercher.
        """
        try:
            if not pd.io.common.is_file_like(self.fichier_excel) and not pd.io.common.is_url(self.fichier_excel):
                if not os.path.exists(self.fichier_excel):
                    logger.error(f"❌ Le fichier {self.fichier_excel} n'existe pas.")
                    return []
                    
            df = pd.read_excel(self.fichier_excel)
            resultats = []
            
            logger.info(f"📑 Traitement de {len(df)} tickets depuis le fichier Excel...")
            total_search_time = 0
            
            # Traitement parallèle si activé et plus de 5 tickets à traiter
            if self.use_parallel and len(df) > 5:
                from concurrent.futures import ThreadPoolExecutor
                
                def process_ticket(row_data):
                    idx, row = row_data
                    logger.info(f"🔍 Analyse du ticket #{idx+1}...")
                    # Combiner toutes les colonnes non-nulles pour créer le texte du ticket
                    raw_ticket_text = " ".join(str(value) for value in row.values if pd.notna(value))
                    # Rechercher des tickets similaires avec embeddings
                    return self.rechercher_tickets_similaires(raw_ticket_text)
                
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # Convertir le DataFrame en liste d'index et lignes
                    row_list = list(df.iterrows())
                    # Exécuter en parallèle
                    resultats = list(executor.map(process_ticket, row_list))
                
            else:
                # Traitement séquentiel traditionnel
                for idx, row in df.iterrows():
                    logger.info(f"🔍 Analyse du ticket #{idx+1}...")
                    # Combiner toutes les colonnes non-nulles pour créer le texte du ticket
                    raw_ticket_text = " ".join(str(value) for value in row.values if pd.notna(value))
                    # Rechercher des tickets similaires avec embeddings
                    resultat = self.rechercher_tickets_similaires(raw_ticket_text)
                    resultats.append(resultat)
            
            # Calculer les statistiques
            total_search_time = sum(r.get("temps_recherche", 0) for r in resultats)
            avg_search_time = total_search_time / len(df) if len(df) > 0 else 0
            
            logger.info(f"🎯 Traitement terminé. {len(resultats)} tickets analysés.")
            logger.info(f"⏱️ Temps moyen de recherche par ticket: {avg_search_time:.4f} secondes")
           
            # Calculer des statistiques
            succes = sum(1 for r in resultats if r.get("status") == "success")
            taux_succes = (succes / len(resultats)) * 100 if resultats else 0
            logger.info(f"📊 Taux de succès: {taux_succes:.1f}% ({succes}/{len(resultats)})")
           
            # Sauvegarder les résultats dans un fichier CSV
            self.sauvegarder_resultats_csv(resultats)
           
            return resultats
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du fichier Excel: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def sauvegarder_resultats_csv(self, resultats):
        """
        Sauvegarde les résultats de recherche dans un fichier CSV
        """
        if not resultats:
            logger.warning("⚠ Aucun résultat à sauvegarder.")
            return
           
        resultats_simples = []
        for r in resultats:
            if r.get("status") == "success" and "tickets" in r:
                for ticket in r["tickets"]:
                    resultats_simples.append({
                        "probleme_original": r.get("query", ""),
                        "ticket_similaire_id": ticket["ticket_id"],
                        "probleme_similaire": ticket["problem"],
                        "solution_proposee": ticket["solution"],
                        "score_similarite": ticket["similarity_score"],
                        "temps_recherche": r.get("temps_recherche", 0)
                    })
       
        if resultats_simples:
            df_resultats = pd.DataFrame(resultats_simples)
            output_file = "resultats_recherche_embeddings.csv"
            df_resultats.to_csv(output_file, index=False)
            logger.info(f"✅ Résultats sauvegardés dans '{output_file}'")
        else:
            logger.warning("⚠ Aucun résultat de similarité à sauvegarder.")

def main():
    try:
        recherche = RechercheTicketsEmbeddingsOptimized()
        # Vérifier et générer les embeddings si nécessaire
        if recherche.verifier_vecteurs_cache() or recherche.generer_embeddings_manquants():
            print(f"\n📑 Base de données prête pour les recherches")
        else:
            print("⚠ Vérifiez que la base contient des tickets avant d'effectuer des recherches.")
        
        # Return the instance so it can be used by the FastAPI server
        return recherche
            
    except Exception as e:
        print(f"❌ Erreur principale: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()