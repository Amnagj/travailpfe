import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import re
import json
import requests
import nltk
from sentence_transformers import SentenceTransformer

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pymongo import MongoClient
from nettoyage import nettoyer_texte 
from traitement_ai import extract_ticket_info
import logging
# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LM_STUDIO_API = "http://localhost:1234/v1/chat/completions"
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
        "similarity_threshold": 0.25,
        "top_n_results": 5,
        "use_ai_refinement": True , 


    }
}



# Renommez la classe pour refléter l'utilisation des embeddings
class RechercheTicketsEmbeddings:
    def __init__(self, config=CONFIG):
        """
        Initialise la connexion à MongoDB et prépare le modèle d'embeddings.
        """
        try:
            # Stocker les paramètres de configuration
            self.config = config
            mongodb_config = config["MONGODB"]
            search_params = config["SEARCH_PARAMS"]
            # Ajouter cette ligne pour définir l'attribut fichier_excel
            self.fichier_excel = config["PATHS"]["excel_input"]
            
            # Connexion à MongoDB
            self.client = MongoClient(mongodb_config["uri"])
            self.db = self.client[mongodb_config["db_name"]]
            self.tickets_collection = self.db[mongodb_config["collections"]["tickets"]]
            self.resultats_collection = self.db[mongodb_config["collections"]["resultats"]]
            self.vecteurs_collection = self.db[mongodb_config["collections"]["vecteurs"]]
            self.use_ai_refinement = search_params.get("use_ai_refinement", True)
            
            # Initialiser le modèle d'embeddings
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_matrix = None
            self.tickets_db = None
            
            # Initialiser le stemmer et les stop words
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english') + stopwords.words('french'))
            
            # Configurer les paramètres de recherche
            self.max_tickets = search_params.get("max_tickets")
            self.use_cached_vectors = search_params.get("use_cached_vectors", True)
            self.similarity_threshold = search_params.get("similarity_threshold", 0.25)
            self.top_n_results = search_params.get("top_n_results", 5)
            
            print(f"✅ Connexion réussie à MongoDB, base '{mongodb_config['db_name']}'")
            count = self.tickets_collection.count_documents({})
            print(f"📊 Nombre de tickets résolus dans la base: {count}")
            
            logger.info(f"✅ Connexion réussie à MongoDB, base '{mongodb_config['db_name']}'")
            count = self.tickets_collection.count_documents({})
            logger.info(f"📊 Nombre de tickets résolus dans la base: {count}")
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
            print(f"❌ Erreur d'initialisation: {e}")
            raise
    def verifier_vecteurs_cache(self):
        """
        Vérifie si des embeddings sont déjà stockés dans MongoDB
        """
        count = self.vecteurs_collection.count_documents({"embedding_vector": {"$exists": True}})
        tickets_count = self.tickets_collection.count_documents({})
        # Vérifie si le nombre de vecteurs correspond au nombre de tickets
        if count > 0 and count == tickets_count:
            print(f"✅ Utilisation de {count} embeddings en cache")
            return True
        return False
    def formater_requete(self, ticket_text):
        """
        Formate la requête de ticket avec extraction d'informations et prétraitement
        """
        try:
            # Nettoyer le texte
            cleaned_text = nettoyer_texte(ticket_text)
            
            # Extraire les informations
            extracted_info = extract_ticket_info(cleaned_text)
            
            # Prétraiter le texte
            processed_text = self.preprocess_text(cleaned_text)
            
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
                    "processed_text": processed_text  # Ajout du champ processed_text
                }
            else:
                return {
                    "problem": cleaned_text, 
                    "original_text": cleaned_text,
                    "processed_text": processed_text  # Ajout du champ processed_text
                }
        except Exception as e:
            logger.error(f"❌ Erreur formatage requête: {e}")
            # Fallback si tout échoue
            processed_text = self.preprocess_text(ticket_text)
            return {
                "problem": ticket_text, 
                "original_text": ticket_text,
                "processed_text": processed_text
            }

    def refine_similarity_with_ai(self, base_ticket, similar_tickets):
        """
        Utilise l'API LM Studio pour affiner la similarité des tickets avec analyse IA
        """
        try:
            if not similar_tickets:
                return []

            # Formater les tickets similaires pour le prompt
            tickets_text = "\n\n".join([
                f"Ticket ID: {ticket['ticket_id']}\n"
                f"Problème: {ticket['problem']}\n"
                f"Solution: {ticket['solution']}\n"
                f"Score de similarité initial: {ticket['similarity_score']}%"
                for ticket in similar_tickets
            ])

            prompt = f"""
            Tâche: Analyser la similarité sémantique entre un ticket de base et des tickets similaires.

            Instructions:
            1. Comparez chaque ticket avec le ticket de base
            2. Évaluez la similarité sémantique (0-100%)
            3. Identifiez les aspects techniques correspondants
            4. Fournissez une brève explication de pertinence

            Ticket de base:
            {base_ticket}

            Tickets similaires:
            {tickets_text}

            FORMAT DE RÉPONSE STRICTEMENT JSON:
            [
                {{
                    "ticket_id": "ID du ticket",
                    "similarite_semantique": 75,
                    "aspects_correspondants": ["aspect1", "aspect2"],
                    "explication_pertinence": "Explication concise"
                }}
            ]
            """

            payload = {
                "model": "mistral-nemo-instruct-2407",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000
            }

            logger.info("📤 Envoi de la requête à l'API IA...")
            response = requests.post(LM_STUDIO_API, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            full_response = result["choices"][0]["message"]["content"]
            
            logger.info("🔍 Réponse complète de l'IA:")
            logger.info(full_response)

            # Tentatives multiples de parsing JSON
            try:
                # Méthode 1 : Parsing JSON direct
                refined_tickets = json.loads(full_response)
                logger.info("✅ Parsing JSON direct réussi")
                return refined_tickets
            except json.JSONDecodeError:
                # Méthode 2 : Extraction JSON entre ```json et ```
                json_match = re.search(r'```json(.*?)```', full_response, re.DOTALL)
                if json_match:
                    try:
                        refined_tickets = json.loads(json_match.group(1))
                        logger.info("✅ Parsing JSON entre ``` réussi")
                        return refined_tickets
                    except json.JSONDecodeError:
                        logger.warning("❌ Parsing JSON entre ``` échoué")

                # Méthode 3 : Extraction JSON personnalisée
                logger.warning("⚠️ Tentative d'extraction JSON personnalisée")
                return self._parse_ai_response(full_response)

        except Exception as e:
            logger.error(f"❌ Erreur de raffinement par IA: {e}")
            return []

    def _parse_ai_response(self, ai_response):
        """
        Méthode de repli pour parser la réponse de l'IA
        """
        try:
            # Ajout de regex plus robuste
            refined_tickets = []
            pattern = re.compile(
                r'"ticket_id"\s*:\s*"?(\S+)"?.*'
                r'"similarite_semantique"\s*:\s*(\d+).*'
                r'"aspects_correspondants"\s*:\s*\[(.*?)\].*'
                r'"explication_pertinence"\s*:\s*"(.*?)"',
                re.DOTALL | re.IGNORECASE
            )

            matches = list(pattern.finditer(ai_response))
            
            for match in matches:
                ticket_id = match.group(1).strip('"')
                similarite = int(match.group(2))
                aspects = [aspect.strip().strip('"') for aspect in match.group(3).split(',') if aspect.strip()]
                explication = match.group(4).strip()

                refined_tickets.append({
                    "ticket_id": ticket_id,
                    "similarite_semantique": similarite,
                    "aspects_correspondants": aspects,
                    "explication_pertinence": explication
                })

            logger.info(f"✅ Extraction manuelle : {len(refined_tickets)} tickets raffinés")
            return refined_tickets

        except Exception as e:
            logger.error(f"❌ Échec de l'extraction manuelle : {e}")
            return []
    def preprocess_text(self, text):
        """
        Prétraite le texte pour la vectorisation TF-IDF
        
        Args:
            text (str): Texte à prétraiter
        
        Returns:
            str: Texte prétraité
        """
        try:
            # Convertir en minuscules
            text = text.lower()
            
            # Supprimer les caractères spéciaux et la ponctuation
            text = re.sub(r'[^\w\s]', '', text)
            
            # Tokeniser le texte
            tokens = text.split()
            
            # Supprimer les stop words
            tokens = [token for token in tokens if token not in self.stop_words]
            
            # Stemmatisation des tokens
            tokens = [self.stemmer.stem(token) for token in tokens]
            
            # Rejoindre les tokens
            processed_text = ' '.join(tokens)
            
            return processed_text
        
        except Exception as e:
            logger.error(f"❌ Erreur de prétraitement du texte: {e}")
            return text

    def verifier_vecteurs_cache(self):
        """
        Vérifie si des vecteurs TF-IDF sont déjà stockés dans MongoDB
        """
        count = self.vecteurs_collection.count_documents({})
        tickets_count = self.tickets_collection.count_documents({})
       
        # Vérifie si le nombre de vecteurs correspond au nombre de tickets
        if count > 0 and count == tickets_count:
            print(f"✅ Utilisation de {count} vecteurs TF-IDF en cache")
            return True
        return False




    def charger_tickets_depuis_mongo(self):
        """
        Charge les tickets depuis MongoDB et construit la matrice d'embeddings
        Utilise des vecteurs en cache si disponibles
        """
        try:
            start_time = time.time()
            print("📝 Chargement des tickets depuis MongoDB...")
            
            # Vérifier si on peut utiliser des vecteurs en cache
            if self.use_cached_vectors and self.verifier_vecteurs_cache():
                # Charger les vecteurs depuis la collection de vecteurs
                vecteurs_data = list(self.vecteurs_collection.find({}))
                
                # Trier par ID pour correspondre à l'ordre des tickets
                vecteurs_data.sort(key=lambda x: str(x.get('ticket_id', '')))
                
                # Charger les tickets pour les métadonnées (mais pas pour la vectorisation)
                query = {}
                if self.max_tickets:
                    cursor = self.tickets_collection.find(query).limit(self.max_tickets)
                else:
                    cursor = self.tickets_collection.find(query)
                    
                self.tickets_db = list(cursor)
                if not self.tickets_db:
                    print("⚠ Aucun ticket trouvé dans la base")
                    return False
                    
                print(f"✅ {len(self.tickets_db)} tickets chargés")
                
                # Reconstruire la matrice d'embeddings à partir des vecteurs sauvegardés
                embeddings_list = [vecteur.get('embedding_vector', []) for vecteur in vecteurs_data]
                self.embedding_matrix = np.array(embeddings_list)
                
                vectorization_time = time.time() - start_time
                print(f"⏱ Temps de chargement des embeddings: {vectorization_time:.2f} secondes")
                return True
            else:
                # Procédure standard: récupérer les tickets et calculer les embeddings
                query = {}
                if self.max_tickets:
                    cursor = self.tickets_collection.find(query).limit(self.max_tickets)
                else:
                    cursor = self.tickets_collection.find(query)
                    
                self.tickets_db = list(cursor)
                if not self.tickets_db:
                    print("⚠ Aucun ticket trouvé dans la base")
                    return False
                    
                print(f"✅ {len(self.tickets_db)} tickets chargés")
                
                # Calculer les embeddings pour chaque ticket
                print("🧠 Calcul des embeddings...")
                embeddings_list = []
                
                for ticket in self.tickets_db:
                    problem_text = ticket.get('problem', '')
                    keywords = ticket.get('keywords', '')
                    combined_text = f"{problem_text} {keywords}"
                    # Pas besoin de prétraiter le texte pour les embeddings, le modèle s'en charge
                    embedding_vector = self.model.encode(combined_text).tolist()
                    embeddings_list.append(embedding_vector)
                    
                    # Sauvegarder les embeddings dans la collection de vecteurs
                    if self.use_cached_vectors:
                        self.vecteurs_collection.update_one(
                            {'ticket_id': ticket.get('_id')},
                            {'$set': {
                                'ticket_id': ticket.get('_id'),
                                'embedding_vector': embedding_vector,
                                'last_updated': time.time()
                            }},
                            upsert=True
                        )
                
                # Convertir la liste d'embeddings en numpy array
                self.embedding_matrix = np.array(embeddings_list)
                
                vectorization_time = time.time() - start_time
                print(f"⏱ Temps de calcul des embeddings: {vectorization_time:.2f} secondes")
                
                if self.use_cached_vectors:
                    print("✅ Embeddings sauvegardés dans MongoDB pour une utilisation future")
                return True
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement des tickets: {e}")
            return False



        
    def rechercher_tickets_similaires(self, ticket_text):
        """
        Recherche les tickets similaires en utilisant les embeddings
        """
        try:
            start_time = time.time()
            if self.embedding_matrix is None:  # Modifié ici de tfidf_matrix à embedding_matrix
                success = self.charger_tickets_depuis_mongo()
                if not success:
                    return {"status": "error", "message": "❌ Impossible de charger les tickets"}
            
            query_metadata = self.formater_requete(ticket_text)
            # Pour les embeddings, nous utilisons le texte original, pas le texte prétraité
            problem_text = query_metadata["problem"]
            keywords = query_metadata.get("keywords", "")
            combined_text = f"{problem_text} {keywords}"
            
            logger.info(f"🔍 Recherche avec la problématique: {problem_text[:100]}...")
            
            # Calculer l'embedding de la requête
            query_embedding = self.model.encode(combined_text)
            
            # Calculer les similarités cosinus
            similarities = cosine_similarity([query_embedding], self.embedding_matrix).flatten()
            
            found_tickets = []
            for i, similarity in enumerate(similarities):
                if similarity >= self.similarity_threshold:
                    ticket = self.tickets_db[i]
                    found_tickets.append({
                        "ticket_id": str(ticket.get("_id", ticket.get("ID", ""))),
                        "problem": ticket.get("problem", "Non spécifié"),
                        "solution": ticket.get("solution", "Non spécifié"),
                        "keywords": ticket.get("keywords", ""),
                        "similarity_score": float(similarity)
                    })
            
            found_tickets.sort(key=lambda x: x["similarity_score"], reverse=True)
            found_tickets = found_tickets[:self.top_n_results]
            
            # Le reste de la fonction reste identique...
            # (Raffinement par IA, stockage des résultats, etc.)
            
            # Raffinement par IA si activé
            if self.use_ai_refinement and found_tickets:
                logger.info("🤖 Raffinement des résultats par IA...")
                refined_tickets = self.refine_similarity_with_ai(query_metadata["original_text"], found_tickets)
                
                # Préparer les informations de raffinement pour MongoDB
                raffinement_details = {}
                if refined_tickets:
                    for ticket in found_tickets:
                        refined_info = next((rt for rt in refined_tickets if rt["ticket_id"] == ticket["ticket_id"]), None)
                        if refined_info:
                            raffinement_details[ticket["ticket_id"]] = {
                                "semantic_similarity": refined_info.get("similarite_semantique"),
                                "corresponding_aspects": refined_info.get("aspects_correspondants"),
                                "relevance_explanation": refined_info.get("explication_pertinence")
                            }
            
            search_time = time.time() - start_time
            logger.info(f"⏱ Temps de recherche: {search_time:.4f} secondes")
            
            if found_tickets:
                # Stocker tous les détails dans un seul document MongoDB
                resultats_document = {
                    "timestamp": time.time(),
                    "nouveau_probleme": query_metadata["problem"],
                    "resultats": found_tickets,
                    "raffinement_ia": raffinement_details if self.use_ai_refinement else None,
                    "temps_recherche": search_time
                }
                self.resultats_collection.insert_one(resultats_document)
                
                logger.info(f"✅ {len(found_tickets)} tickets similaires trouvés")
                for ticket in found_tickets[:3]:
                    logger.info(f" - ID: {ticket['ticket_id']}, Score: {ticket.get('similarity_score', 'N/A')}")
                
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
            return {"status": "error", "message": f"❌ Erreur: {str(e)}"}
        



    def traiter_fichier_excel(self):
        """
        Traite le fichier Excel configuré contenant des tickets à rechercher.
        """
        try:
            if not pd.io.common.is_file_like(self.fichier_excel) and not pd.io.common.is_url(self.fichier_excel):
                if not os.path.exists(self.fichier_excel):
                    print(f"❌ Le fichier {self.fichier_excel} n'existe pas.")
                    return []
                    
            df = pd.read_excel(self.fichier_excel)
            resultats = []
            
            # S'assurer que les tickets sont chargés et vectorisés
            if self.embedding_matrix is None:  # Modifié ici de tfidf_matrix à embedding_matrix
                success = self.charger_tickets_depuis_mongo()
                if not success:
                    return []
            
            print(f"📑 Traitement de {len(df)} tickets depuis le fichier Excel...")
            total_search_time = 0
            
            for idx, row in df.iterrows():
                print(f"\n🔍 Analyse du ticket #{idx+1}...")
                # Combiner toutes les colonnes non-nulles pour créer le texte du ticket
                raw_ticket_text = " ".join(str(value) for value in row.values if pd.notna(value))
                # Rechercher des tickets similaires avec embeddings
                resultat = self.rechercher_tickets_similaires(raw_ticket_text)
                resultats.append(resultat)
                if "temps_recherche" in resultat:
                    total_search_time += resultat["temps_recherche"]
                
           
            avg_search_time = total_search_time / len(df) if len(df) > 0 else 0
            print(f"\n🎯 Traitement terminé. {len(resultats)} tickets analysés.")
            print(f"⏱️ Temps moyen de recherche par ticket: {avg_search_time:.4f} secondes")
           
            # Calculer des statistiques
            succes = sum(1 for r in resultats if r.get("status") == "success")
            taux_succes = (succes / len(resultats)) * 100 if resultats else 0
            print(f"📊 Taux de succès: {taux_succes:.1f}% ({succes}/{len(resultats)})")
           
            # Sauvegarder les résultats dans un fichier CSV
            self.sauvegarder_resultats_csv(resultats)
           
            return resultats
        except Exception as e:
            print(f"❌ Erreur lors du traitement du fichier Excel: {e}")
            return []
   
    def sauvegarder_resultats_csv(self, resultats):
        """
        Sauvegarde les résultats de recherche dans un fichier CSV
        """
        if not resultats:
            print("⚠ Aucun résultat à sauvegarder.")
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
            output_file = "resultats_recherche_tfidf.csv"
            df_resultats.to_csv(output_file, index=False)
            print(f"✅ Résultats sauvegardés dans '{output_file}'")
        else:
            print("⚠ Aucun résultat de similarité à sauvegarder.")




def main():
    try:
        recherche = RechercheTicketsEmbeddings()
        if recherche.charger_tickets_depuis_mongo():
            print(f"\n📑 Traitement du fichier: {recherche.config['PATHS']['excel_input']}")
            recherche.traiter_fichier_excel()
        else:
            print("⚠ Vérifiez que la base contient des tickets avant d'effectuer des recherches.")
    except Exception as e:
        print(f"❌ Erreur principale: {e}")

if __name__ == "__main__":
    main()
