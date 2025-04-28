""" 
FastAPI server file for integration with the React frontend. 
This file serves as a template for implementing the backend API. 
To run this server, install FastAPI and uvicorn: 
    pip install fastapi uvicorn python-multipart 
Then run: 
    uvicorn fastapi_server:app --reload 
""" 

from fastapi import Body 
from fastapi.security import OAuth2PasswordRequestForm 
from datetime import timedelta 
from backend.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES 
from fastapi.security import OAuth2PasswordRequestForm 
from backend.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES 
from datetime import timedelta 
from fastapi import FastAPI, UploadFile, File, HTTPException 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, validator 
from fastapi import FastAPI, UploadFile, File, HTTPException, status 

from typing import List, Optional 
import os 
import tempfile 
import shutil 
import sys 
import time 
from backend.history_management import add_history_item, get_user_history, hide_history_item, clear_user_history 
from fastapi import Depends 
from backend.auth import get_current_user 
# Import the ticket similarity search functionality 
# sys.path.append('/path/to/your/python/scripts') 
# from embeddings_ai_optimisé import RechercheTicketsEmbeddingsOptimized 
# from main import process_file  # Import file processing function 

from fastapi import Depends, HTTPException, status 
from fastapi.security import OAuth2PasswordBearer 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False) 

def get_current_user_optional(token: str = Depends(oauth2_scheme)): 
    if not token: 
        return None 
    try: 
        return get_current_user(token) 
    except: 
        return None 


app = FastAPI( 
    title="Ticket AI API", 
    description="API for processing and analyzing support tickets", 
    version="1.0.0", 
) 

# Configure CORS to allow requests from the frontend 
app.add_middleware( 
    CORSMiddleware, 
    allow_origins=["*"],  # Your frontend URL 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
) 

class TicketData(BaseModel): 
    ID: str 
    problem: str 
    solution: str 
    keywords: str 
    status: Optional[str] = None
    
    @validator('status') 
    def validate_status(cls, v): 
        if v and v.lower() != 'fixed': 
            raise ValueError("Seuls les tickets avec le statut 'Fixed' sont acceptés") 
        return v 

class TicketSearchRequest(BaseModel): 
    ticket_text: str 

class TicketResponse(BaseModel): 
    ticket_id: str 
    problem: str 
    solution: str 
    keywords: str 
    similarity_score: float 

class SearchResponse(BaseModel): 
    status: str 
    message: str 
    tickets: Optional[List[TicketResponse]] = None 
    temps_recherche: Optional[float] = None 
    query: Optional[str] = None 

class HistoryResponse(BaseModel): 
    status: str 
    history: Optional[List[dict]] = None 
    message: Optional[str] = None 

class DeleteHistoryRequest(BaseModel): 
    item_id: str 

class Token(BaseModel): 
    access_token: str 
    token_type: str 
    user: dict 

@app.post("/token") 
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()): 
    user = authenticate_user(form_data.username, form_data.password) 
    if not user: 
        raise HTTPException( 
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Email ou mot de passe incorrect", 
            headers={"WWW-Authenticate": "Bearer"}, 
        ) 
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES) 
    access_token = create_access_token( 
        data={"sub": str(user["_id"])}, 
        expires_delta=access_token_expires 
    ) 
    return { 
        "access_token": access_token, 
        "token_type": "bearer", 
        "user": { 
            "id": str(user["_id"]), 
            "username": user["username"], 
            "email": user["email"], 
            "isAdmin": user.get("isAdmin", False) 
        } 
    } 

@app.post("/search-tickets", response_model=SearchResponse) 
async def search_tickets(request: TicketSearchRequest): 
    """ 
    Search for tickets similar to the provided ticket text. 
    Utilizes embedding-based similarity search to find relevant tickets. 
    """ 
    try: 
        start_time = time.time() 
        
        # In production, uncomment and use the actual search implementation 
        # search_engine = RechercheTicketsEmbeddingsOptimized() 
        # result = search_engine.rechercher_tickets_similaires(request.ticket_text) 
        # return result 
        
        # Mock implementation for demonstration 
        # Return a successful response with mock data 
        elapsed_time = time.time() - start_time 
        
        # Simulate finding a ticket 
        return { 
            "status": "success", 
            "message": "1 ticket similaire trouvé", 
            "tickets": [ 
                { 
                    "ticket_id": "MOCK-123", 
                    "problem": "Problème de connexion à la base de données", 
                    "solution": "Vérifier les paramètres de connexion et redémarrer le service", 
                    "keywords": "connexion, database, paramètres", 
                    "similarity_score": 0.85 
                } 
            ], 
            "temps_recherche": elapsed_time, 
            "query": "Erreur lors de l'accès à la base de données" 
        } 
    except Exception as e: 
        return { 
            "status": "error", 
            "message": f"Une erreur s'est produite: {str(e)}" 
        } 

@app.post("/upload-file", response_model=SearchResponse) 
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user_optional)): 
    try: 
        # Créer un fichier temporaire 
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp: 
            # Copier le fichier uploadé dans le fichier temporaire 
            shutil.copyfileobj(file.file, tmp) 
            temp_file_path = tmp.name 
            
        print(f"Fichier temporaire créé: {temp_file_path}") 
        
        try: 
            # Vérifier que le fichier existe 
            if not os.path.exists(temp_file_path): 
                raise HTTPException( 
                    status_code=500, 
                    detail="Le fichier temporaire n'a pas été correctement créé" 
                ) 
                
            # Essayer de lire le fichier avec pandas d'abord pour vérifier qu'il est lisible 
            import pandas as pd 
            try: 
                df = pd.read_excel(temp_file_path) 
                print(f"Fichier Excel lu avec succès, {len(df)} lignes trouvées") 
            except Exception as excel_error: 
                print(f"Erreur lors de la lecture du fichier Excel: {excel_error}") 
                raise HTTPException( 
                    status_code=500, 
                    detail=f"Erreur lors de la lecture du fichier Excel: {str(excel_error)}" 
                ) 
            
            # Importer la classe de recherche d'embeddings avec gestion d'erreurs 
            try: 
                from backend.embeddings_final import RechercheTicketsEmbeddingsOptimized 
                print("Module embeddings_final importé avec succès") 
            except ImportError as import_error: 
                print(f"Erreur d'importation du module embeddings_final: {import_error}") 
                # Nettoyer et renvoyer une erreur 
                os.unlink(temp_file_path) 
                raise HTTPException( 
                    status_code=500, 
                    detail=f"Erreur d'importation du module embeddings_final: {str(import_error)}" 
                ) 
            
            # Initialiser le moteur de recherche avec le chemin du fichier 
            try: 
                from backend.embeddings_final import CONFIG 
                custom_config = CONFIG.copy()  # Copier la configuration par défaut 
                custom_config["PATHS"]["excel_input"] = temp_file_path  # Mettre à jour le chemin du fichier Excel 
                recherche = RechercheTicketsEmbeddingsOptimized(config=custom_config) 
                print("Instance de RechercheTicketsEmbeddingsOptimized créée") 
            except Exception as instance_error: 
                print(f"Erreur lors de la création de l'instance RechercheTicketsEmbeddingsOptimized: {instance_error}") 
                # Nettoyer et renvoyer une erreur 
                os.unlink(temp_file_path) 
                raise HTTPException( 
                    status_code=500, 
                    detail=f"Erreur lors de l'initialisation du moteur de recherche: {str(instance_error)}" 
                ) 

            # Lire le contenu du fichier Excel avec gestion d'erreurs 
            try: 
                resultats = recherche.traiter_fichier_excel()  # Utiliser la méthode qui traite le fichier 
                print(f"Fichier Excel traité avec succès, {len(resultats) if resultats else 0} résultats trouvés") 
            except Exception as process_error: 
                print(f"Erreur lors du traitement du fichier Excel: {process_error}") 
                # Nettoyer et renvoyer une erreur 
                os.unlink(temp_file_path) 
                raise HTTPException( 
                    status_code=500, 
                    detail=f"Erreur lors du traitement du fichier Excel: {str(process_error)}" 
                ) 

            # Nettoyer le fichier temporaire 
            os.unlink(temp_file_path) 
            print("Fichier temporaire supprimé") 
            
            if current_user and resultats and len(resultats) > 0: 
                try: 
                    # Utiliser la fonction add_history_item pour enregistrer dans MongoDB 
                    add_history_item( 
                        user_id=current_user["id"], 
                        query_text=file.filename,  # Utiliser le nom du fichier comme requête 
                        result=resultats[0].get("message", ""), 
                        ticket_ids=[ticket["ticket_id"] for ticket in resultats[0].get("tickets", [])] if resultats[0].get("tickets") else [] 
                    ) 
                    print("Résultat ajouté à l'historique avec succès") 
                except Exception as hist_error: 
                    print(f"Erreur lors de l'ajout à l'historique: {hist_error}") 
            
                return resultats[0] 
            else: 
                print("Aucun résultat trouvé") 
                return { 
                    "status": "not_found", 
                    "message": "Aucun ticket similaire trouvé" 
                } 
                
        except Exception as e: 
            print(f"Erreur générale lors du traitement: {e}") 
            if os.path.exists(temp_file_path): 
                os.unlink(temp_file_path) 
            raise HTTPException( 
                status_code=500, 
                detail=f"Erreur lors du traitement du fichier: {str(e)}" 
            ) 
            
    except Exception as e: 
        print(f"Erreur lors du téléchargement: {e}") 
        # Si le fichier temporaire existe, le nettoyer 
        if 'temp_file_path' in locals(): 
            try: 
                os.unlink(temp_file_path) 
            except: 
                pass 
        raise HTTPException( 
            status_code=500, 
            detail=f"Erreur lors du téléchargement du fichier: {str(e)}" 
        ) 

@app.get("/ticket-stats") 
async def ticket_stats(): 
    """ 
    Get statistics about the tickets stored in MongoDB. 
    """ 
    try: 
        # In production, uncomment and use the actual function 
        # from stockage import get_collection_stats 
        # stats = get_collection_stats() 
        # return stats 
        
        # Mock implementation 
        return { 
            "count": 150,  # Mock number of tickets 
            "status": "active", 
            "resolved_count": 120, 
            "unresolved_count": 30, 
            "categories": { 
                "login": 45, 
                "database": 38, 
                "ui": 25, 
                "other": 42 
            } 
        } 
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des statistiques: {str(e)}") 

@app.post("/search-history/add", response_model=dict) 
async def add_search_to_history( 
    data: dict = Body(...), 
    current_user: dict = Depends(get_current_user) 
): 
    """ 
    Ajoute une recherche à l'historique de l'utilisateur 
    """ 
    try: 
        query_text = data.get("ticket_text", "") 
        result = data.get("result", {}) 
        
        ticket_ids = [ticket["ticket_id"] for ticket in result.get("tickets", [])] if result.get("tickets") else [] 
        
        history_item = add_history_item( 
            user_id=current_user["id"], 
            query_text=query_text, 
            result=result.get("message", ""), 
            ticket_ids=ticket_ids 
        ) 
        return history_item 
    except Exception as e: 
        return { 
            "status": "error", 
            "message": f"Erreur lors de l'ajout à l'historique: {str(e)}" 
        } 

@app.get("/search-history", response_model=HistoryResponse) 
async def get_search_history(current_user: dict = Depends(get_current_user)): 
    """ 
    Récupère l'historique de recherche de l'utilisateur 
    """ 
    try: 
        return get_user_history(current_user["id"]) 
    except Exception as e: 
        return { 
            "status": "error", 
            "message": f"Erreur lors de la récupération de l'historique: {str(e)}" 
        } 

@app.post("/search-history/hide", response_model=dict) 
async def hide_from_history( 
    request: DeleteHistoryRequest, 
    current_user: dict = Depends(get_current_user) 
): 
    """ 
    Masque un élément de l'historique 
    """ 
    try: 
        return hide_history_item(request.item_id) 
    except Exception as e: 
        return { 
            "status": "error", 
            "message": f"Erreur lors du masquage de l'élément: {str(e)}" 
        } 

@app.post("/search-history/clear", response_model=dict) 
async def clear_history(current_user: dict = Depends(get_current_user)): 
    """ 
    Masque tous les éléments de l'historique d'un utilisateur 
    """ 
    try: 
        return clear_user_history(current_user["id"]) 
    except Exception as e: 
        return { 
            "status": "error", 
            "message": f"Erreur lors de l'effacement de l'historique: {str(e)}" 
        } 

@app.post("/validate-excel") 
async def validate_excel(file: UploadFile = File(...)): 
    """ 
    Validate that the Excel file is in the correct format for processing. 
    """ 
    try: 
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp: 
            shutil.copyfileobj(file.file, tmp) 
            temp_file_path = tmp.name 
        
        try: 
            import pandas as pd 
            df = pd.read_excel(temp_file_path) 
            
            # Vérifier que le fichier n'est pas vide 
            if df.empty: 
                return {"isValid": False, "message": "Le fichier Excel est vide."} 
            
            # Vous pouvez ajouter d'autres validations spécifiques ici 
            # Par exemple, vérifier la présence de certaines colonnes 
            
            os.unlink(temp_file_path) 
            return {"isValid": True, "message": "Le fichier Excel est valide."} 
            
        except Exception as e: 
            os.unlink(temp_file_path) 
            return {"isValid": False, "message": f"Erreur lors de la lecture du fichier Excel: {str(e)}"} 
    except Exception as e: 
        if 'temp_file_path' in locals(): 
            try: 
                os.unlink(temp_file_path) 
            except: 
                pass 
        return {"isValid": False, "message": f"Erreur lors de la validation du fichier: {str(e)}"} 

@app.post("/messages/record") 
async def record_message(message_data: dict = Body(...), current_user: dict = Depends(get_current_user)): 
    """ 
    Enregistre un message dans l'historique 
    """ 
    try: 
        # Récupérer les données du message 
        message_text = message_data.get("message_text", "") 
        ticket_ids = message_data.get("ticket_ids", []) 
        
        # Ajouter le message à l'historique en utilisant la fonction existante 
        result = add_history_item( 
            user_id=current_user["id"], 
            query_text=message_text, 
            result=message_data.get("result", ""), 
            ticket_ids=ticket_ids 
        ) 
        
        return result 
    except Exception as e: 
        raise HTTPException( 
            status_code=500, 
            detail=f"Erreur lors de l'enregistrement du message: {str(e)}" 
        ) 

@app.post("/telecharger-excel") 
async def telecharger_excel(file: UploadFile = File(...)): 
    """ 
    Upload et traitement d'un fichier Excel contenant des tickets 
    depuis l'interface admin. Seuls les tickets avec le statut "Fixed" 
    seront traités et stockés dans MongoDB. 
    """ 
    try: 
        # Créer un fichier temporaire 
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp: 
            # Copier le fichier uploadé dans le fichier temporaire 
            shutil.copyfileobj(file.file, tmp) 
            temp_file_path = tmp.name 
            print(f"📄 Fichier temporaire créé: {temp_file_path}") 
        
        # Importer le main avec le chemin correct 
        from backend.main import main 
        
        # Appeler la fonction de traitement avec le chemin du fichier 
        stats = main(temp_file_path) 
 
        # Nettoyer le fichier temporaire 
        try: 
            os.unlink(temp_file_path) 
            print("🧹 Fichier temporaire supprimé") 
        except Exception as e: 
            print(f"⚠ Erreur lors de la suppression du fichier temporaire: {e}") 
        
        # Retourner la réponse avec les résultats 
        return { 
            "status": "success", 
            "message": f"Le fichier {file.filename} a été traité et importé avec succès", 
            "processed_tickets": stats.get("processed", 0), 
            "fixed_tickets": stats.get("fixed", 0), 
            "skipped_tickets": stats.get("skipped", 0), 
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") 
        } 
    except Exception as e: 
        # Si le fichier temporaire a été créé, le nettoyer 
        if 'temp_file_path' in locals(): 
            try: 
                os.unlink(temp_file_path) 
            except: 
                pass 
        import traceback 
        error_details = traceback.format_exc() 
        print(f"❌ Erreur détaillée: {error_details}") 
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du fichier: {str(e)}") 

# Root endpoint for API health check 
@app.get("/") 
async def root(): 
    return {"status": "API is running", "version": "1.0.0"}