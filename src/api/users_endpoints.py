"""
Router pour les endpoints liés à la gestion des utilisateurs.
"""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import sys
import os
from bson.objectid import ObjectId
from pymongo import MongoClient

from backend.auth import get_current_user
client = MongoClient("mongodb://localhost:27017/")
db = client["Access"]
users_collection = db["Users"]
# Ajouter le chemin du backend au path pour pouvoir importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.user_management import create_user_endpoint, connect_to_mongodb, generate_password, send_email

class UserCreate(BaseModel):
    username: str
    email: EmailStr

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    isAdmin: bool = False
    password: Optional[str] = None
    
# Ajout du modèle manquant
class UserListResponse(BaseModel):
    status: str
    message: str = ""
    users: Optional[List[dict]] = None

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.delete("/delete/{user_id}")
async def delete_user(user_id: str):
    """
    Supprime un utilisateur par son ID
    """
    try:
        client, collection = connect_to_mongodb()
        # Convertir la chaîne ID en ObjectId
        object_id = ObjectId(user_id)
        # Vérifier si l'utilisateur existe
        user = collection.find_one({"_id": object_id})
        if not user:
            client.close()
            return {"status": "error", "message": f"Utilisateur avec ID {user_id} non trouvé"}
        # Supprimer l'utilisateur
        result = collection.delete_one({"_id": object_id})
        client.close()
        if result.deleted_count == 1:
            return {"status": "success", "message": f"Utilisateur supprimé avec succès"}
        else:
            return {"status": "error", "message": "Erreur lors de la suppression de l'utilisateur"}
    except Exception as e:
        return {"status": "error", "message": f"Erreur lors de la suppression de l'utilisateur: {str(e)}"}

@router.post("/create", response_model=UserResponse)
async def create_new_user(user_data: UserCreate = Body(...)):
    """
    Crée un nouvel utilisateur avec un mot de passe généré automatiquement
    et envoie les identifiants par email.
    """
    try:
        result = create_user_endpoint(user_data.username, user_data.email)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création de l'utilisateur: {str(e)}")

@router.get("/list", response_model=UserListResponse)
async def list_users():
    """
    Récupère la liste de tous les utilisateurs
    """
    try:
        client, collection = connect_to_mongodb()
        users = list(collection.find({}))
        # Convertir les ObjectId en string pour la sérialisation JSON
        for user in users:
            user["_id"] = str(user["_id"])
            # Ne pas renvoyer les mots de passe
            if "password" in user:
                del user["password"]
        client.close()
        return {"status": "success", "users": users}
    except Exception as e:
        return {"status": "error", "message": f"Erreur lors de la récupération des utilisateurs: {str(e)}"}

@router.post("/resend-invitation")
async def resend_invitation(data: dict = Body(...)):
    """
    Renvoie l'email d'invitation avec un nouveau mot de passe
    """
    try:
        user_id = data.get("userId")
        email = data.get("email")
        if not user_id or not email:
            return {"status": "error", "message": "ID utilisateur et email requis"}
            
        client, collection = connect_to_mongodb()
        # Convertir la chaîne ID en ObjectId
        object_id = ObjectId(user_id)
        # Vérifier si l'utilisateur existe
        user = collection.find_one({"_id": object_id})
        if not user:
            client.close()
            return {"status": "error", "message": f"Utilisateur avec ID {user_id} non trouvé"}
            
        # Générer un nouveau mot de passe
        new_password = generate_password()
        # Mettre à jour le mot de passe
        collection.update_one(
            {"_id": object_id},
            {"$set": {"password": new_password}}
        )
        
        # Envoyer l'email
        email_sent = send_email(email, user["username"], new_password)
        client.close()
        
        if email_sent:
            return {"status": "success", "message": "Email d'invitation renvoyé avec succès"}
        else:
            return {"status": "error", "message": "Erreur lors de l'envoi de l'email"}
    except Exception as e:
        return {"status": "error", "message": f"Erreur lors du renvoi de l'invitation: {str(e)}"}