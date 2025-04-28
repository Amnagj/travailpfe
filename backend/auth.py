# backend/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pymongo import MongoClient
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
from bson.objectid import ObjectId  
from fastapi import Depends, HTTPException, status


# Configuration de la connexion MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["Access"]
users_collection = db["Users"]  # Assurez-vous que c'est le bon nom de collection

# Configuration JWT
SECRET_KEY = os.getenv("SECRET_KEY", "your_default_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Schéma OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password, stored_password):
    # Dans une application de production, utilisez un hachage sécurisé comme bcrypt
    return plain_password == stored_password

def authenticate_user(email: str, password: str):
    user = users_collection.find_one({"email": email})
    if not user:
        return False
    if not verify_password(password, user.get("password", "")):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Informations d'identification invalides",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    try:
        # Vérifier si l'ID est un ObjectId valide
        if ObjectId.is_valid(user_id):
            object_id = ObjectId(user_id)
            user = users_collection.find_one({"_id": object_id})
        else:
            # Peut-être que l'ID est stocké sous forme de chaîne
            user = users_collection.find_one({"_id": user_id})
    except Exception as e:
        print(f"Error fetching user: {str(e)}")
        raise credentials_exception
        
    if user is None:
        raise credentials_exception
        
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "isAdmin": user.get("isAdmin", False)
    }