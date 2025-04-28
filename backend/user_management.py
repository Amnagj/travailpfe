"""
Module de gestion des utilisateurs pour l'application Ticket AI Wizard.
Permet de créer des utilisateurs, générer des mots de passe et envoyer des emails.
"""
import secrets
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from pymongo import MongoClient
import logging
import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration MongoDB
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "Access"
COLLECTION_NAME = "Users"

# Dans user_management.py
def connect_to_mongodb():
    """Établit une connexion à la base de données MongoDB."""
    try:
        logger.info(f"Tentative de connexion à MongoDB à {MONGO_URI}")
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        logger.info(f"Connexion à MongoDB établie avec succès. Base: {DB_NAME}, Collection: {COLLECTION_NAME}")
        # Essayez de faire une requête simple pour vérifier la connexion
        count = collection.count_documents({})
        logger.info(f"Nombre d'utilisateurs dans la collection: {count}")
        return client, collection
    except Exception as e:
        logger.error(f"Erreur de connexion à MongoDB: {e}")
        raise
def generate_password(length=12):
    """
    Génère un mot de passe aléatoire et sécurisé.
    Args:
        length (int): Longueur du mot de passe (par défaut: 12)
    Returns:
        str: Mot de passe généré
    """
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    # Garantir au moins un caractère de chaque type
    password = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*")
    ]
    # Compléter avec des caractères aléatoires
    password.extend(secrets.choice(alphabet) for _ in range(length - 4))
    # Mélanger le mot de passe
    secrets.SystemRandom().shuffle(password)
    
    return ''.join(password)

def send_email(recipient_email, username, password):
    """
    Simule l'envoi d'un email à l'utilisateur avec ses identifiants.
    Pour un déploiement réel, configurez cette fonction avec vos identifiants SMTP.
    """
    try:
        # Simulation de l'envoi d'email pour le développement
        print(f"[SIMULATION EMAIL] Envoi d'un email à {recipient_email}")
        print(f"[SIMULATION EMAIL] Sujet: Vos identifiants de connexion à Ticket AI Wizard")
        print(f"[SIMULATION EMAIL] Contenu: Nom d'utilisateur: {username}, Mot de passe: {password}")
        
        # Pour un déploiement réel, décommentez le code ci-dessous et configurez-le
        """
        # Configuration SMTP
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_username = "votre_email@gmail.com"  # Remplacez par votre email
        smtp_password = "votre_mot_de_passe_app"  # Remplacez par votre mot de passe d'application Gmail
        
        # Créer le message
        msg = MIMEMultipart()
        msg["From"] = smtp_username
        msg["To"] = recipient_email
        msg["Subject"] = "Vos identifiants de connexion à Ticket AI Wizard"
        
        # Corps du message HTML
        body = f'''
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #4F46E5;">Bienvenue dans l'application Ticket AI Wizard!</h2>
            <p>Bonjour {username},</p>
            <p>Votre compte a été créé avec succès. Voici vos identifiants de connexion :</p>
            <div style="background-color: #f3f4f6; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p><strong>Nom d'utilisateur :</strong> {username}</p>
                <p><strong>Mot de passe temporaire :</strong> {password}</p>
                <p><strong>Lien de connexion :</strong> <a href="http://localhost:3000/login" style="color: #4F46E5;">Cliquez ici pour vous connecter</a></p>
            </div>
            <p>Pour des raisons de sécurité, nous vous recommandons de changer votre mot de passe après votre première connexion.</p>
            <p>Si vous avez des questions, n'hésitez pas à contacter l'administrateur.</p>
            <p>Cordialement,<br />L'équipe Ticket AI Wizard</p>
        </div>
        '''
        msg.attach(MIMEText(body, "html"))
        
        # Connexion au serveur SMTP
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Sécuriser la connexion
        server.login(smtp_username, smtp_password)
        
        # Envoyer l'email
        server.send_message(msg)
        server.quit()
        """
        
        logger.info(f"Email simulé envoyé avec succès à {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la simulation d'envoi de l'email: {e}")
        return False

def create_user(username, email):
    """
    Crée un nouvel utilisateur dans la base de données MongoDB.
    Args:
        username (str): Nom d'utilisateur
        email (str): Adresse email
    Returns:
        dict: Informations sur l'utilisateur créé, y compris le mot de passe généré
    """
    try:
        # Générer un mot de passe
        password = generate_password()
        
        # Se connecter à MongoDB
        client, collection = connect_to_mongodb()
        
        # Vérifier si l'utilisateur existe déjà
        existing_user = collection.find_one({"email": email})
        if existing_user:
            logger.warning(f"Un utilisateur avec l'email {email} existe déjà")
            client.close()
            return {
                "status": "error",
                "message": f"Un utilisateur avec l'email {email} existe déjà"
            }
            
        # Créer le nouvel utilisateur
        new_user = {
            "username": username,
            "email": email,
            "password": password,  # En production, il faudrait hacher le mot de passe
            "isAdmin": False,
            "createdAt": datetime.datetime.now()
        }
        
        # Insérer l'utilisateur dans la base de données
        result = collection.insert_one(new_user)
        user_id = str(result.inserted_id)
        
        # Envoyer l'email avec les identifiants
        email_sent = send_email(email, username, password)
        
        # Fermer la connexion
        client.close()
        
        # Retourner le résultat
        if email_sent:
            return {
                "status": "success",
                "message": f"Utilisateur {username} créé avec succès et email envoyé",
                "user": {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "isAdmin": False
                },
                "password": password  # Ne pas inclure dans un environnement de production
            }
        else:
            return {
                "status": "partial_success",
                "message": f"Utilisateur {username} créé avec succès mais erreur lors de l'envoi de l'email",
                "user": {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "isAdmin": False
                },
                "password": password  # Ne pas inclure dans un environnement de production
            }
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'utilisateur: {e}")
        return {
            "status": "error",
            "message": f"Erreur lors de la création de l'utilisateur: {str(e)}"
        }

# Endpoint pour être utilisé avec FastAPI
def create_user_endpoint(username, email):
    """
    Point d'entrée pour la création d'utilisateur via l'API.
    Args:
        username (str): Nom d'utilisateur
        email (str): Adresse email
    Returns:
        dict: Résultat de l'opération
    """
    return create_user(username, email)