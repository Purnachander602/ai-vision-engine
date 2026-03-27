import bcrypt
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["ai_vision_engine"]
users = db["users"]


def add_user(email, password):

    existing = users.find_one({"email": email})

    if existing:
        return False

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    users.insert_one({
        "email": email,
        "password": hashed,
        "chat_id": None
    })

    return True


def login_user(email, password):

    user = users.find_one({"email": email})

    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return user

    return None


def update_chat_id(email, chat_id):

    users.update_one(
        {"email": email},
        {"$set": {"chat_id": chat_id}}
    )


def get_chat_id(email):

    user = users.find_one({"email": email})

    if user:
        return user.get("chat_id")

    return None