from pymongo import MongoClient
import os
# Hardcoded MongoDB config (adjust if needed)
from dotenv import load_dotenv
load_dotenv()

DB_NAME = "hackrx"
COLLECTION_NAME = "policy_clauses"
MONGO_URI = os.getenv("MONGO_URI")

def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def fetch_all_clauses():
    """
    Fetch all clauses from MongoDB.
    Each document must have a 'clause' field.
    """
    collection = get_mongo_collection()
    return list(collection.find({}, {"_id": 0, "clause": 1}))

def save_clauses_to_mongo(clauses: list[str]):
    """
    Save a list of clause strings into MongoDB.
    """
    collection = get_mongo_collection()
    docs = [{"clause": clause} for clause in clauses]
    collection.insert_many(docs)
