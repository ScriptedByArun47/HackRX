import os
import numpy as np
import faiss
from pymongo import MongoClient
from app.extract_clauses import extract_clauses_from_url
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

# Load environment variables from .env
load_dotenv()

# --- Gemini Configuration ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API")
    if not GEMINI_API_KEY:
        raise ValueError("❌ GEMINI_API environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    exit()

# Embedding config
EMBEDDING_DIM = 3072  # For gemini-embedding-001
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("❌ MONGO_URI environment variable not set.")
client = MongoClient(MONGO_URI)
db = client["hackrx"]
collection = db["policy_clauses"]

# FAISS index
index = faiss.IndexFlatL2(EMBEDDING_DIM)

# --- Embedding Function ---
async def get_gemini_embeddings_async(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"):
    """Get embeddings for a list of texts using Gemini async batch API."""
    if not texts:
        return np.array([])

    try:
        response = await genai.batch_embed_contents_async(
            model=GEMINI_EMBEDDING_MODEL,
            contents=texts,
            task_type=task_type
        )
        embeddings = [item.embedding.values for item in response.embeddings]
        return np.array(embeddings).astype("float32")
    except Exception as e:
        print(f"❌ Error generating Gemini embeddings: {e}")
        return np.array([])

# --- Embed clauses ---
async def embed_clauses(clauses):
    texts = [clause['clause'] for clause in clauses]
    return await get_gemini_embeddings_async(texts, task_type="RETRIEVAL_DOCUMENT")

# --- Preload function: extracts, embeds, stores ---
async def preload(url: str):
    print(f"🌐 Preloading clauses from: {url}")
    clauses = extract_clauses_from_url(url)
    embeddings = await embed_clauses(clauses)

    if embeddings.size == 0:
        print("⚠️ No embeddings generated, skipping FAISS and MongoDB.")
        return

    index.add(embeddings)
    collection.delete_many({})  # Clean old records

    current_faiss_ntotal = index.ntotal
    for i, clause in enumerate(clauses):
        collection.insert_one({
            "faiss_id": current_faiss_ntotal - len(clauses) + i,
            "clause": clause['clause']
        })

    os.makedirs("app/data", exist_ok=True)
    faiss.write_index(index, "app/data/faiss.index")
    print(f"✅ Loaded {len(clauses)} clauses into FAISS and MongoDB.")

# --- Run from terminal ---
if __name__ == "__main__":
    TEST_URL = "https://www.google.com"  # Replace with real URL for testing
    asyncio.run(preload(TEST_URL))
