import os
import numpy as np
import faiss
from pymongo import MongoClient
# REMOVE: from sentence_transformers import SentenceTransformer
from app.extract_clauses import extract_clauses_from_url
from dotenv import load_dotenv
import google.generativeai as genai # ADDED

load_dotenv()

# --- Gemini Embedding Configuration ---
# IMPORTANT: Use an environment variable for your API key
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API") # Using your existing env var name
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API environment variable not set. Please set it.")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # Exit or handle the error appropriately in a production environment
    exit()

# Gemini Embedding model typically outputs 768 or 3072 dimensions.
# Let's use 3072 for gemini-embedding-001 as the default
EMBEDDING_DIM = 3072 # Updated dimension for gemini-embedding-001
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001" # Or models/text-embedding-004 if preferred

# --- Helper function to get Gemini embeddings ---
async def get_gemini_embeddings_async(texts: list[str], task_type: str = "SEMANTIC_SIMILARITY"):
    """
    Generates embeddings for a list of texts using the Gemini API.
    Handles batching for efficiency.
    """
    if not texts:
        return np.array([])

    embeddings = []
    try:
        # The Gemini API's batch_embed_contents is efficient.
        # However, for very large lists, you might want to manually chunk them
        # if you hit API limits or timeout issues with very large batches.
        response = await genai.batch_embed_contents_async( # Use async version
            model=GEMINI_EMBEDDING_MODEL,
            contents=texts,
            task_type=task_type
        )
        embeddings = [item.embedding.values for item in response.embeddings]
        return np.array(embeddings).astype("float32")
    except Exception as e:
        print(f"Error generating Gemini embeddings: {e}")
        return np.array([]) # Return empty array on error

# Load embedding model - NOW REMOVED, using API instead
# model = SentenceTransformer("all-MiniLM-L6-v2")


MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["hackrx"]
collection = db["policy_clauses"]

# FAISS index
# Initialize with the correct dimension for Gemini embeddings
index = faiss.IndexFlatL2(EMBEDDING_DIM)

async def embed_clauses(clauses): # Changed to async
    texts = [clause['clause'] for clause in clauses]
    embeddings = await get_gemini_embeddings_async(texts, task_type="RETRIEVAL_DOCUMENT") # Use RETRIEVAL_DOCUMENT for clauses
    return embeddings

async def preload(url): # Changed to async
    clauses = extract_clauses_from_url(url)
    embeddings = await embed_clauses(clauses) # Await the async function

    if embeddings.size == 0:
        print("No embeddings generated, skipping FAISS addition and MongoDB insertion.")
        return

    index.add(embeddings) # embeddings is already np.array here

    # To ensure atomicity and correct mapping, you might want to clear and re-insert
    # or use updates based on unique identifiers if you plan to update existing clauses.
    # For now, assuming fresh load:
    collection.delete_many({}) # Clear existing clauses before preloading new ones
    
    # Store faiss_id along with clause for easy retrieval
    # Ensure faiss_id calculation is correct after clearing/reloading
    current_faiss_ntotal = index.ntotal
    for i, clause in enumerate(clauses):
        collection.insert_one({
            "faiss_id": current_faiss_ntotal - len(clauses) + i,
            "clause": clause['clause']
        })

    # Save FAISS index
    faiss.write_index(index, "app/data/faiss.index")

    print(f"✅ Loaded {len(clauses)} clauses into FAISS and MongoDB.")

# Example (will need to be run with asyncio)
if __name__ == "__main__":
    import asyncio
    TEST_URL = "https://www.google.com" # Replace with a real URL to test
    # Ensure app/data directory exists
    os.makedirs("app/data", exist_ok=True)
    asyncio.run(preload(TEST_URL)) # Run the async preload function