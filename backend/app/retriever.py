# app/retriever.py
import json
import faiss
import numpy as np
# REMOVE: from sentence_transformers import SentenceTransformer
import google.generativeai as genai # ADDED
import os # ADDED for API key

# --- Gemini Embedding Configuration (repeated for self-contained module, but ideally central) ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API environment variable not set. Please set it.")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API in retriever.py: {e}")
    # Handle this more gracefully in a real application

EMBEDDING_DIM = 3072 # Must match the dimension used in clause_loader.py
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

async def get_gemini_embedding_single(text: str, task_type: str = "SEMANTIC_SIMILARITY"):
    """
    Generates a single embedding for a given text using the Gemini API.
    """
    if not text:
        return np.array([])
    try:
        response = await genai.embed_content_async( # Use async version
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        return np.array(response.embedding.values).astype("float32")
    except Exception as e:
        print(f"Error generating single Gemini embedding: {e}")
        return np.array([])

CLAUSE_FILE = "app/data/clauses.json" # This file doesn't seem to be used anywhere, clauses are from Mongo or preloaded

class ClauseRetriever:
    def __init__(self):
        # REMOVED: self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Instead, we will configure genai for embeddings directly.
        
        # Load the pre-saved FAISS index. This assumes preload has been run.
        self.index = self.load_faiss_index()
        # Note: self.embeddings (the original clause embeddings) are not re-loaded here,
        # as they are implicitly part of the FAISS index and MongoDB.
        # You'll retrieve clauses from MongoDB based on FAISS IDs.
        
        # You'll need access to the MongoDB collection to map FAISS IDs back to clauses
        from app.db import get_mongo_collection
        self.mongo_collection = get_mongo_collection()


    def load_faiss_index(self):
        faiss_index_path = "app/data/faiss.index"
        if os.path.exists(faiss_index_path):
            print(f"Loading FAISS index from {faiss_index_path}")
            return faiss.read_index(faiss_index_path)
        else:
            print("FAISS index not found. Run 'python -m app.clause_loader' first to preload clauses.")
            # For a real app, you might raise an error or try to rebuild it if possible
            return faiss.IndexFlatL2(EMBEDDING_DIM) # Return an empty index

    # REMOVED: build_index method as it's now handled by clause_loader.py

    async def search(self, query: str, top_k: int = 5): # Changed to async
        # Use Gemini API for query embedding
        query_embedding = await get_gemini_embedding_single(query, task_type="RETRIEVAL_QUERY") # Use RETRIEVAL_QUERY for query
        
        if query_embedding.size == 0:
            return [] # No embedding, no search

        # FAISS search expects a 2D array, even for a single query
        D, I = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        # Retrieve clauses from MongoDB using the FAISS IDs
        found_faiss_ids = I[0].tolist()
        
        # Fetch clauses from MongoDB based on their faiss_id
        # Note: This assumes 'faiss_id' is stored in MongoDB documents
        # and that the IDs in FAISS correspond to these.
        retrieved_clauses_docs = list(self.mongo_collection.find({"faiss_id": {"$in": found_faiss_ids}}, {"_id": 0, "clause": 1, "faiss_id": 1}))

        # Reorder clauses based on the FAISS search results (I)
        # Create a dictionary for quick lookup
        clause_map = {doc["faiss_id"]: doc["clause"] for doc in retrieved_clauses_docs}
        
        # Build the ordered list using the indices from FAISS
        ordered_clauses = []
        for faiss_id in I[0]: # I[0] contains the indices in search result order
            if faiss_id in clause_map:
                ordered_clauses.append({"clause": clause_map[faiss_id]})
            # else: print(f"Warning: FAISS ID {faiss_id} not found in MongoDB.") # Debugging

        return ordered_clauses

# Example usage (needs to be run with asyncio)
if __name__ == "__main__":
    import asyncio
    # Ensure you have run clause_loader.py first to populate the index and MongoDB
    # python -m app.clause_loader (assuming `app` is your package root)

    async def main():
        retriever = ClauseRetriever()
        if retriever.index.ntotal == 0:
            print("FAISS index is empty. Please run clause_loader.py first.")
            return

        query = "What are the benefits for maternity?"
        top_clauses = await retriever.search(query, top_k=3)
        print(f"\nQuery: {query}")
        for i, clause in enumerate(top_clauses):
            print(f"Clause {i+1}: {clause['clause']}")

    asyncio.run(main())