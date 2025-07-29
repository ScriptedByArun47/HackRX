import os
import json
import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any

# --- Load environment variables ---
from dotenv import load_dotenv
load_dotenv()

# --- Configure Gemini ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API")
    if not GEMINI_API_KEY:
        raise ValueError("❌ GEMINI_API environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"❌ Error configuring Gemini API in retriever.py: {e}")
    exit()

# --- Embedding Model Settings ---
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIM = 3072


# --- Single Query Embedding ---
async def get_gemini_embedding_single(text: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    if not text:
        return np.array([])
    try:
        response = await genai.embed_content_async(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        return np.array(response.embedding.values).astype("float32")
    except Exception as e:
        print(f"❌ Error generating Gemini embedding: {e}")
        return np.array([])


# --- Clause Retriever Class ---
class ClauseRetriever:
    """
    Encapsulates FAISS-based clause retrieval using Gemini embeddings and MongoDB.
    """
    def __init__(self):
        self.index = self._load_faiss_index()
        self.embedding_dim = self.index.d if self.index else DEFAULT_EMBEDDING_DIM

        # Import MongoDB collection from a shared db utility
        from app.db import get_mongo_collection
        self.mongo_collection = get_mongo_collection()

    def _load_faiss_index(self) -> faiss.Index:
        faiss_index_path = "app/data/faiss.index"
        if os.path.exists(faiss_index_path):
            print(f"📦 Loading FAISS index from {faiss_index_path}")
            index = faiss.read_index(faiss_index_path)
            print(f"✅ FAISS index loaded with {index.ntotal} vectors.")
            return index
        else:
            print("⚠️ FAISS index not found. Returning empty index.")
            return faiss.IndexFlatL2(DEFAULT_EMBEDDING_DIM)

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most relevant clauses based on semantic similarity to the query.
        """
        if self.index.ntotal == 0:
            print("⚠️ FAISS index is empty. Cannot perform search.")
            return []

        query_embedding = await get_gemini_embedding_single(query)
        if query_embedding.size == 0:
            print("⚠️ Failed to generate embedding for query.")
            return []

        try:
            distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        except Exception as e:
            print(f"❌ FAISS search error: {e}")
            return []

        faiss_ids = indices[0].tolist()

        # Fetch corresponding clauses from MongoDB
        try:
            docs = list(self.mongo_collection.find(
                {"faiss_id": {"$in": faiss_ids}},
                {"_id": 0, "clause": 1, "faiss_id": 1}
            ))
        except Exception as e:
            print(f"❌ MongoDB fetch error: {e}")
            return []

        if len(docs) < len(faiss_ids):
            print(f"⚠️ Only found {len(docs)} out of {len(faiss_ids)} clauses in MongoDB.")

        # Reconstruct ordered clauses
        clause_map = {doc["faiss_id"]: doc["clause"] for doc in docs}
        ordered_clauses = [
            {"clause": clause_map[faiss_id]}
            for faiss_id in faiss_ids
            if faiss_id in clause_map
        ]

        return ordered_clauses


# --- CLI Usage for Testing ---
if __name__ == "__main__":
    import asyncio

    async def main():
        retriever = ClauseRetriever()

        if retriever.index.ntotal == 0:
            print("⚠️ FAISS index is empty. Please preload data before testing.")
            return

        query = input("🔍 Enter a query: ").strip() or "What are the benefits for maternity?"
        top_k = 3

        print(f"\n🔍 Running search for: \"{query}\"")
        results = await retriever.search(query, top_k=top_k)

        if not results:
            print("❌ No clauses found.")
        else:
            for i, result in enumerate(results):
                print(f"\n📄 Clause {i + 1}:\n{result['clause']}")

    asyncio.run(main())
