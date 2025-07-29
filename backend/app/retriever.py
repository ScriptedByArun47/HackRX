import os
import json
import faiss
import numpy as np
import google.generativeai as genai

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
EMBEDDING_DIM = 3072  # Must match with preload
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# --- Single Query Embedding ---
async def get_gemini_embedding_single(text: str, task_type: str = "RETRIEVAL_QUERY"):
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
    def __init__(self):
        self.index = self.load_faiss_index()

        # Import MongoDB collection from a central db module
        from app.db import get_mongo_collection
        self.mongo_collection = get_mongo_collection()

    def load_faiss_index(self):
        faiss_index_path = "app/data/faiss.index"
        if os.path.exists(faiss_index_path):
            print(f"📦 Loading FAISS index from {faiss_index_path}")
            return faiss.read_index(faiss_index_path)
        else:
            print("⚠️ FAISS index not found. Run clause_loader.py first.")
            return faiss.IndexFlatL2(EMBEDDING_DIM)

    async def search(self, query: str, top_k: int = 5):
        # Generate embedding for the user query
        query_embedding = await get_gemini_embedding_single(query)

        if query_embedding.size == 0:
            print("⚠️ Failed to generate query embedding.")
            return []

        D, I = self.index.search(query_embedding.reshape(1, -1), top_k)
        faiss_ids = I[0].tolist()

        # Retrieve corresponding clauses from MongoDB
        docs = list(self.mongo_collection.find(
            {"faiss_id": {"$in": faiss_ids}},
            {"_id": 0, "clause": 1, "faiss_id": 1}
        ))

        # Reorder by FAISS ID rank
        clause_map = {doc["faiss_id"]: doc["clause"] for doc in docs}
        ordered_clauses = [{"clause": clause_map[faiss_id]} for faiss_id in faiss_ids if faiss_id in clause_map]

        return ordered_clauses

# --- CLI Usage for Testing ---
if __name__ == "__main__":
    import asyncio

    async def main():
        retriever = ClauseRetriever()
        if retriever.index.ntotal == 0:
            print("⚠️ FAISS index is empty. Please run the preload script first.")
            return

        query = "What are the benefits for maternity?"
        top_k = 3
        print(f"\n🔍 Query: {query}")
        results = await retriever.search(query, top_k=top_k)

        for i, result in enumerate(results):
            print(f"Clause {i+1}: {result['clause']}")

    asyncio.run(main())
