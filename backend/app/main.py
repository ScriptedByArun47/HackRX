# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from app.extract_clauses import extract_clauses_from_url
from app.prompts import MISTRAL_SYSTEM_PROMPT_TEMPLATE # Keep this for LLM prompt
import faiss
import numpy as np
import json
import requests
import asyncio
import sys
import os
import re
from app.prompts import build_mistral_prompt
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

# REMOVE: model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Gemini API Configuration ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API environment variable not set. Please set it.")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API in main.py: {e}")
    sys.exit(1) # Exit if API key is not configured

genai_model = genai.GenerativeModel('models/gemini-1.5-flash')

# --- Gemini Embedding Configuration for this file ---
EMBEDDING_DIM = 3072 # Must match the dimension used in clause_loader.py
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# --- Helper functions for Gemini Embeddings ---
async def get_gemini_embeddings_batch_async(texts: list[str], task_type: str = "SEMANTIC_SIMILARITY"):
    """Generates embeddings for a list of texts using the Gemini API (batch)."""
    if not texts:
        return np.array([])
    try:
        response = await genai.batch_embed_contents_async(
            model=GEMINI_EMBEDDING_MODEL,
            contents=texts,
            task_type=task_type
        )
        return np.array([item.embedding.values for item in response.embeddings]).astype("float32")
    except Exception as e:
        print(f"Error generating Gemini batch embeddings: {e}")
        return np.array([])

async def get_gemini_embedding_single_async(text: str, task_type: str = "SEMANTIC_SIMILARITY"):
    """Generates a single embedding for a text using the Gemini API."""
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
        print(f"Error generating Gemini single embedding: {e}")
        return np.array([])


# Changed to async
async def build_faiss_index(clauses):
    texts = [c["clause"] for c in clauses]
    # Use Gemini for encoding clauses
    vectors = await get_gemini_embeddings_batch_async(texts, task_type="RETRIEVAL_DOCUMENT")
    
    if vectors.size == 0:
        print("No vectors generated for FAISS index.")
        return faiss.IndexFlatL2(EMBEDDING_DIM), [], [] # Return empty index, texts, vectors

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, texts, vectors

# keyword extractor (no changes needed here)
def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    stopwords = {"what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on", "to", "does", "do", "are"}
    return [t for t in tokens if t not in stopwords and len(t) > 2]

# FAISS + keyword relevance hybrid retrieval - Changed to async
async def get_top_clauses(question, index, texts, k=15):
    # Use Gemini for encoding the query
    q_vector = await get_gemini_embedding_single_async(question, task_type="RETRIEVAL_QUERY")
    
    if q_vector.size == 0:
        return []

    # Reshape for FAISS search
    _, I = index.search(q_vector.reshape(1, -1), k)
    top_clauses = [texts[i] for i in I[0]]

    # Add keyword matching clauses
    keywords = extract_keywords(question)
    keyword_matches = [c for c in texts if any(k in c.lower() for k in keywords)]

    # Merge + deduplicate
    combined = list(dict.fromkeys(top_clauses + keyword_matches))  # preserves order

    # Re-rank by simple keyword overlap
    def keyword_score(clause):
        return sum(1 for word in keywords if word in clause.lower())

    combined = sorted(combined, key=keyword_score, reverse=True)
    return combined[:10]


async def call_genai_llm_async(prompt: str, timeout: int = 120) -> dict:
    try:
        response = await asyncio.to_thread(
            genai_model.generate_content,
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={
                "response_mime_type": "application/json"
            }
        )

        raw_output = response.text.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        usage = getattr(response, "usage_metadata", None)
        if usage:
            print(f"🔹 Prompt tokens: {usage.prompt_token_count}")
            print(f"🔹 Response tokens: {usage.candidates_token_count}")

        return json.loads(raw_output)

    except Exception as e:
        print(f"❌ Error calling GenAI API: {e}")
        return {
            "answer": "Error",
            "supporting_clause": "None",
            "explanation": f"Error while calling LLM API: {str(e)}"
        }

# --- This part uses AutoTokenizer from transformers, which might still bring in
#     some dependencies. If you need to *completely* remove transformers,
#     you'd need a different tokenization method (e.g., tiktoken for OpenAI models,
#     or a simple split/count if exact token count isn't critical for prompting).
#     For now, leaving it as is, as it's just for prompt trimming.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2") # This still requires the tokenizer

def trim_clauses(clauses, max_tokens=1800):
    result = []
    total = 0
    for c in clauses:
        clause = c["clause"]
        tokens = len(tokenizer.tokenize(clause))
        if total + tokens > max_tokens:
            break
        result.append({"clause": clause})
        total += tokens
    return result


@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]

    all_clauses = []
    for url in doc_urls:
        all_clauses.extend(extract_clauses_from_url(url))

    # Build FAISS index using Gemini embeddings
    index, clause_texts, _ = await build_faiss_index(all_clauses)

    if index.ntotal == 0:
        return {"answers": ["Error: Could not build FAISS index. No clauses or embedding failed."]}


    async def process_question(q):
        print(f"\n🧠 Processing question: {q}")
        # Get top clauses using Gemini embeddings for search
        top_clauses_raw = await get_top_clauses(q, index, clause_texts, k=15)
        print(f"📌 Top clauses retrieved: {len(top_clauses_raw)}") # Log actual retrieved count
        
        # Ensure top_clauses_raw is a list of {"clause": "text"} objects or convert it
        # Current get_top_clauses returns list of strings, convert them back for trim_clauses
        clause_objects_for_trim = [{"clause": c} for c in top_clauses_raw]
        clause_objects = trim_clauses(clause_objects_for_trim) # Now it's a list of dicts again
        
        print(f"✂️ Trimmed {len(clause_objects)} clauses for prompt")

        prompt = build_mistral_prompt(q, clause_objects, max_tokens=1800)
        print(f"📝 Final prompt:\n{prompt[:300]}...")
        response = await call_genai_llm_async(prompt)
        print(f"📥 LLM response: {response}")
        return response

    results = await asyncio.gather(*[process_question(q) for q in req.questions])
    return {"answers": [res.get("answer", "No answer found") for res in results]}