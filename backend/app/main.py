# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from app.extract_clauses import extract_clauses_from_url
from app.prompts import MISTRAL_SYSTEM_PROMPT_TEMPLATE, build_mistral_prompt
import faiss
import numpy as np
import json
import requests
import asyncio
import sys
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="HackRx Gemini QA API",
    description="Extracts legal clauses and answers questions using Gemini models.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

# Gemini API configuration
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    sys.exit(1)

# Gemini Model Setup
genai_model = genai.GenerativeModel('models/gemini-1.5-flash')
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072

# Tokenizer for prompt trimming
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------- Embedding Helpers ----------------------

async def get_gemini_embeddings_batch_async(texts: list[str], task_type: str = "SEMANTIC_SIMILARITY"):
    if not texts:
        return np.array([])
    try:
        response = await genai.batch_embed_contents_async(
            model=GEMINI_EMBEDDING_MODEL,
            contents=texts,
            task_type=task_type
        )
        embeddings = response.embeddings
        if len(embeddings) != len(texts):
            print("⚠️ Mismatch between input texts and returned embeddings.")
        return np.array([e.values for e in embeddings]).astype("float32")
    except Exception as e:
        print(f"Error generating Gemini batch embeddings: {e}")
        return np.array([])


async def get_gemini_embedding_single_async(text: str, task_type: str = "SEMANTIC_SIMILARITY"):
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


# ---------------------- FAISS Index ----------------------

async def build_faiss_index(clauses):
    texts = [c["clause"] for c in clauses]
    vectors = await get_gemini_embeddings_batch_async(texts, task_type="RETRIEVAL_DOCUMENT")
    
    if vectors.size == 0:
        print("No vectors generated.")
        return faiss.IndexFlatL2(EMBEDDING_DIM), [], []

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, texts, vectors


# ---------------------- Retrieval Helpers ----------------------

def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    stopwords = {"what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on", "to", "does", "do", "are"}
    return [t for t in tokens if t not in stopwords and len(t) > 2]


async def get_top_clauses(question, index, texts, k=15):
    q_vector = await get_gemini_embedding_single_async(question, task_type="RETRIEVAL_QUERY")
    if q_vector.size == 0:
        return []

    _, I = index.search(q_vector.reshape(1, -1), k)
    top_clauses = [texts[i] for i in I[0]]

    keywords = extract_keywords(question)
    keyword_matches = [c for c in texts if any(k in c.lower() for k in keywords)]

    combined = list(dict.fromkeys(top_clauses + keyword_matches))

    def keyword_score(clause):
        return sum(1 for word in keywords if word in clause.lower())

    combined = sorted(combined, key=keyword_score, reverse=True)
    return combined[:10]


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
            print(f"🔹 Tokens used: Prompt {usage.prompt_token_count}, Response {usage.candidates_token_count}")

        return json.loads(raw_output)
    except Exception as e:
        print(f"❌ LLM API Error: {e}")
        return {
            "answer": "Error",
            "supporting_clause": "None",
            "explanation": f"LLM error: {str(e)}"
        }


# ---------------------- Main Route ----------------------

@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]

    all_clauses = []
    for url in doc_urls:
        extracted = extract_clauses_from_url(url)
        if not isinstance(extracted, list):
            print(f"⚠️ Clause extraction failed for URL: {url}")
            continue
        all_clauses.extend(extracted)

    index, clause_texts, _ = await build_faiss_index(all_clauses)

    if index.ntotal == 0:
        return {"answers": ["Error: Could not build FAISS index. No clauses or embedding failed."]}

    async def process_question(q):
        print(f"\n🧠 Processing: {q}")
        top_clauses_raw = await get_top_clauses(q, index, clause_texts, k=15)
        print(f"📌 Clauses retrieved: {len(top_clauses_raw)}")

        clause_objects = trim_clauses([{"clause": c} for c in top_clauses_raw])
        print(f"✂️ Trimmed clauses: {len(clause_objects)}")

        prompt = build_mistral_prompt(q, clause_objects, max_tokens=1800)
        print(f"📝 Prompt:\n{prompt[:300]}...\n")

        response = await call_genai_llm_async(prompt)
        print(f"📥 LLM response: {response}")
        return response

    results = await asyncio.gather(*[process_question(q) for q in req.questions])
    
    # You can return full responses if needed
    return {
        "answers": [res.get("answer", "No answer found") for res in results],
        "details": results
    }


# ---------------------- Entry Point ----------------------

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
