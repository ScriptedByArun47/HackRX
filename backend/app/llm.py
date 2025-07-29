# llm.py

import json
import os
import google.generativeai as genai
from prompts import build_mistral_prompt
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set and validate the Gemini API key
GEMINI_API = os.getenv("GEMINI_API")
if not GEMINI_API:
    raise ValueError("❌ GEMINI_API environment variable not found in .env file.")
genai.configure(api_key=GEMINI_API)

# Use the Gemini model (Gemma equivalent via Google)
genai_model = genai.GenerativeModel('models/gemini-1.5-flash')


def query_mistral_with_clauses(query, clauses, max_tokens=1800):
    """
    Builds a prompt from the question and clauses, sends it to Gemini, and parses the JSON response.
    """
    prompt = build_mistral_prompt(query, clauses, max_tokens=max_tokens)

    try:
        response = genai_model.generate_content(
            contents=[
                {"role": "user", "parts": [prompt]}
            ],
            generation_config={
                "response_mime_type": "application/json"
            }
        )

        content = response.text.strip()

        # Handle JSON wrapped in markdown format
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        return json.loads(content)

    except json.JSONDecodeError:
        return {
            "answer": "The document does not contain any clear or relevant clause to address the query. Please refer to the policy document directly or contact the insurer for further clarification."
        }
    except Exception as e:
        print(f"❌ Error calling GenAI API from llm.py: {e}")
        return {
            "answer": "Error",
            "supporting_clause": "None",
            "explanation": f"Error while calling LLM API: {str(e)}"
        }
