# prompts.py

from transformers import AutoTokenizer

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Prompt template for Gemini / Mistral
MISTRAL_SYSTEM_PROMPT_TEMPLATE = """
You are an expert insurance assistant. Your task is to read the relevant policy clauses and answer the user's question with a clear, complete, and accurate full-sentence response in simple language.

Instructions:
- ONLY use the information explicitly provided in the policy clauses.
- Do NOT assume, guess, or include outside knowledge.
- Do NOT mention clause numbers, section names, or document formatting.
- Your answer must be factual, specific, and based only on the content of the clauses.
- Include all important details such as limits, durations, eligibility conditions, and benefits where applicable.

Output format:
{{
  "answer": "<One complete and factual sentence derived strictly from the given clauses>"
}}

User Question:
{query}

Relevant Policy Clauses:
{clauses}

Respond with only the raw JSON (no markdown or formatting).
"""

def build_mistral_prompt(query: str, clauses: list[dict], max_tokens: int = 1500) -> str:
    """
    Builds a prompt string for the LLM using a limited number of tokens from clauses.
    Parameters:
        - query: the user question
        - clauses: list of dicts, each containing a 'clause' string
        - max_tokens: token budget for the clauses (default: 1500)
    Returns:
        - formatted prompt string
    """
    trimmed_clauses = []
    token_count = 0

    for clause_obj in clauses:
        clause = clause_obj.get("clause", "").strip()
        tokens = len(tokenizer.tokenize(clause))
        if token_count + tokens > max_tokens:
            break
        trimmed_clauses.append(clause)
        token_count += tokens

    clause_text = "\n\n".join(trimmed_clauses)
    return MISTRAL_SYSTEM_PROMPT_TEMPLATE.format(query=query.strip(), clauses=clause_text)
