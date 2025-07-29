import re

def parse_query(query: str) -> dict:
    """
    Parse a natural language insurance-related query and return a metadata dict with tags.

    Parameters:
        query (str): User's raw input string.

    Returns:
        dict: {
            "original_query": <str>,
            "tags": <list[str]>,
            "has_medical": <bool>,  # True if query relates to medical conditions or procedures
            "has_benefit": <bool>   # True if query mentions optional or financial benefits
        }
    """
    query = query.strip()

    keywords = {
        "surgery": ["surgery", "operation", "procedure"],
        "maternity": ["maternity", "pregnancy"],
        "hospital": ["hospital", "facility"],
        "waiting_period": ["waiting period", "eligibility"],
        "pre_existing": ["pre-existing", "PED"],
        "discount": ["NCD", "discount"],
        "ayush": ["ayurveda", "homeopathy"],
        "organ_donor": ["organ donor", "transplant"]
    }

    found_tags = []

    for tag, terms in keywords.items():
        for term in terms:
            # Ensure full-word match using \b and case-insensitive search
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, query, re.IGNORECASE):
                found_tags.append(tag)
                break  # Tag matched; skip remaining synonyms for this tag

    return {
        "original_query": query,
        "tags": found_tags,
        "has_medical": any(tag in found_tags for tag in ["surgery", "maternity", "pre_existing", "organ_donor"]),
        "has_benefit": any(tag in found_tags for tag in ["discount", "ayush"])
    }
