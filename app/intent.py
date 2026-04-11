"""
User query intent analyzer — routes retrieval toward general knowledge,
personal context, or both (matches proposed architecture diagram).
"""
import re
from enum import Enum


class QueryIntent(str, Enum):
    GENERAL_KNOWLEDGE = "general_knowledge"
    PERSONAL_CONTEXT = "personal_context"
    HYBRID = "hybrid"


def analyze_intent(query: str) -> QueryIntent:
    q = query.lower().strip()

    personal = bool(
        re.search(
            r"\b(do i have|my |mine\b|show my|find my|remind me|"
            r"meeting|calendar|schedule|photos?\b|trip\b|emails?\b|deadline)\b",
            q,
        )
    )
    general = bool(
        re.search(
            r"\b(explain|what is|what are|define|how does|how do|describe|why |"
            r"compare|difference between)\b",
            q,
        )
    )

    if personal and general:
        return QueryIntent.HYBRID
    if personal:
        return QueryIntent.PERSONAL_CONTEXT
    return QueryIntent.GENERAL_KNOWLEDGE
