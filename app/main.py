"""
Offline assistant pipeline aligned with the thesis architecture:
Intent analyzer → context retrieval (general + personal FAISS) → on-device SLM (Ollama).
"""
import os
import time

from app.intent import analyze_intent
from app.llm import generate_response
from app.metrics import elapsed_ms, rss_memory_mb
from app.retriever import ContextRetrievalEngine


def _format_context(chunks: dict[str, list[str]]) -> str:
    parts = []
    g = chunks.get("general_knowledge") or []
    p = chunks.get("personal_context") or []
    if g:
        parts.append("General knowledge (retrieved):\n" + "\n".join(f"- {x}" for x in g))
    if p:
        parts.append("Personal context (retrieved):\n" + "\n".join(f"- {x}" for x in p))
    if not parts:
        return "(No retrieved context — answer from general reasoning if appropriate.)"
    return "\n\n".join(parts)


def run_once(query: str) -> None:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    general_path = os.path.join(base, "data", "general_knowledge.txt")
    personal_path = os.path.join(base, "data", "personal_context.txt")

    t0 = time.perf_counter()
    engine = ContextRetrievalEngine()
    engine.load_general_knowledge(general_path)
    engine.load_personal_context(personal_path)
    load_ms = elapsed_ms(t0)

    intent = analyze_intent(query)
    print(f"Detected intent: {intent.value}")

    t1 = time.perf_counter()
    chunks = engine.retrieve(query, intent, k=3)
    retrieve_ms = elapsed_ms(t1)

    context_block = _format_context(chunks)
    print("Retrieved context:\n", context_block, "\n", sep="")

    prompt = f"""You are a privacy-preserving offline assistant. Use ONLY the context below when it is relevant; otherwise answer briefly from general knowledge.

{context_block}

User question: {query}

Answer:"""

    t2 = time.perf_counter()
    response = generate_response(prompt)
    llm_ms = elapsed_ms(t2)

    print("Response:\n", response, "\n", sep="")
    print(
        f"Metrics — load_index: {load_ms:.1f} ms | retrieval: {retrieve_ms:.1f} ms | "
        f"llm: {llm_ms:.1f} ms | rss_memory≈{rss_memory_mb():.1f} MB"
    )


def main() -> None:
    print("Offline AI assistant (on-device SLM + contextual retrieval)")
    query = input("Ask: ").strip()
    if not query:
        print("Empty query.")
        return
    run_once(query)


if __name__ == "__main__":
    main()
