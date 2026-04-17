"""
Offline assistant pipeline aligned with the thesis architecture:
Intent analyzer → context retrieval (general + personal FAISS) → on-device SLM (Ollama).

Personal context can include manual notes plus text synced from Gmail, Calendar, and
Google Photos (see app/google_sync.py).
"""
import argparse
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
        parts.append("Optional local notes (general, retrieved):\n" + "\n".join(f"- {x}" for x in g))
    if p:
        parts.append("Personal context (retrieved):\n" + "\n".join(f"- {x}" for x in p))
    if not parts:
        return "(No matching local documents — rely on the on-device model for offline answers.)"
    return "\n\n".join(parts)


def _general_source_paths(base: str) -> list[str]:
    paths: list[str] = []
    for name in ("general_knowledge.txt", "general_knowledge_extra.txt"):
        p = os.path.join(base, "data", name)
        if os.path.isfile(p):
            paths.append(p)
    gen_dir = os.path.join(base, "data", "general")
    if os.path.isdir(gen_dir):
        for fn in sorted(os.listdir(gen_dir)):
            if fn.endswith(".txt"):
                paths.append(os.path.join(gen_dir, fn))
    return paths


def _offline_system_instructions(intent_value: str) -> str:
    base = (
        "You run fully offline: no cloud APIs during inference. "
        "The on-device small language model holds broad general knowledge in its weights."
    )
    if intent_value == "general_knowledge":
        return (
            base + " Optional lines below are local text retrieval (RAG) only—use them if relevant; "
            "if none appear or they do not apply, answer the question directly from your training knowledge."
        )
    if intent_value == "personal_context":
        return base + " Prefer the personal context below when it is relevant to the question."
    return (
        base + " Use retrieved notes below when they help; combine general and personal parts as needed."
    )


def _personal_source_paths(base: str) -> list[str]:
    manual = os.path.join(base, "data", "personal_context.txt")
    google = os.path.join(base, "data", "google_personal_sync.txt")
    paths = [manual]
    if os.path.isfile(google):
        paths.append(google)
    return paths


def run_once(query: str) -> None:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    general_paths = _general_source_paths(base)
    personal_paths = _personal_source_paths(base)

    t0 = time.perf_counter()
    engine = ContextRetrievalEngine()
    engine.load_general_sources(general_paths)
    engine.load_personal_sources(personal_paths)
    load_ms = elapsed_ms(t0)

    intent = analyze_intent(query)
    print(f"Detected intent: {intent.value}")
    if general_paths:
        print(f"General knowledge files (optional RAG): {', '.join(general_paths)}")
    else:
        print("General knowledge RAG: none (offline answers come from the SLM only).")
    if len(personal_paths) > 1:
        print(f"Personal sources: {', '.join(personal_paths)}")

    t1 = time.perf_counter()
    chunks = engine.retrieve(query, intent, k=3)
    retrieve_ms = elapsed_ms(t1)

    context_block = _format_context(chunks)
    print("Retrieved context:\n", context_block, "\n", sep="")

    instructions = _offline_system_instructions(intent.value)
    prompt = f"""You are a privacy-preserving offline assistant.

{instructions}

{context_block}

User question: {query}

Answer concisely:"""

    t2 = time.perf_counter()
    response = generate_response(prompt)
    llm_ms = elapsed_ms(t2)

    print("Response:\n", response, "\n", sep="")
    print(
        f"Metrics — load_index: {load_ms:.1f} ms | retrieval: {retrieve_ms:.1f} ms | "
        f"llm: {llm_ms:.1f} ms | rss_memory≈{rss_memory_mb():.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline AI assistant (FAISS + on-device SLM). "
        "Optionally sync Google data first."
    )
    parser.add_argument(
        "--sync-google",
        action="store_true",
        help="Fetch Gmail, Calendar, and Photos metadata into data/google_personal_sync.txt",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question (omit for interactive prompt)",
    )
    args = parser.parse_args()

    if args.sync_google:
        from app.google_sync import sync_google_data

        path = sync_google_data()
        print(f"Google data synced to {path}\n")

    print("Offline AI assistant (on-device SLM + contextual retrieval)")
    query = args.query
    if not query:
        query = input("Ask: ").strip()
    if not query:
        print("Empty query.")
        return
    run_once(query)


if __name__ == "__main__":
    main()
