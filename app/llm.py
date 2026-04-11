import os
import subprocess
from typing import Optional

DEFAULT_MODEL = "phi3"


def generate_response(prompt: str, model: Optional[str] = None) -> str:
    """
    On-device SLM via Ollama (Phi-3 / LLaMA / Mistral — pull the model you use locally).
    Set OLLAMA_MODEL to override the default.
    """
    name = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    try:
        result = subprocess.run(
            ["ollama", "run", name],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=120,
        )
        out = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            return f"Error (ollama exit {result.returncode}): {result.stderr.strip()}"
        return out or (result.stderr.strip() or "(empty model output)")

    except Exception as e:
        return f"Error calling LLM: {e}"
