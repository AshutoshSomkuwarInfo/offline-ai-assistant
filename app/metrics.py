import sys
import time
import resource


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def rss_memory_mb() -> float:
    """Resident set size for reporting (approximate; OS units differ)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0
