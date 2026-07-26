"""
Runtime environment detection — imports nothing but `os`.

Must stay dependency-free so it can be imported at the very top of app.py,
BEFORE numpy / sklearn / lightgbm load their OpenMP runtimes.

Why this matters
----------------
On managed cloud containers (HuggingFace Spaces, Streamlit Cloud) the CPU
quota is small and cgroup-limited, but OpenMP still reads the *host* core
count. Combining that over-subscription with joblib's fork-based `n_jobs=-1`
corrupts the heap and kills the process with SIGSEGV (exit code 139).

Forcing single-threaded BLAS/OpenMP plus `n_jobs=1` removes both halves of
that interaction. Local development is unaffected and keeps full parallelism.
"""

import os


def _detect_cloud() -> bool:
    # HuggingFace Spaces
    if os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"):
        return True
    if os.path.isdir("/home/user/app"):
        return True
    # Streamlit Community Cloud
    if os.path.exists("/mount/src"):
        return True
    return False


ON_CLOUD = _detect_cloud()

# n_jobs=-1 + OpenMP oversubscription = SIGSEGV on cloud containers
N_JOBS = 1 if ON_CLOUD else -1

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_threads() -> None:
    """Pin every BLAS/OpenMP backend to 1 thread. Call before importing numpy."""
    if not ON_CLOUD:
        return
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")
