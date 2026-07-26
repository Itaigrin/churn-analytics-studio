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


def usable_cpus() -> int:
    """
    Cores this process may actually use.

    In a container, `os.cpu_count()` reports the HOST's cores — often dozens —
    while the cgroup quota allows only a couple. That gap is the whole bug:
    `n_jobs=-1` spawns one worker per host core onto a 2-core quota, and the
    resulting oversubscription is what corrupts memory. Read the quota instead.
    """
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except Exception:
        pass
    # Affinity mask (respects taskset / some orchestrators)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    return max(1, os.cpu_count() or 1)


ON_CLOUD = _detect_cloud()

# Use every core the container really has — just never more. -1 locally.
N_JOBS = usable_cpus() if ON_CLOUD else -1

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_threads() -> None:
    """
    Pin BLAS/OpenMP to 1 thread per worker. Call before importing numpy.

    Parallelism comes from joblib (`N_JOBS` processes). Letting each of those
    ALSO spin up a full OpenMP thread pool multiplies the two and re-creates
    the oversubscription we are avoiding, so the inner pools stay at 1.
    """
    if not ON_CLOUD:
        return
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")
