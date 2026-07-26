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


def _cgroup_quota() -> "int | None":
    """CPU quota from the cgroup, or None when the container does not expose one."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max" and int(period) > 0:
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
    return None


ON_CLOUD    = _detect_cloud()
CPU_SOURCE  = "local"   # where the core count came from; surfaced for debugging
_QUOTA      = _cgroup_quota() if ON_CLOUD else None


def usable_cpus() -> int:
    """
    Cores this process may actually use.

    Inside a container `os.cpu_count()` reports the HOST's cores — often dozens
    — while the real quota is a couple. Sizing worker pools off that number is
    the original bug: dozens of processes land on two cores and the resulting
    oversubscription corrupts memory (SIGSEGV, exit 139).

    So on cloud we trust ONLY an explicit cgroup quota. If the container does
    not expose one we deliberately fall back to 1 rather than to cpu_count():
    a needlessly serial run is slow, but guessing high crashes the app.
    """
    global CPU_SOURCE
    if not ON_CLOUD:
        CPU_SOURCE = "local"
        return max(1, os.cpu_count() or 1)
    if _QUOTA:
        CPU_SOURCE = f"cgroup quota ({_QUOTA})"
        return max(1, min(_QUOTA, 8))
    CPU_SOURCE = "no quota exposed — pinned to 1"
    return 1


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
    Cap BLAS/OpenMP at the container's real core count. Call before numpy.

    The goal is to stop OpenMP from sizing its pools off the HOST's core
    count — not to serialise the work. Pinning these to 1 would cripple the
    many single-process paths (mutual information, final model fits) that
    have no joblib parallelism to fall back on. joblib's worker processes
    limit their own inner pools, so this setting cannot multiply with N_JOBS.
    """
    if not ON_CLOUD:
        return
    n = str(usable_cpus())
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, n)
