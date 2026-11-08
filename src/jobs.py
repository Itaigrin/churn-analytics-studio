"""
Training runs as a job on the server, not inside the browser session.

A Streamlit session lives as long as its browser connection. Training takes
minutes, and in that time a tab put in the background or a short network drop
can end the session, which used to throw the whole run away. Here the run
belongs to a job that outlives any session: the page keeps the job's id in its
address, so a reloaded or reconnected page picks the job up again, with its
progress while it runs and its result once it is done.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

from .errors import ChurnAppError

logger = logging.getLogger(__name__)

MAX_RUNNING = 2              # the server has a couple of cores; more runs would only slow every one
KEEP_FINISHED_SECONDS = 3 * 60 * 60


class TooManyJobs(ChurnAppError):
    """The server is already training as many models as it can."""


@dataclass
class TrainingJob:
    id: str
    started_at: float = field(default_factory=time.time)
    pct: int = 0
    message: str = "Starting…"
    result: object = None
    error: Optional[str] = None
    error_details: Optional[str] = None
    finished_at: Optional[float] = None

    @property
    def done(self) -> bool:
        return self.finished_at is not None

    def update(self, pct: int, message: str) -> None:
        self.pct, self.message = int(pct), message


class JobRegistry:
    """Every training job on this server, shared by all sessions."""

    def __init__(self, max_running: int = MAX_RUNNING):
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._max_running = max_running

    def start(self, train: Callable[..., object], **kwargs) -> TrainingJob:
        """Run train(**kwargs, progress=...) in the background and return its job at once."""
        with self._lock:
            self._forget_old()
            running = sum(not job.done for job in self._jobs.values())
            if running >= self._max_running:
                raise TooManyJobs(
                    "The server is already training other models. Please try again in a few minutes."
                )
            # Unguessable, because the id in the page address is all it takes to see the results
            job = TrainingJob(id=secrets.token_urlsafe(16))
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, train, kwargs), daemon=True,
                         name=f"training-{job.id[:6]}").start()
        return job

    def get(self, job_id: Optional[str]) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(job_id) if job_id else None

    @staticmethod
    def _run(job: TrainingJob, train: Callable[..., object], kwargs: dict) -> None:
        try:
            job.result = train(**kwargs, progress=job.update)
        except ChurnAppError as exc:
            job.error = str(exc)
        except Exception as exc:
            logger.exception("Training job %s failed", job.id)
            job.error = f"Pipeline error: {exc}"
            job.error_details = traceback.format_exc()
        finally:
            job.finished_at = time.time()

    def _forget_old(self) -> None:
        cutoff = time.time() - KEEP_FINISHED_SECONDS
        for job_id in [i for i, job in self._jobs.items() if job.done and job.finished_at < cutoff]:
            del self._jobs[job_id]
