"""Training as a server job that outlives the browser session."""
import threading
import time

import pytest

from src.errors import DataValidationError
from src.jobs import KEEP_FINISHED_SECONDS, JobRegistry, TooManyJobs


def wait(job, timeout=10):
    deadline = time.time() + timeout
    while not job.done:
        assert time.time() < deadline
        time.sleep(0.01)


def test_a_job_reports_progress_and_keeps_its_result():
    registry = JobRegistry()
    release = threading.Event()

    def train(progress, value):
        progress(40, "halfway")
        release.wait(5)
        return value * 2

    job = registry.start(train, value=21)
    time.sleep(0.05)
    assert not job.done and (job.pct, job.message) == (40, "halfway")
    release.set()
    wait(job)
    assert registry.get(job.id).result == 42 and job.error is None


def test_a_users_mistake_is_kept_as_a_plain_message():
    def train(progress):
        raise DataValidationError("At least 100 rows are needed.")

    job = JobRegistry().start(train)
    wait(job)
    assert job.error == "At least 100 rows are needed." and job.error_details is None


def test_a_bug_keeps_its_traceback():
    def train(progress):
        raise KeyError("boom")

    job = JobRegistry().start(train)
    wait(job)
    assert job.error.startswith("Pipeline error") and "KeyError" in job.error_details


def test_the_server_refuses_more_runs_than_it_can_handle():
    registry = JobRegistry(max_running=1)
    release = threading.Event()
    first = registry.start(lambda progress: release.wait(5))
    with pytest.raises(TooManyJobs):
        registry.start(lambda progress: None)
    release.set()
    wait(first)
    wait(registry.start(lambda progress: None))


def test_job_ids_cannot_be_guessed_and_unknown_ids_find_nothing():
    registry = JobRegistry()
    ids = {registry.start(lambda progress: None).id for _ in range(2)}
    assert len(ids) == 2 and all(len(i) >= 20 for i in ids)
    assert registry.get("nope") is None and registry.get(None) is None


def test_old_finished_jobs_are_forgotten():
    registry = JobRegistry()
    old = registry.start(lambda progress: None)
    wait(old)
    old.finished_at -= KEEP_FINISHED_SECONDS + 1
    registry.start(lambda progress: None)
    assert registry.get(old.id) is None
