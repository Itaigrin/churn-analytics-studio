"""Saving a model, and refusing any file this installation did not make."""
import pickle

import numpy as np
import pytest

from src import model_store
from src.model_store import MAGIC, ModelFileError, SavedModel, load_model, save_model, signing_key

KEY = b"test-key"
RAN_FLAG = []


def _run_code():
    RAN_FLAG.append(True)
    return "ran"


class Malicious:
    """Unpickling this runs code, which is what an attacker's file would do."""
    def __reduce__(self):
        return (_run_code, ())


@pytest.fixture(scope="module")
def saved_file(training_run):
    return save_model(SavedModel.from_run(training_run), KEY)


def test_a_saved_model_scores_exactly_the_same_after_loading(training_run, saved_file, customers):
    new = customers.drop(columns=["Churn"]).head(50)
    loaded = load_model(saved_file, KEY)
    np.testing.assert_array_equal(loaded.artifact.predict_proba(new), training_run.artifact.predict_proba(new))


def test_a_loaded_model_keeps_what_the_prediction_page_needs(training_run, saved_file):
    loaded = load_model(saved_file, KEY)
    assert loaded.model_name == training_run.best_model_name
    assert loaded.artifact.threshold == training_run.artifact.threshold
    assert loaded.churn_rate == training_run.churn_rate
    assert loaded.holdout["roc_auc"] == training_run.holdout["roc_auc"]
    assert "threshold_curve" in loaded.holdout


def test_the_file_holds_no_customer_ids_or_losing_models(saved_file):
    loaded = load_model(saved_file, KEY)
    assert not hasattr(loaded, "results")
    assert "customer_id" not in loaded.artifact.schema.columns
    assert not any("C00001" in str(ref.values) for ref in loaded.artifact.reference.values())


def _header_len(data):
    return int.from_bytes(data[len(MAGIC):len(MAGIC) + 4], "big")


def test_a_changed_file_is_refused(saved_file):
    tampered = bytearray(saved_file)
    tampered[-100] ^= 0xFF
    with pytest.raises(ModelFileError, match="not created by this installation"):
        load_model(bytes(tampered), KEY)


def test_a_changed_header_is_refused(saved_file):
    tampered = bytearray(saved_file)
    tampered[len(MAGIC) + 6] ^= 0xFF
    with pytest.raises(ModelFileError):
        load_model(bytes(tampered), KEY)


def test_a_file_from_another_installation_is_refused(saved_file):
    with pytest.raises(ModelFileError, match="not created by this installation"):
        load_model(saved_file, b"another-key")


def test_a_malicious_file_never_runs(saved_file):
    payload = pickle.dumps(Malicious())
    header_end = len(MAGIC) + 4 + _header_len(saved_file)
    forged = saved_file[:header_end + 32] + payload
    RAN_FLAG.clear()
    with pytest.raises(ModelFileError):
        load_model(forged, KEY)
    assert RAN_FLAG == []


@pytest.mark.parametrize("data", [b"", b"hello", b"PK\x03\x04 not a model", MAGIC, MAGIC + b"\x00\x00"])
def test_other_files_get_a_clear_message(data):
    with pytest.raises(ModelFileError, match="not a model file"):
        load_model(data, KEY)


def test_other_library_versions_are_refused(saved_file, monkeypatch):
    real = model_store._library_versions()
    monkeypatch.setattr(model_store, "_library_versions", lambda: {**real, "scikit-learn": "0.1"})
    with pytest.raises(ModelFileError, match="different library versions"):
        load_model(saved_file, KEY)


def test_a_hosted_app_without_a_key_keeps_saving_off(monkeypatch):
    monkeypatch.delenv(model_store.KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(model_store, "ON_CLOUD", True)
    assert signing_key() is None


def test_a_configured_key_is_used(monkeypatch):
    monkeypatch.setenv(model_store.KEY_ENV_VAR, "abc")
    assert signing_key() == b"abc"


def test_a_private_install_creates_and_keeps_its_own_key(monkeypatch, tmp_path):
    monkeypatch.delenv(model_store.KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(model_store, "ON_CLOUD", False)
    monkeypatch.setattr(model_store, "_KEY_FILE", tmp_path / "dir" / "model_key")
    first = signing_key()
    assert len(first) == 64 and signing_key() == first
