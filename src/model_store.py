"""
Saving a trained model to a file and loading it back.

A model file contains pickled Python objects, and unpickling a file can run any
code inside it. So a file is only opened after its signature proves it was made
by this installation. The signature is checked before a single byte is unpickled.

Layout: magic | header length | header JSON | HMAC-SHA256 | compressed pickle

The key comes from the CHURN_MODEL_KEY environment variable. On a private
installation one is created on first use. On a hosted app it has to be set by the
owner, since a key stored on a shared disk would protect nothing. Without a key,
saving and loading stay off.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import io
import json
import os
import secrets
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .errors import ChurnAppError
from .runtime_env import ON_CLOUD
from .workflow import ChurnArtifact, TrainingRun

MAGIC = b"CHURNMODEL1\n"
KEY_ENV_VAR = "CHURN_MODEL_KEY"
_KEY_FILE = Path.home() / ".churn_studio" / "model_key"
_SIGNATURE_BYTES = hashlib.sha256().digest_size
_HEADER_LENGTH = struct.Struct(">I")
# A model only behaves the same under the libraries that built it
_LIBRARIES = ("scikit-learn", "numpy", "pandas", "xgboost", "lightgbm", "catboost")


class ModelFileError(ChurnAppError):
    """The file cannot be used as a model."""


@dataclass
class SavedModel:
    """What the prediction page needs, without the training data or the losing models."""
    artifact: ChurnArtifact
    model_name: str
    holdout: dict
    churn_rate: float
    n_train: int
    n_holdout: int
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    target_col: str = ""

    @classmethod
    def from_run(cls, run: TrainingRun) -> "SavedModel":
        return cls(run.artifact, run.best_model_name, run.holdout, run.churn_rate, run.n_train, run.n_holdout,
                   target_col=run.target_col)


def signing_key() -> bytes | None:
    """This installation's key, or None when saving has to stay off."""
    configured = os.environ.get(KEY_ENV_VAR)
    if configured:
        return configured.encode()
    if ON_CLOUD:
        return None
    if not _KEY_FILE.exists():
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(secrets.token_hex(32), encoding="ascii")
        _KEY_FILE.chmod(0o600)
    return _KEY_FILE.read_text(encoding="ascii").strip().encode()


def save_model(model: SavedModel, key: bytes) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    payload = buffer.getvalue()
    header = json.dumps({
        "model_name": model.model_name,
        "trained_at": model.trained_at,
        "libraries": _library_versions(),
    }).encode()
    signature = _sign(key, header, payload)
    return MAGIC + _HEADER_LENGTH.pack(len(header)) + header + signature + payload


def load_model(data: bytes, key: bytes) -> SavedModel:
    header, signature, payload = _split(data)
    if not hmac.compare_digest(signature, _sign(key, header, payload)):
        raise ModelFileError(
            "This model file was not created by this installation, or it was changed "
            "after it was saved, so it was not opened."
        )
    _require_same_libraries(json.loads(header))
    return joblib.load(io.BytesIO(payload))


def _split(data: bytes) -> tuple[bytes, bytes, bytes]:
    not_a_model = ModelFileError("This is not a model file saved by Churn Analytics Studio.")
    if not data.startswith(MAGIC):
        raise not_a_model
    start = len(MAGIC) + _HEADER_LENGTH.size
    if len(data) < start:
        raise not_a_model
    header_end = start + _HEADER_LENGTH.unpack_from(data, len(MAGIC))[0]
    payload_start = header_end + _SIGNATURE_BYTES
    if len(data) <= payload_start:
        raise not_a_model
    return data[start:header_end], data[header_end:payload_start], data[payload_start:]


def _sign(key: bytes, header: bytes, payload: bytes) -> bytes:
    return hmac.new(key, _HEADER_LENGTH.pack(len(header)) + header + payload, hashlib.sha256).digest()


def _library_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in _LIBRARIES}


def _require_same_libraries(header: dict) -> None:
    current = _library_versions()
    different = [f"{name} {version} (now {current.get(name)})"
                 for name, version in header["libraries"].items() if current.get(name) != version]
    if different:
        raise ModelFileError(
            f"This model was trained with different library versions: {', '.join(different)}. "
            "Its predictions could not be trusted here, so please train it again."
        )
