"""Validate inputs and predict with a trusted local experiment artifact."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd

from .dataset import CATEGORIES, FEATURES, NULLABLE, SHA256, parse_value
from .evaluation import classify

MODEL_VERSION = "1.0"


def input_schema():
    return {
        name: {
            "nullable": name in NULLABLE,
            **({"allowed": list(CATEGORIES[name])} if name in CATEGORIES else {
                "type": "nonnegative number" if name == "oldpeak" else "positive integer",
            }),
        }
        for name in FEATURES
    }


def validated_frame(values):
    if not isinstance(values, Mapping) or set(values) != set(FEATURES):
        raise ValueError("Provide exactly the 13 Cleveland predictor names")
    row = []
    for name in FEATURES:
        value = values[name]
        if value is None and name in NULLABLE:
            row.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name}: expected a numeric value")
        try:
            row.append(parse_value(name, str(value)))
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
    return pd.DataFrame([row], columns=list(FEATURES), dtype=float)


@dataclass
class ModelArtifact:
    pipeline: object
    metadata: dict

    def predict(self, values):
        score = float(self.pipeline.predict_proba(validated_frame(values))[0, 1])
        threshold = self.metadata["threshold"]
        return {"classification": int(classify([score], threshold)[0]),
                "score": score, "threshold": threshold}


def load_trusted_model(path: Path):
    """Load only a trusted joblib file because deserialization can execute code."""
    artifact = joblib.load(path)
    if (
        not isinstance(artifact, ModelArtifact)
        or artifact.metadata.get("model_version") != MODEL_VERSION
        or artifact.metadata.get("dataset_sha256") != SHA256
        or artifact.metadata.get("features") != list(FEATURES)
        or artifact.metadata.get("input_schema") != input_schema()
    ):
        raise ValueError("Incompatible model artifact")
    return artifact
