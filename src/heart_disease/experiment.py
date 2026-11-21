"""Run one locked Cleveland model comparison and held-out evaluation."""

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from .dataset import FEATURES, binary_target, load_snapshot
from .evaluation import THRESHOLDS, classify, metrics
from .inference import MODEL_VERSION, ModelArtifact, input_schema, load_trusted_model
from .modeling import (
    F2_TOLERANCE, INNER_FOLDS, OUTER_FOLDS, candidates, compare, configure, select_model,
)
from .preprocessing import training_features
from .split import SEED, split_rows

def frozen_split(path, records, checksum):
    payload = path.read_bytes()
    # Accept alternative checkout line endings only after validating every row assignment.
    manifest = json.loads(payload)
    train, test = split_rows(records)
    if (manifest["sha256"], manifest["seed"], manifest["train_row_ids"], manifest["test_row_ids"]) != (
        checksum, SEED, train, test,
    ):
        raise ValueError("Frozen holdout manifest disagrees with the verified snapshot and split")
    return train, test, hashlib.sha256(payload).hexdigest()


def dataset_rows(records, row_ids):
    features = training_features(records, row_ids)
    labels = np.array([binary_target(records[row_id - 1][-1]) for row_id in row_ids])
    return features, labels


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def select_and_fit(all_candidates, features, labels, checksum, split_checksum, output):
    comparison = compare(all_candidates, features, labels)
    selected = select_model(comparison)
    configurations = {}
    estimators = {}
    for candidate in all_candidates:
        estimators[candidate.name], configurations[candidate.name] = configure(
            candidate, features, labels,
        )
    estimator = estimators[selected]
    configuration = configurations[selected]
    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": selected,
        "parameters": configuration["parameters"],
        "classifier_configuration": {
            key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in estimator.named_steps["classifier"].get_params().items()
        },
        "preprocessing": {
            "numerical": ["median imputation", "standard scaling"],
            "categorical": ["most frequent imputation", "one-hot with fixed UCI codes"],
        },
        "threshold": configuration["threshold"],
        "features": list(FEATURES),
        "input_schema": input_schema(),
        "dataset_sha256": checksum,
        "split_sha256": split_checksum,
        "training_row_ids": [int(value) for value in features.index],
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "threshold_grid": list(THRESHOLDS),
        "threshold_rule": "maximize inner OOF F2, then precision, then threshold",
        "model_rule": "maximize mean outer F2 within tolerance, then logistic, forest, XGBoost",
        "f2_tolerance": F2_TOLERANCE,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in (
                "numpy", "pandas", "scikit-learn", "xgboost", "joblib", "matplotlib",
            )},
        },
    }
    write_json(output / "validation.json", {
        "outer_comparison": comparison, "full_training_configurations": configurations,
    })
    write_json(output / "selection.json", metadata)
    estimator.fit(features, labels)
    return ModelArtifact(estimator, metadata), comparison, configurations


def evaluate_holdout(artifact, features, labels):
    scores = artifact.pipeline.predict_proba(features)[:, 1]
    threshold = artifact.metadata["threshold"]
    predictions = classify(scores, threshold)
    errors = [
        {"source_row": int(row_id), "actual": int(label), "prediction": int(pred), "score": float(score)}
        for row_id, label, pred, score in zip(features.index, labels, predictions, scores, strict=True)
        if label != pred
    ]
    return metrics(labels, scores, threshold), errors, scores


def run(data, manifest, output):
    from .reporting import save_report

    records, checksum = load_snapshot(data)
    train_ids, test_ids, split_checksum = frozen_split(manifest, records, checksum)
    all_candidates = candidates()
    # An existing directory can represent a started or completed holdout evaluation.
    output.mkdir(parents=True, exist_ok=False)
    features, labels = dataset_rows(records, train_ids)
    artifact, comparison, configs = select_and_fit(
        all_candidates, features, labels, checksum, split_checksum, output,
    )
    artifact_path = output / "model.joblib"
    joblib.dump(artifact, artifact_path)
    reloaded = load_trusted_model(artifact_path)
    first_row = dict(zip(FEATURES, records[train_ids[0] - 1][:-1], strict=True))
    if reloaded.predict(first_row) != artifact.predict(first_row):
        raise RuntimeError("Serialized inference differs on the training probe")

    # Reserve the evaluation before scoring. Do not retry automatically after this point.
    (output / "holdout.started").touch(exist_ok=False)
    test_x, test_y = dataset_rows(records, test_ids)
    result, errors, scores = evaluate_holdout(reloaded, test_x, test_y)
    write_json(output / "heldout.json", {
        "metrics": result,
        "errors": errors,
        "row_ids": test_ids,
        "labels": test_y.tolist(),
        "scores": scores.tolist(),
    })
    save_report(output, reloaded.metadata, comparison, configs, result, errors, test_y, scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed.cleveland.data"))
    parser.add_argument("--manifest", type=Path, default=Path("data/split.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/milestone-2"))
    args = parser.parse_args()
    run(args.data, args.manifest, args.output)


if __name__ == "__main__":
    main()
